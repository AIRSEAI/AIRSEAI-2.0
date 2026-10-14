"""Save detached episodes with bounded background work."""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "EpisodeWrite",
    "EpisodeWriteQueue",
    "add_write_arguments",
    "write_json",
]


def add_write_arguments(parser):
    parser.add_argument(
        "--write_mode",
        choices=("sync", "background"),
        default=None,
        help="Episode saving mode; default background for native environments",
    )
    parser.add_argument(
        "--write_queue_size",
        type=int,
        default=4,
        help="Maximum outstanding episode saves, including the running save",
    )
    parser.add_argument(
        "--episode_write_workers",
        type=int,
        default=2,
        help="Concurrent background episode saves; ignored in sync mode",
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_plain) + "\n")
    temporary.replace(path)


@dataclass
class EpisodeWrite:
    folder: Path
    row: dict
    terminal: dict
    trajectory: dict
    observation: dict | None
    recorder: object
    keep: bool
    intervention: bool

    def write(self):
        row = dict(self.row, saved=False, write_status="writing")
        try:
            write_json(self.folder / "terminal_state.json", self.terminal)
            write_json(self.folder / "trajectory.json", self.trajectory)
            if row.get("recording_mode") == "xr_state_action_only":
                row["trajectory_saved"] = bool(self.trajectory.get("frames"))
            if self.observation is not None:
                np.savez_compressed(self.folder / "terminal_observation.npz", **self.observation)
            if self.recorder:
                if self.keep:
                    self.recorder.save_episode(0, row["criterion_success"], is_intervention=self.intervention)
                else:
                    self.recorder.discard_episode()
                self.recorder.finalize()
                if self.keep:
                    annotation = dict(row, saved=True, write_status="complete", episode_index=0)
                    write_json(self.recorder.root / "meta/episode_annotations.json", [annotation])
                    row["saved"] = True
            row["write_status"] = "complete"
            write_json(self.folder / "result.json", row)
        except Exception as error:
            row.update(saved=False, write_status="failed", error=f"{type(error).__name__}: {error}")
            if self.recorder:
                try:
                    self.recorder.abort()
                except Exception as cleanup_error:
                    row["cleanup_error"] = str(cleanup_error)
            try:
                write_json(self.folder / "result.json", row)
            except Exception as report_error:
                row["report_error"] = str(report_error)
        return row


class EpisodeWriteQueue:
    """Own submitted episodes until their results are consumed by the caller."""

    def __init__(self, mode="background", capacity=4, workers=2):
        if mode not in ("sync", "background") or capacity < 1 or workers < 1:
            raise ValueError("Invalid episode writing mode, capacity, or worker count")
        self.capacity = capacity
        self.pending = deque()
        self.executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="episode-writer")
            if mode == "background"
            else None
        )
        self.closed = False

    @property
    def full(self):
        return len(self.pending) >= self.capacity

    @property
    def attempt_ids(self):
        return [attempt_id for attempt_id, _ in self.pending]

    def submit(self, job):
        if self.closed:
            raise RuntimeError("Episode writer is closed")
        if self.full:
            raise RuntimeError("Collect a completed write before submitting another episode")
        if self.executor:
            future = self.executor.submit(job.write)
        else:
            future = Future()
            future.set_result(job.write())
        self.pending.append((job.row["attempt_id"], future))

    def collect(self, *, block=False, drain=False):
        rows = []
        while self.pending:
            _, future = self.pending[0]
            if not (drain or block or future.done()):
                break
            rows.append(future.result())
            self.pending.popleft()
            block = False
        return rows

    def close(self):
        self.closed = True
        if self.executor:
            self.executor.shutdown(wait=True)
