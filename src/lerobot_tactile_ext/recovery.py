"""Crash-safe raw frame spool and offline, review-only recording recovery.

No Isaac/Kit imports. Publication is a directory rename after every modality
and the commit manifest have been flushed. Only consecutive published frames
are replayed. The original spool is never changed by recovery.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

__all__ = ["recover_recording"]


def _sync_dir(path):
    # Windows has no portable directory fsync; file fsync + atomic rename still
    # protects against process death. POSIX additionally flushes directory links.
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write_json(path, document):
    path = Path(path)
    payload = json.dumps(document, indent=2, allow_nan=False).encode() + b"\n"
    temp = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
        _sync_dir(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _features_to_json(features):
    result = {}
    for key, spec in features.items():
        entry = dict(spec)
        if "profile" in entry:
            profile = entry.pop("profile")
            if not is_dataclass(profile):
                raise TypeError("Durable tactile profiles must be QuantProfile dataclasses")
            # Constructor fields, rather than display-oriented to_dict(), allow
            # public QuantProfile(**fields) reconstruction without private APIs.
            from dataclasses import fields

            values = asdict(profile)
            entry["profile_constructor"] = {f.name: values[f.name] for f in fields(profile) if f.init}
        result[key] = entry
    # Fail before accepting frames if context/features cannot be represented.
    return json.loads(json.dumps(result, allow_nan=False))


def _features_from_json(features):
    from tactile_codec import QuantProfile

    result = {}
    for key, value in features.items():
        spec = dict(value)
        spec["shape"] = tuple(spec["shape"])
        if "profile_constructor" in spec:
            spec["profile"] = QuantProfile(**spec.pop("profile_constructor"))
        result[key] = spec
    return result


class _CommittedFrames:
    def __init__(self, directory, count, feature_id):
        self.directory, self.count, self.feature_id = directory, count, feature_id

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        return np.load(self.directory / f"{index:08d}" / f"{self.feature_id}.npy", allow_pickle=False)


class DurableSpool:
    def __init__(self, root, fps, features, task, robot_type):
        self.root = Path(root) / ".recovery"
        self.root.mkdir()
        self.keys = [k for k, s in features.items() if s["dtype"] != "int64"]
        if not self.keys:
            raise ValueError("Durable recording needs at least one per-frame feature")
        self.features = features
        self.session = {
            "version": 1, "status": "recording", "fps": fps, "task": task,
            "robot_type": robot_type, "features": _features_to_json(features),
            "frame_keys": self.keys,
        }
        _write_json(self.root / "session.json", self.session)
        self.ordinal = 0
        self.episode = None
        self.count = 0
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("Durable recording is already closed")

    def _ensure_episode(self):
        self.check_open()
        if self.episode is None:
            self.directory = self.root / f"episode-{self.ordinal:06d}"
            self.directory.mkdir()
            self.frame_dir = self.directory / "frames"
            self.frame_dir.mkdir()
            self.episode = {"ordinal": self.ordinal, "status": "recording", "count": 0,
                            "decision": None, "context": {}}
            self.count = 0
            self._persist()
            _sync_dir(self.root)

    def _persist(self):
        _write_json(self.directory / "episode.json", self.episode)

    def set_context(self, context):
        self._ensure_episode()
        updated = {**self.episode["context"], **context}
        # Serialize before changing in-memory state.
        json.dumps(updated, allow_nan=False)
        self.episode["context"] = updated
        self._persist()

    def set_decision(self, decision, scalars=None):
        if decision not in ("save", "discard"):
            raise ValueError("Recovery decision must be save or discard")
        self._ensure_episode()
        if self.episode["decision"] == "discard" and decision != "discard":
            raise RuntimeError("Discard is irreversible")
        self.episode.update(decision=decision, scalars=scalars)
        self._persist()

    def set_status(self, status):
        self._ensure_episode()
        self.episode["status"] = status
        self._persist()

    def append(self, frame):
        self._ensure_episode()
        if self.episode["decision"] == "discard":
            raise RuntimeError("Discarded episode must be cleared before recording")
        values = {}
        for key in self.keys:
            if key not in frame:
                raise KeyError(f"frame missing feature {key!r}")
            value = np.asarray(frame[key]).copy()
            if value.dtype.hasobject or tuple(value.shape) != tuple(self.features[key]["shape"]):
                raise ValueError(f"Invalid shape or object dtype for {key!r}")
            values[key] = value
        staging = Path(tempfile.mkdtemp(prefix=".frame-", dir=self.frame_dir))
        for i, key in enumerate(self.keys):
            with open(staging / f"{i}.npy", "xb") as output:
                np.save(output, values[key], allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
        _write_json(staging / "commit.json", {"index": self.count, "keys": self.keys})
        _sync_dir(staging)
        # Never overwrite a published frame. Destination is an unused directory.
        staging.rename(self.frame_dir / f"{self.count:08d}")
        _sync_dir(self.frame_dir)
        self.count += 1
        self.episode["count"] = self.count
        self._persist()

    def frames(self, key):
        return _CommittedFrames(self.frame_dir, self.count, self.keys.index(key))

    def _advance(self):
        self.ordinal += 1
        self.episode = None
        self.count = 0

    def discard(self):
        self.set_decision("discard")  # Tombstone is durable BEFORE raw deletion.
        self.set_status("discarded")
        shutil.rmtree(self.frame_dir)
        self._advance()

    def saved(self, scalars):
        self.episode.update(status="saved", scalars=scalars)
        self._persist()
        self._advance()

    def session_status(self, status):
        self.session["status"] = status
        _write_json(self.root / "session.json", self.session)

    def abort(self):
        self.session_status("aborted")
        self.closed = True

    def complete(self):
        # Flush all published dataset output before deleting the only raw copy.
        for path in self.root.parent.rglob("*"):
            if path.is_file() and self.root not in path.parents:
                with open(path, "r+b") as source:
                    os.fsync(source.fileno())
        self.session_status("finalized")
        self.closed = True
        shutil.rmtree(self.root)
        _sync_dir(self.root.parent)


def _candidate_episodes(spool, session):
    keys = session["frame_keys"]
    for directory in sorted(spool.glob("episode-*")):
        try:
            state = _read_json(directory / "episode.json")
        except (OSError, ValueError):
            # Process death while creating the next episode must not hide
            # earlier committed episodes. Unknown decisions are never replayed.
            continue
        if state.get("decision") == "discard" or state.get("status") == "discarded":
            continue
        count = 0
        while True:
            frame_dir = directory / "frames" / f"{count:08d}"
            if not frame_dir.is_dir():
                break
            try:
                commit = _read_json(frame_dir / "commit.json")
                if commit != {"index": count, "keys": keys}:
                    break
                for i, key in enumerate(keys):
                    value = np.load(frame_dir / f"{i}.npy", allow_pickle=False)
                    if list(value.shape) != session["features"][key]["shape"]:
                        raise ValueError("Frame shape differs from schema")
            except (OSError, ValueError, EOFError):
                break
            count += 1
        if count:
            yield directory, state, count


def recover_recording(source: Path | str, destination: Path | str) -> Path:
    """Replay committed raw frames into a NEW, review-only LeRobot dataset.

    All episodes are conservatively labelled is_success=0, interrupted=1,
    needs_review=1; train splits are empty. Human review is required before any
    training use. An existing destination is always refused, including retries.
    On failure the source spool remains untouched and the unpublished staging
    directory is removed. Use only after the original writer process has exited.
    """
    from .writer import LeRobotTactileWriter

    source, destination = Path(source).resolve(), Path(destination).resolve()
    spool = source if source.name == ".recovery" else source / ".recovery"
    if destination.exists():
        raise FileExistsError(destination)
    recording_root = spool.parent
    if (destination == recording_root or recording_root in destination.parents
            or destination in recording_root.parents):
        raise ValueError("Recovery destination must be separate from source")
    session = _read_json(spool / "session.json")
    if session.get("version") != 1 or session.get("status") == "finalized":
        raise ValueError("No unfinished supported recording to recover")
    episodes = list(_candidate_episodes(spool, session))
    if not episodes:
        raise ValueError("No committed non-discarded frames to recover")
    features = _features_from_json(session["features"])
    for key in ("is_success", "interrupted", "needs_review"):
        features[key] = {"dtype": "int64", "shape": (1,)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-recovery-", dir=destination.parent))
    writer = None
    try:
        writer = LeRobotTactileWriter(staging, session["fps"], features, session["task"],
                                     session["robot_type"], disk_buffer=True)
        annotations = []
        for directory, state, count in episodes:
            for index in range(count):
                writer.add_frame({key: np.load(directory / "frames" / f"{index:08d}" / f"{i}.npy",
                                                allow_pickle=False)
                                  for i, key in enumerate(session["frame_keys"])})
            scalars = {key: 0 for key, spec in features.items() if spec["dtype"] == "int64"}
            scalars.update(is_success=0, interrupted=1, needs_review=1)
            writer.save_episode(scalars)
            annotations.append({"episode_index": len(annotations), "source_episode": state,
                                "committed_frames": count, "interrupted": True,
                                "needs_review": True, "is_success": 0})
        writer.finalize()
        info_path = staging / "meta" / "info.json"
        info = _read_json(info_path)
        info.update(splits={}, recovery={"interrupted": True, "needs_review": True,
                                        "training_eligible": False})
        _write_json(info_path, info)
        _write_json(staging / "meta" / "recovery.json", {"source": str(source),
                    "interrupted": True, "needs_review": True, "training_eligible": False,
                    "episodes": annotations})
        # mkdir is the exclusive publication reservation; never replace a prior
        # destination on POSIX where rename could replace an empty directory.
        destination.mkdir()
        try:
            for path in sorted(staging.iterdir(), key=lambda p: p.name == "meta"):
                path.rename(destination / path.name)
            staging.rmdir()
        except BaseException:
            shutil.rmtree(destination)
            raise
        return destination
    except BaseException:
        if writer is not None:
            try:
                writer.abort()
            except BaseException:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise
