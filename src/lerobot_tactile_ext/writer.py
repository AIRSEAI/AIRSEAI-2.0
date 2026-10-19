# SPDX-License-Identifier: Apache-2.0
"""Assemble a LeRobot **v3.0** dataset with tactile ``data`` features.

Writes video features to mp4, tactile ``data`` features (float mm, quantised
here against a named profile) to ffv1/mkv, and vector features to parquet,
one chunk/file per feature with every episode appended. Each data feature's
quantisation profile is recorded in its `info` in meta/info.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from tactile_codec import QuantProfile, quantize

from .codec import DataStream, Mp4Stream

__all__ = ["LeRobotTactileWriter"]

CODEBASE_VERSION = "v3.0"
_QUANTS = ("q01", "q10", "q50", "q90", "q99")
_QVALS = (0.01, 0.10, 0.50, 0.90, 0.99)
_IMG_STATS_SAMPLES = 30  # frames sampled per episode for image stats


class _DiskFrames:
    """An indexed frame sequence whose payload stays on disk."""

    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True)
        self.count = 0

    def append(self, value):
        np.save(self.root / f"{self.count:08d}.npy", value, allow_pickle=False)
        self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        return np.load(self.root / f"{index:08d}.npy", allow_pickle=False)


def _vector_stats(arr: np.ndarray) -> dict:
    """Stats block for parquet-resident features; arr is (N,) or (N, D)."""
    a = arr.reshape(arr.shape[0], -1).astype(np.float64)
    qs = np.quantile(a, _QVALS, axis=0)
    return {
        "min": a.min(0),
        "max": a.max(0),
        "mean": a.mean(0),
        "std": a.std(0),
        "count": np.array([a.shape[0]]),
        **{name: qs[i] for i, name in enumerate(_QUANTS)},
    }


def _image_stats(frames: list[np.ndarray]) -> dict:
    """Per-channel stats over sampled frames, in [0,1], shaped (3,1,1)."""
    idx = np.linspace(0, len(frames) - 1, min(_IMG_STATS_SAMPLES, len(frames))).astype(int)
    px = np.stack([frames[i] for i in idx]).astype(np.float64) / 255.0
    ch = px.reshape(-1, px.shape[-1])  # (n_px, 3)
    qs = np.quantile(ch, _QVALS, axis=0)
    out = {
        "min": ch.min(0),
        "max": ch.max(0),
        "mean": ch.mean(0),
        "std": ch.std(0),
        "count": np.array([len(idx)]),
    }
    out.update({name: qs[i] for i, name in enumerate(_QUANTS)})
    return {k: v.reshape(3, 1, 1) if v.size == 3 else v for k, v in out.items()}


class LeRobotTactileWriter:
    """One writer, four feature kinds.

    features: dict key -> spec, where spec is one of
      {"dtype": "video", "shape": (h, w, 3)}
      {"dtype": "data",  "shape": (h, w, c), "profile": QuantProfile}
        c is the *logical* channel count; 2-channel data rides 3 planes on
        disk and is declared shape (h, w, 3) + original_channels 2, matching
        the real recordings.
      {"dtype": "float32", "shape": (n,), "names": [...]}   (names optional)
      {"dtype": "int64", "shape": (1,)}
        int64 features are per-episode constants (is_success and friends);
        their value is passed to `save_episode`, not `add_frame`.
    """

    def __init__(
        self,
        root: Path | str,
        fps: int,
        features: dict,
        task: str,
        robot_type: str = "openarm_bimanual_tactile",
        *,
        disk_buffer=False,
        durable=False,
    ):
        """`task` is the natural-language instruction stored in
        meta/tasks.parquet and handed to a language-conditioned policy."""
        self.root = Path(root)
        self.fps = fps
        self.features = features
        self.task = task
        self.robot_type = robot_type
        self.disk_buffer = disk_buffer
        self.durable = durable
        self._durable_failed = False
        if durable and not disk_buffer:
            raise ValueError("durable=True requires disk_buffer=True")
        self._episode = 0
        self._total_frames = 0
        self._buffers: dict[str, list] = {}
        self._episode_rows: list[dict] = []
        self._data_cols: dict[str, list] = {}  # aggregated parquet columns
        self._streams: dict[str, Mp4Stream | DataStream] = {}
        self._stream_frames: dict[str, int] = {}  # frames already in the file
        self._saturation_total = 0
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"{self.root} exists and is not empty")
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        self._spool = None
        if durable:
            from .recovery import DurableSpool

            self._spool = DurableSpool(self.root, fps, features, task, robot_type)

    # ------------------------------------------------------------------
    def _frame_keys(self):
        return [k for k, s in self.features.items() if s["dtype"] != "int64"]

    def add_frame(self, frame: dict) -> None:
        if self._spool is not None:
            self._check_durable()
            try:
                self._spool.append(frame)
            except BaseException:
                # A failure after publication but before count metadata is
                # written must never permit finalize to delete that raw frame.
                self._durable_failed = True
                raise
            self._buffers = {key: self._spool.frames(key) for key in self._frame_keys()}
            return
        for key in self._frame_keys():
            if key not in frame:
                raise KeyError(f"frame missing feature {key!r}")
            if key not in self._buffers:
                self._buffers[key] = _DiskFrames(self.root / ".pending" / key) if self.disk_buffer else []
            self._buffers[key].append(np.asarray(frame[key]).copy())

    def discard_episode(self) -> int:
        """Drop the buffered episode without writing it. Returns the number of
        frames thrown away. Nothing has touched the streams yet -- frames only
        reach them in `save_episode` -- so this is a buffer clear, and the
        episode index is not consumed."""
        n = len(next(iter(self._buffers.values()), []))
        if self._spool is not None:
            self._spool.discard()
        self._buffers.clear()
        if self.disk_buffer and not self.durable:
            shutil.rmtree(self.root / ".pending", ignore_errors=True)
        return n

    # ------------------------------------------------------------------
    def _stream_for(self, key: str, spec: dict):
        if key not in self._streams:
            if spec["dtype"] == "video":
                path = self.root / "videos" / key / "chunk-000" / "file-000.mp4"
                self._streams[key] = Mp4Stream(path, self.fps)
            else:
                path = self.root / "videos" / key / "chunk-000" / "file-000.mkv"
                self._streams[key] = DataStream(path, self.fps, channels=spec["shape"][-1])
            self._stream_frames[key] = 0
        return self._streams[key]

    def save_episode(self, scalars: dict | None = None) -> dict:
        """Close out the buffered episode. `scalars` supplies the int64
        per-episode constants declared in features (e.g. is_success)."""
        if self._spool is None:
            return self._save_episode(scalars)
        self._check_durable()
        if not self._buffers:
            raise ValueError("Cannot save an empty episode")
        self._spool.set_decision("save", scalars)
        self._spool.set_status("encoding")
        try:
            result = self._save_episode(scalars)
            self._spool.saved(scalars)
            return result
        except BaseException as error:
            self._durable_failed = True
            try:
                self._spool.set_status("encoding_failed")
            except Exception as journal_error:
                error.add_note(f"Could not persist failure status: {journal_error}")
            raise

    # ------------------------------------------------------------------
    def _aggregate_stats(self) -> dict:
        """Aggregate episode moments and compute numeric quantiles from frames."""
        agg: dict = {}
        feats = set()
        for row in self._episode_rows:
            for k in row:
                if k.startswith("stats/"):
                    feats.add(k.split("/", 2)[1])
        for feat in sorted(feats):
            counts = np.array([r[f"stats/{feat}/count"][0] for r in self._episode_rows], dtype=np.float64)
            w = counts / counts.sum()

            def col(s, feat=feat):
                return np.stack(
                    [np.asarray(r[f"stats/{feat}/{s}"], dtype=np.float64) for r in self._episode_rows]
                )

            mean = (col("mean") * w.reshape(-1, *([1] * (col("mean").ndim - 1)))).sum(0)
            ex2 = col("std") ** 2 + col("mean") ** 2
            var = (ex2 * w.reshape(-1, *([1] * (ex2.ndim - 1)))).sum(0) - mean**2
            entry = {
                "min": col("min").min(0),
                "max": col("max").max(0),
                "mean": mean,
                "std": np.sqrt(np.clip(var, 0, None)),
                "count": np.array([counts.sum()]),
            }
            if feat in self._data_cols:
                data = np.concatenate(self._data_cols[feat])
                data = data.reshape(len(data), -1)
                entry.update(zip(_QUANTS, np.quantile(data, _QVALS, axis=0), strict=True))
            agg[feat] = {k: v.tolist() for k, v in entry.items()}
        return agg

    def abort(self) -> None:
        """Close every encoder while retaining pending data for inspection."""
        errors = []
        if self._spool is not None:
            self._durable_failed = True
            try:
                if not self._spool.closed:
                    self._spool.abort()
            except Exception as error:
                errors.append(error)
        for stream in self._streams.values():
            try:
                stream.abort()
            except Exception as error:
                errors.append(error)
        self._streams.clear()
        if errors:
            raise ExceptionGroup("Encoder cleanup failed", errors)

    def finalize(self) -> dict:
        if self._spool is None:
            return self._finalize()
        self._check_durable()
        if self._buffers:
            raise RuntimeError("Save or discard pending frames before finalize")
        try:
            self._spool.session_status("finalizing")
            result = self._finalize()
            self._spool.complete()
            return result
        except BaseException as error:
            self._durable_failed = True
            try:
                self._spool.session_status("finalization_failed")
            except Exception as journal_error:
                error.add_note(f"Could not persist failure status: {journal_error}")
            raise

    def _finalize(self) -> dict:
        import pyarrow as pa
        import pyarrow.parquet as pq

        stream_stats = {}
        for key, stream in list(self._streams.items()):
            stream_stats[key] = stream.close()
            del self._streams[key]

        if not self._episode_rows:
            result = {"episodes": 0, "frames": 0, "saturated_total": 0, "streams": stream_stats}
            (self.root / "meta" / "empty.json").write_text(json.dumps(result) + "\n")
            return result

        # data/chunk-000/file-000.parquet -- aggregated across episodes
        cols = {}
        order = [k for k, s in self.features.items() if s["dtype"] in ("float32", "int64")] + [
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
        for key in order:
            arr = np.concatenate(self._data_cols[key])
            cols[key] = pa.array(list(arr)) if arr.ndim > 1 else pa.array(arr)
        pdir = self.root / "data" / "chunk-000"
        pdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), pdir / "file-000.parquet")

        # meta/episodes/chunk-000/file-000.parquet
        edir = self.root / "meta" / "episodes" / "chunk-000"
        edir.mkdir(parents=True, exist_ok=True)
        ecols = {}
        for k in self._episode_rows[0]:
            vals = [r[k] for r in self._episode_rows]
            if isinstance(vals[0], np.ndarray):
                ecols[k] = pa.array([v.tolist() for v in vals])
            else:
                ecols[k] = pa.array(vals)
        pq.write_table(pa.table(ecols), edir / "file-000.parquet")

        # meta/tasks.parquet, meta/stats.json
        pq.write_table(
            pa.table({"task_index": pa.array([0]), "task": pa.array([self.task])}),
            self.root / "meta" / "tasks.parquet",
        )
        with open(self.root / "meta" / "stats.json", "w") as f:
            json.dump(self._aggregate_stats(), f)

        # meta/info.json
        feats = {}
        for key, spec in self.features.items():
            shape = list(spec["shape"])
            entry: dict = {"dtype": spec["dtype"], "shape": shape}
            if spec["dtype"] == "video":
                entry["names"] = ["height", "width", "channels"]
                entry["info"] = {
                    "video.height": shape[0],
                    "video.width": shape[1],
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "has_audio": False,
                }
            elif spec["dtype"] == "data":
                orig_c = shape[-1]
                if orig_c == 2:  # on-disk contract: 2ch rides 3 planes
                    entry["shape"] = shape[:-1] + [3]
                entry["names"] = ["height", "width", "channel"]
                entry["info"] = {"original_channels": orig_c, "quant": spec["profile"].to_dict()}
            elif spec["dtype"] == "float32":
                if spec.get("names"):
                    entry["names"] = list(spec["names"])
            feats[key] = entry
        for key, dt in (
            ("timestamp", "float32"),
            ("frame_index", "int64"),
            ("episode_index", "int64"),
            ("index", "int64"),
            ("task_index", "int64"),
        ):
            feats[key] = {"dtype": dt, "shape": [1]}

        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": len(self._episode_rows),
            "total_frames": self._total_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": float(self.fps),
            "splits": {"train": f"0:{len(self._episode_rows)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "data_video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mkv",
            "features": feats,
        }
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        return {
            "episodes": len(self._episode_rows),
            "frames": self._total_frames,
            "saturated_total": self._saturation_total,
            "streams": stream_stats,
        }


    def _save_episode(self, scalars: dict | None = None) -> dict:
        ep = self._episode
        n = len(next(iter(self._buffers.values())))
        row: dict = {
            "episode_index": ep,
            "tasks": [self.task],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": self._total_frames,
            "dataset_to_index": self._total_frames + n,
        }
        stats = {"episode_index": ep, "length": n, "saturated": 0}
        ep_stats: dict[str, dict] = {}

        for key, spec in self.features.items():
            if spec["dtype"] == "int64":
                if scalars is None or key not in scalars:
                    raise KeyError(f"save_episode needs int64 scalar {key!r}")
                col = np.full(n, int(scalars[key]), dtype=np.int64)
                self._data_cols.setdefault(key, []).append(col)
                ep_stats[key] = _vector_stats(col)
                continue
            if spec["dtype"] == "video" or spec["dtype"] == "data":
                stream = self._stream_for(key, spec)
                t0 = self._stream_frames[key]
                if spec["dtype"] == "video":
                    for f in self._buffers[key]:
                        stream.add(f)
                    ep_stats[key] = _image_stats(self._buffers[key])
                else:
                    prof: QuantProfile = spec["profile"]
                    for frame in self._buffers[key]:
                        u16, sat = quantize(frame, prof)
                        stats["saturated"] += sat
                        stream.add(u16)
                self._stream_frames[key] += n
                row[f"videos/{key}/chunk_index"] = 0
                row[f"videos/{key}/file_index"] = 0
                row[f"videos/{key}/from_timestamp"] = t0 / self.fps
                row[f"videos/{key}/to_timestamp"] = (t0 + n) / self.fps
                continue
            # float32 vectors -> aggregated parquet
            arr = np.stack(list(self._buffers[key])).astype(np.float32)
            self._data_cols.setdefault(key, []).append(arr)
            ep_stats[key] = _vector_stats(arr)

        # bookkeeping columns
        book = {
            "timestamp": (np.arange(n) / self.fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep, dtype=np.int64),
            "index": np.arange(self._total_frames, self._total_frames + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
        }
        for k, v in book.items():
            self._data_cols.setdefault(k, []).append(v)
            ep_stats[k] = _vector_stats(v)

        for feat, st in ep_stats.items():
            for sname, val in st.items():
                row[f"stats/{feat}/{sname}"] = np.asarray(val)
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0

        self._episode_rows.append(row)
        self._saturation_total += stats["saturated"]
        self._total_frames += n
        self._buffers = {}
        if self.disk_buffer and not self.durable:
            shutil.rmtree(self.root / ".pending")
        self._episode += 1
        return stats


    def _check_durable(self):
        if self._durable_failed:
            raise RuntimeError("Durable writer failed; abort and recover into a new directory")
        if self._spool is not None:
            self._spool.check_open()


    def set_recovery_decision(self, decision: str, scalars: dict | None = None) -> None:
        """Persist ``save`` or irreversible ``discard`` before GPU finish work.

        A save decision is intent, never evidence of a complete/successful episode.
        Discarded frames cannot become recovery candidates, even after a crash.
        """
        if self._spool is not None:
            self._spool.set_decision(decision, scalars)


    def set_recovery_context(self, **context) -> None:
        """Persist JSON context before GPU capture/finish; durable mode only."""
        if self._spool is not None:
            self._spool.set_context(context)
