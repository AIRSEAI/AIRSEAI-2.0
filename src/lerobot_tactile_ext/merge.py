# SPDX-License-Identifier: Apache-2.0
"""Combine compatible raw recordings without re-encoding their media."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def inspect_recording(root: Path) -> dict:
    root = root.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info["total_episodes"] <= 0:
        raise ValueError(f"No saved episodes: {root}")
    tables = sorted((root / "data").glob("chunk-*/*.parquet"))
    episodes = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if len(tables) != 1 or len(episodes) != 1:
        raise ValueError(f"Expected a single raw recording stream per feature: {root}")
    table = pq.read_table(tables[0])
    rows = pq.read_table(episodes[0]).to_pylist()
    if len(table) != info["total_frames"] or len(rows) != info["total_episodes"]:
        raise ValueError(f"Frame or episode count mismatch: {root}")
    if table["index"].to_pylist() != list(range(len(table))):
        raise ValueError(f"Non-contiguous frame indices: {root}")
    media = {}
    for key, feature in info["features"].items():
        if feature["dtype"] not in ("video", "data"):
            continue
        suffix = ".mp4" if feature["dtype"] == "video" else ".mkv"
        path = root / "videos" / key / "chunk-000" / ("file-000" + suffix)
        if not path.is_file():
            raise FileNotFoundError(path)
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,width,height,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if int(stream["nb_read_frames"]) != len(table):
            raise ValueError(f"Media frame count mismatch: {path}")
        if [stream["height"], stream["width"]] != feature["shape"][:2]:
            raise ValueError(f"Media dimensions mismatch: {path}")
        numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
        if denominator == 0 or not np.isclose(numerator / denominator, info["fps"]):
            raise ValueError(f"Media frame rate mismatch: {path}")
        media[key] = path
    boundary = 0
    for i, row in enumerate(rows):
        if (
            row["episode_index"] != i
            or row["length"] <= 0
            or row["dataset_from_index"] != boundary
            or row["dataset_to_index"] - row["dataset_from_index"] != row["length"]
        ):
            raise ValueError(f"Invalid episode boundaries: {root}")
        indices = np.asarray(table["episode_index"])[row["dataset_from_index"] : row["dataset_to_index"]]
        if len(indices) != row["length"] or not np.all(indices == i):
            raise ValueError(f"Episode rows do not match metadata: {root}")
        frames = table.slice(row["dataset_from_index"], row["length"])
        if frames["frame_index"].to_pylist() != list(range(row["length"])):
            raise ValueError(f"Invalid frame indices: {root}, episode {i}")
        if not np.allclose(
            np.asarray(frames["timestamp"]), np.arange(row["length"]) / info["fps"], atol=1e-6
        ):
            raise ValueError(f"Invalid frame timestamps: {root}, episode {i}")
        boundary = row["dataset_to_index"]
    if boundary != len(table):
        raise ValueError(f"Unassigned dataset frames: {root}")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    if len(tasks) != 1 or tasks[0]["task_index"] != 0:
        raise ValueError(f"Expected one task per raw recording: {root}")
    digest = hashlib.sha256()
    annotations_path = root / "meta/episode_annotations.json"
    annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else []
    annotated_ids = [a["episode_index"] for a in annotations]
    if len(set(annotated_ids)) != len(annotated_ids) or any(i not in range(len(rows)) for i in annotated_ids):
        raise ValueError(f"Invalid annotation episode indices: {root}")
    if annotations_path.exists():
        digest.update(annotations_path.read_bytes())
    for path in [root / "meta/info.json", tables[0], episodes[0], root / "meta/tasks.parquet"]:
        digest.update(path.read_bytes())
    for path in media.values():
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
    return dict(
        root=root,
        info=info,
        table=table,
        episodes=rows,
        tasks=tasks,
        media=media,
        annotations=annotations,
        digest=digest.hexdigest(),
    )


def _stats(episodes: list[dict], table: pa.Table) -> dict:
    """Aggregate moments and compute vector quantiles from all stored frames."""
    result = {}
    features = {k.split("/")[1] for k in episodes[0] if k.startswith("stats/")}
    for key in sorted(features):

        def values(stat, key=key):
            return np.asarray([r[f"stats/{key}/{stat}"] for r in episodes], dtype=float)

        counts = values("count").reshape(-1)
        means = values("mean")
        weight = (counts / counts.sum()).reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (means * weight).sum(0)
        variance = ((values("std") ** 2 + means**2) * weight).sum(0) - mean**2
        stats = dict(
            min=values("min").min(0),
            max=values("max").max(0),
            mean=mean,
            std=np.sqrt(np.maximum(variance, 0)),
            count=np.array([counts.sum()]),
        )
        if key in table.column_names:
            data = np.asarray(table[key].to_pylist(), dtype=float).reshape(len(table), -1)
            quantiles = np.quantile(data, (0.01, 0.10, 0.50, 0.90, 0.99), axis=0)
            stats.update(zip(("q01", "q10", "q50", "q90", "q99"), quantiles, strict=True))
        result[key] = {name: value.tolist() for name, value in stats.items()}
    return result


def _drop_previews(recording: dict) -> None:
    features = recording["info"]["features"]
    previews = {key for key, spec in features.items() if spec.get("info", {}).get("preview_of")}
    for key in previews:
        source = features[key]["info"]["preview_of"]
        if source not in features or source in previews or features[key]["dtype"] != "video":
            raise ValueError(f"Invalid preview source: {key} -> {source}")
        recording["media"].pop(key)
    recording["info"]["features"] = {key: spec for key, spec in features.items() if key not in previews}
    recording["table"] = recording["table"].drop_columns(
        [key for key in previews if key in recording["table"].column_names]
    )
    recording["episodes"] = [
        {
            key: value
            for key, value in row.items()
            if not (key.startswith(("videos/", "stats/")) and key.split("/")[1] in previews)
        }
        for row in recording["episodes"]
    ]


def merge_recordings(inputs: list[str | Path], output: str | Path, *, drop_previews: bool = False) -> Path:
    roots = [Path(p).expanduser().resolve() for p in inputs]
    dst = Path(output).expanduser().resolve()
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Provide distinct input recordings")
    if any(
        dst == p or p in dst.parents or (dst in p.parents and dst / "run" not in p.parents) for p in roots
    ):
        raise ValueError("Output and input directories must not overlap")
    if dst.exists() and (
        not dst.is_dir()
        or set(dst.iterdir()) != {dst / "run"}
        or not (dst / "run").is_dir()
        or (dst / "run").is_symlink()
    ):
        raise FileExistsError(dst)
    recordings = [inspect_recording(p) for p in roots]
    if drop_previews:
        for recording in recordings:
            _drop_previews(recording)
    first = recordings[0]
    for item in recordings[1:]:
        for key in ("features", "fps", "robot_type", "codebase_version"):
            if item["info"][key] != first["info"][key]:
                raise ValueError(f"Incompatible {key}: {item['root']}")
        if item["tasks"] != first["tasks"]:
            raise ValueError(f"Incompatible task: {item['root']}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}-", dir=dst.parent))
    try:
        tables, rows, sources, annotations = [], [], [], []
        frame_offset = episode_offset = 0
        fps = first["info"]["fps"]
        for source_index, item in enumerate(recordings):
            annotations.extend(
                dict(row, episode_index=row["episode_index"] + episode_offset) for row in item["annotations"]
            )
            table = item["table"]
            for key, offset in (("index", frame_offset), ("episode_index", episode_offset)):
                values = np.asarray(table[key]) + offset
                table = table.set_column(table.schema.get_field_index(key), key, pa.array(values))
            tables.append(table)
            for original in item["episodes"]:
                row = dict(original)
                old_episode = row["episode_index"]
                row["episode_index"] += episode_offset
                for key in ("dataset_from_index", "dataset_to_index"):
                    row[key] += frame_offset
                for key in list(row):
                    if key.startswith("videos/") and key.endswith(("/from_timestamp", "/to_timestamp")):
                        row[key] += frame_offset / fps
                for key, offset in (("index", frame_offset), ("episode_index", episode_offset)):
                    for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
                        k = f"stats/{key}/{stat}"
                        if k in row:
                            row[k] = (np.asarray(row[k]) + offset).tolist()
                rows.append(row)
                sources.append(
                    dict(
                        episode_index=row["episode_index"],
                        source=f"source_{source_index:03d}",
                        source_episode_index=old_episode,
                        source_sha256=item["digest"],
                    )
                )
                review = item["root"] / "review/third_person" / f"episode_{old_episode:06d}.mp4"
                if review.exists():
                    target = staging / "review/third_person" / f"episode_{row['episode_index']:06d}.mp4"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(review, target)
            frame_offset += len(table)
            episode_offset += len(item["episodes"])
        for key, path in first["media"].items():
            target = staging / path.relative_to(first["root"])
            target.parent.mkdir(parents=True, exist_ok=True)
            listing = staging / "concat.txt"
            listing.write_text(
                "".join(
                    "file '" + str(item["media"][key]).replace("'", "'\\''") + "'\n" for item in recordings
                )
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listing),
                    "-map",
                    "0:v:0",
                    "-c",
                    "copy",
                    str(target),
                ],
                check=True,
            )
            listing.unlink()
        (staging / "data/chunk-000").mkdir(parents=True)
        (staging / "meta/episodes/chunk-000").mkdir(parents=True)
        combined = pa.concat_tables(tables)
        pq.write_table(combined, staging / "data/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(rows), staging / "meta/episodes/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(first["tasks"]), staging / "meta/tasks.parquet")
        info = dict(
            first["info"],
            total_frames=frame_offset,
            total_episodes=episode_offset,
            splits={"train": f"0:{episode_offset}"},
        )
        for name, value in [("info", info), ("stats", _stats(rows, combined)), ("source_episodes", sources)]:
            (staging / "meta" / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
        if annotations:
            (staging / "meta/episode_annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
        inspect_recording(staging)
        if dst.exists():
            # Publish metadata last so incomplete output cannot be read as a dataset.
            for path in sorted(staging.iterdir(), key=lambda path: path.name == "meta"):
                path.rename(dst / path.name)
            staging.rmdir()
        else:
            staging.rename(dst)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return dst
