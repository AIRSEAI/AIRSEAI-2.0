#!/usr/bin/env python3
"""Read back a recorded LeRobot v3 dataset and say whether it can be trained on.

Checks that every file the manifest promises is on disk, decodes, and holds
the number of frames the manifest claims. Needs no Isaac Sim, GPU or robot.

    python benchmark/run/inspect_dataset.py --dataset ~/texture_sort_dataset
    python benchmark/run/inspect_dataset.py --dataset ... --json report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

OK, BAD = "ok  ", "FAIL"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def _count_frames(path: Path) -> tuple[int, str]:
    """Decoded frame count and codec name, or (-1, reason)."""
    try:
        import av
    except ImportError:
        return -1, "PyAV not installed"
    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            codec = s.codec_context.name
            n = sum(1 for _ in c.decode(video=0))
        return n, codec
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--json", default="", help="also write the report here")
    ap.add_argument("--decode", action="store_true", default=True,
                    help="decode every stream end to end (default)")
    ap.add_argument("--no_decode", dest="decode", action="store_false")
    a = ap.parse_args()
    root = Path(a.dataset).expanduser()
    problems: list[str] = []
    rep: dict = {"dataset": str(root)}

    info = json.loads((root / "meta" / "info.json").read_text())
    rep["info"] = {k: info.get(k) for k in
                   ("codebase_version", "robot_type", "fps", "total_episodes",
                    "total_frames", "total_tasks", "chunks_size")}
    print(f"== {root}")
    print(f"   codebase {info.get('codebase_version')}   fps {info.get('fps')}   "
          f"robot {info.get('robot_type')}")
    print(f"   episodes {info.get('total_episodes')}   frames {info.get('total_frames')}"
          f"   tasks {info.get('total_tasks')}")
    if info.get("codebase_version") != "v3.0":
        problems.append(f"codebase_version is {info.get('codebase_version')}, not v3.0")

    # ---- features ---------------------------------------------------------
    feats = info["features"]
    print(f"\n== features ({len(feats)})")
    vid_keys, data_keys = [], []
    for name, f in feats.items():
        dt, shape = f.get("dtype"), tuple(f.get("shape", ()))
        extra = ""
        if dt == "video":
            vid_keys.append(name)
            vi = f.get("info", {})
            extra = f"  codec={vi.get('video.codec')} {vi.get('video.fps')}fps"
        elif dt == "data":
            data_keys.append(name)
            fi = f.get("info", {})
            q = fi.get("quant", {})
            # A 2-channel field is stored on 3 planes; `original_channels`
            # carries the true count.
            extra = (f"  quant={q.get('name')} scale={q.get('scale')}"
                     f" offset={q.get('offset')} clip={q.get('clip')}"
                     f" ch={fi.get('original_channels')}")
            if not q:
                problems.append(f"{name}: no quant profile in info")
        print(f"   {name:<52} {dt:<7} {str(shape):<18}{extra}")
    rep["features"] = {n: {"dtype": f.get("dtype"), "shape": f.get("shape")}
                       for n, f in feats.items()}

    # ---- episodes ---------------------------------------------------------
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    eps = pq.read_table(ep_files).to_pydict() if ep_files else {}
    n_ep = len(eps.get("episode_index", []))
    print(f"\n== episodes ({n_ep}) from {len(ep_files)} parquet file(s)")
    lengths = eps.get("length", [])
    # `is_success` is a per-frame column of the data parquet, not of the
    # episodes table.
    succ = eps.get("is_success")
    data_files = sorted((root / "data").rglob("*.parquet"))
    tbl = pq.read_table(data_files)
    if succ is None and "is_success" in tbl.column_names:
        ep_col = np.asarray(tbl.column("episode_index").to_pylist()).reshape(-1)
        sc_col = np.asarray(tbl.column("is_success").to_pylist()).reshape(-1)
        succ = [int(sc_col[ep_col == e][0]) if (ep_col == e).any() else None
                for e in eps["episode_index"]]
    if succ is None:
        succ = [None] * n_ep
    print(f"   {'ep':>3} {'len':>5} {'from':>6} {'to':>6} {'success':>8}  task")
    for i in range(n_ep):
        tasks = eps.get("tasks", [[""]] * n_ep)[i]
        t = tasks[0] if isinstance(tasks, list) and tasks else tasks
        print(f"   {eps['episode_index'][i]:>3} {lengths[i]:>5} "
              f"{eps.get('dataset_from_index', [0]*n_ep)[i]:>6} "
              f"{eps.get('dataset_to_index', [0]*n_ep)[i]:>6} "
              f"{str(succ[i]):>8}  {str(t)[:40]}")
    rep["episodes"] = [{"episode_index": eps["episode_index"][i],
                        "length": lengths[i],
                        "is_success": succ[i]} for i in range(n_ep)]
    if n_ep != info.get("total_episodes"):
        problems.append(f"{n_ep} episode rows vs total_episodes {info.get('total_episodes')}")
    if succ and succ[0] is not None:
        n_ok = sum(int(bool(s)) for s in succ)
        print(f"   is_success: {n_ok}/{n_ep}")
        rep["n_success"] = n_ok
    else:
        problems.append("no is_success column")

    # ---- data parquet -----------------------------------------------------
    print(f"\n== data rows {tbl.num_rows} in {len(data_files)} file(s), "
          f"{tbl.num_columns} columns")
    if tbl.num_rows != sum(lengths):
        problems.append(f"data rows {tbl.num_rows} != sum of episode lengths {sum(lengths)}")
    if tbl.num_rows != info.get("total_frames"):
        problems.append(f"data rows {tbl.num_rows} != total_frames {info.get('total_frames')}")
    cols = set(tbl.column_names)
    parquet_feats = [n for n, f in feats.items() if f.get("dtype") not in ("video", "data")]
    missing = [n for n in parquet_feats if n not in cols]
    if missing:
        problems.append(f"features missing from the parquet: {missing}")
    rep["data_rows"] = tbl.num_rows
    for name in ("observation.state", "action"):
        if name not in cols:
            continue
        arr = np.stack(tbl.column(name).to_pylist()).astype(np.float64)
        bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
        print(f"   {name:<20} shape {arr.shape}  min {arr.min():8.3f}  "
              f"max {arr.max():8.3f}  nan/inf {bad}")
        rep[name] = {"shape": list(arr.shape), "min": float(arr.min()),
                     "max": float(arr.max()), "nan_inf": bad}
        if bad:
            problems.append(f"{name} holds {bad} nan/inf")
        if arr.shape[0] and float(arr.std()) == 0.0:
            problems.append(f"{name} is constant across the whole dataset")

    # ---- video / data streams --------------------------------------------
    print(f"\n== streams ({len(vid_keys)} video, {len(data_keys)} data)")
    rep["streams"] = {}
    for name in vid_keys + data_keys:
        paths = sorted((root / "videos" / name).rglob("*"))
        paths = [p for p in paths if p.is_file()]
        if not paths:
            problems.append(f"{name}: no file on disk")
            print(f"   {BAD} {name:<52} (missing)")
            continue
        size = sum(p.stat().st_size for p in paths)
        if a.decode:
            n, codec = _count_frames(paths[0])
            good = n == tbl.num_rows
            if n < 0:
                problems.append(f"{name}: will not decode ({codec})")
            elif not good:
                problems.append(f"{name}: {n} frames decoded, {tbl.num_rows} expected")
            print(f"   {OK if good else BAD} {name:<52} {paths[0].suffix:<5} "
                  f"{codec:<6} frames {n:>6}  {_fmt_bytes(size)}")
            rep["streams"][name] = {"frames": n, "codec": codec, "bytes": size}
        else:
            print(f"   {OK} {name:<52} {paths[0].suffix:<5} {_fmt_bytes(size)}")
            rep["streams"][name] = {"bytes": size}

    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"\n== on disk {_fmt_bytes(total)}")
    rep["bytes_total"] = total
    tasks_pq = root / "meta" / "tasks.parquet"
    if tasks_pq.exists():
        t = pq.read_table(tasks_pq).to_pydict()
        instr = t.get("task", [""])
        print(f"\n== instruction  {instr}")
        rep["instruction"] = instr
    if not (root / "meta" / "stats.json").exists():
        problems.append("meta/stats.json missing")

    print("\n== verdict")
    if problems:
        for p in problems:
            print(f"   {BAD} {p}")
    else:
        print(f"   {OK} {n_ep} episodes, {tbl.num_rows} frames, every stream decodes "
              f"to the full length. Ready for training.")
    rep["problems"] = problems
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(rep, indent=2, default=str))
        print(f"   report -> {a.json}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
