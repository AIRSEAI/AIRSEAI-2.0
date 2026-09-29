#!/usr/bin/env python3
"""Run a task's expert and record a LeRobot v3 dataset.

Use native vector environments where available and combine accepted attempts.

    python benchmark/run/collect_dataset.py --task stack_cube \
        --dataset ~/datasets/sim_stack_v3 --episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from sim.collection_io import add_write_arguments
from sim.tasks.registry import TASKS

REPO = Path(__file__).resolve().parent.parent.parent
RUNNER = REPO / "benchmark" / "tasks" / "run_expert.py"


def collect_parallel(args, passthrough, gpus):
    from lerobot_tactile_ext.merge import merge_recordings

    dataset = Path(args.dataset).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else dataset / "run"
    if dataset.exists() or out.exists():
        raise FileExistsError("Collection output already exists; choose a new output path")
    workers = min(args.num_envs, args.episodes)
    children, logs, assignments = [], [], []
    start = 0
    try:
        for worker in range(workers):
            count = args.episodes // workers + int(worker < args.episodes % workers)
            root = out / "workers" / f"worker_{worker}"
            root.mkdir(parents=True)
            assignment = dict(
                worker=worker, gpu=gpus[worker % len(gpus)], attempt_ids=list(range(start, start + count))
            )
            assignments.append(assignment)
            cmd = [
                sys.executable,
                str(RUNNER),
                "--task",
                args.task,
                "--headless",
                "--enable_cameras",
                "--episodes",
                str(count),
                "--episode_start",
                str(start),
                "--seed",
                str(args.seed),
                "--record_dataset",
                str(root / "raw"),
                "--out",
                str(root),
                *passthrough,
            ]
            if args.robot_usd:
                cmd += ["--robot_usd", args.robot_usd]
            log = (root / "worker.log").open("w")
            logs.append(log)
            child = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(assignment["gpu"])),
                start_new_session=True,
            )
            children.append(child)
            start += count
        (out / "assignments.json").write_text(json.dumps(assignments, indent=2) + "\n")
        while True:
            codes = [p.poll() for p in children]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Collection workers failed: {codes}; logs: {out / 'workers'}")
            if all(code is not None for code in codes):
                break
            time.sleep(0.2)
    finally:
        for p in children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        for log in logs:
            log.close()
        if out.exists():
            (out / "worker_exit_codes.json").write_text(json.dumps([p.returncode for p in children]) + "\n")
    if any(codes):
        raise RuntimeError(f"Collection workers failed: {codes}; logs: {out / 'workers'}")
    reports, inputs, episodes = [], [], []
    saved_offset = 0
    for worker in range(workers):
        root = out / "workers" / f"worker_{worker}"
        report_path = root / f"{args.task}_scripted.json"
        report = json.loads(report_path.read_text())
        reports.append(report)
        if sorted(e["episode"] for e in report["episodes"]) != assignments[worker]["attempt_ids"]:
            raise ValueError(f"Incomplete attempt coverage: {report_path}")
        info_path = root / "raw/meta/info.json"
        saved = json.loads(info_path.read_text())["total_episodes"] if info_path.exists() else 0
        recorded_indices = sorted(
            e["dataset_episode_index"] for e in report["episodes"] if "dataset_episode_index" in e
        )
        if recorded_indices != list(range(saved)):
            raise ValueError(f"Dataset and report episode indices disagree: {report_path}")
        if saved:
            inputs.append(root / "raw")
        for episode in report["episodes"]:
            episode = dict(episode, worker=worker)
            if "dataset_episode_index" in episode:
                episode["dataset_episode_index"] += saved_offset
            episodes.append(episode)
        saved_offset += saved
    combined = dict(
        reports[0],
        episodes=sorted(episodes, key=lambda e: e["episode"]),
        n_episodes=len(episodes),
        n_success=sum(r["n_success"] for r in reports),
        n_clean=sum(r["n_clean"] for r in reports),
        workers=assignments,
    )
    minimum_rate = combined["checks"]["expert_success_rate"]["min_rate"]
    combined["checks"]["expert_success_rate"].update(
        n=combined["n_success"],
        of=len(episodes),
        **{"pass": combined["n_success"] >= minimum_rate * len(episodes)},
    )
    (out / f"{args.task}_scripted.json").write_text(json.dumps(combined, indent=2) + "\n")
    if not inputs:
        raise ValueError(f"No accepted demonstrations; attempts are recorded in {out}")
    merge_recordings(inputs, dataset)
    if not args.keep_shards:
        for source in inputs:
            shutil.rmtree(source)
    if args.previews:
        return subprocess.call(
            [sys.executable, str(REPO / "benchmark/run/render_previews.py"), "--dataset", str(dataset)]
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect a v3 dataset for a task.")
    ap.add_argument("--task", default="stack_cube", choices=sorted(TASKS))
    ap.add_argument(
        "--robot_usd", default="", help="override the asset; default builds/uses the one under assets/"
    )
    ap.add_argument("--dataset", required=True, help="output dataset root")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="", help="report/frames dir (default: <dataset>/run)")
    ap.add_argument(
        "--render",
        action="store_true",
        help="compatibility flag; per-episode review videos are always recorded",
    )
    ap.add_argument(
        "--previews", action="store_true", help="after recording, render preview videos of the tactile data"
    )
    ap.add_argument("--keep-shards", action="store_true", help="Keep raw shards for further selection")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init_states", default="", help="Consume each supplied start once")
    ap.add_argument(
        "--allow_empty", action="store_true", help="Allow completed runs with no accepted episodes"
    )
    add_write_arguments(ap)
    a, passthrough = ap.parse_known_args()
    if a.num_envs < 1 or a.episodes < 1:
        ap.error("num_envs and episodes must be positive")
    if a.seed < 0:
        ap.error("seed must be nonnegative")
    try:
        gpus = [int(v.strip()) for v in a.gpus.split(",")]
    except ValueError:
        ap.error("gpus must be comma-separated integer IDs")
    if any(g < 0 for g in gpus):
        ap.error("GPU IDs must be nonnegative")
    native = bool(TASKS[a.task].manager_cfg)
    if a.write_queue_size < 1:
        ap.error("write_queue_size must be positive")
    if a.episode_write_workers < 1:
        ap.error("episode_write_workers must be positive")
    a.write_mode = a.write_mode or ("background" if native else "sync")
    if not native and a.write_mode != "sync":
        ap.error("Background writing requires a native vector environment")
    passthrough += ["--write_mode", a.write_mode, "--write_queue_size", str(a.write_queue_size)]
    passthrough += ["--episode_write_workers", str(a.episode_write_workers)]
    if native and len(gpus) != 1:
        ap.error("Native vector collection uses one GPU per invocation")
    if a.init_states and not native:
        ap.error("Initial-state collection requires a native vector environment")
    if a.allow_empty and not native:
        ap.error("allow_empty requires a native vector environment")
    reserved = {
        "--episode_start",
        "--init_states",
        "--init_states_out",
        "--init_states_n",
        "--record_dataset",
        "--video",
        "--device",
    }
    if any(value.split("=", 1)[0] in reserved for value in passthrough):
        ap.error("Collection controls attempt IDs, GPU selection and output paths")
    dataset = Path(a.dataset).expanduser().resolve()
    out = Path(a.out).expanduser().resolve() if a.out else dataset / "run"
    if out != dataset / "run" and (dataset == out or dataset in out.parents or out in dataset.parents):
        ap.error("Reports must be <dataset>/run or outside the dataset")
    if dataset.exists() or out.exists():
        ap.error("Collection output already exists; choose a new output path")
    a.dataset, a.out = str(dataset), str(out)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Collection interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    if not native:
        return collect_parallel(a, passthrough, gpus)

    out = a.out
    cmd = [
        sys.executable,
        str(RUNNER),
        "--task",
        a.task,
        "--headless",
        "--enable_cameras",
        *(["--robot_usd", a.robot_usd] if a.robot_usd else []),
        "--episodes",
        str(a.episodes),
        "--seed",
        str(a.seed),
        *(["--num_envs", str(a.num_envs)] if native else []),
        *(["--init_states", a.init_states] if a.init_states else []),
        *(["--allow_empty"] if a.allow_empty else []),
        "--record_dataset",
        a.dataset,
        "--out",
        out,
        *passthrough,
    ]
    print("[collect_dataset]", " ".join(cmd), flush=True)
    child = subprocess.Popen(
        cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[0])), start_new_session=True
    )
    try:
        rc = child.wait()
    finally:
        if child.poll() is None:
            try:
                if native:
                    child.send_signal(signal.SIGTERM)
                else:
                    os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                # Native collection drains its accepted writes before exiting.
                child.wait(timeout=None if native else 30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
    if rc == 0 and not a.allow_empty and not (dataset / "meta/info.json").is_file():
        raise RuntimeError(f"No accepted demonstrations; see {out}")
    if rc == 0 and not a.keep_shards and (dataset / "meta/info.json").is_file():
        for shard in sorted(Path(out).glob("attempts/*/raw")):
            shutil.rmtree(shard)
    if rc == 0 and a.previews and (dataset / "meta/info.json").is_file():
        pcmd = [sys.executable, str(REPO / "benchmark/run/render_previews.py"), "--dataset", a.dataset]
        print("[collect_dataset]", " ".join(pcmd), flush=True)
        rc = subprocess.call(pcmd)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
