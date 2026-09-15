#!/usr/bin/env python3
"""Run a task's expert and record a LeRobot v3 dataset.

Accepts any task in `sim.tasks.registry.TASKS` and runs its expert through
`benchmark/tasks/run_expert.py` in a subprocess, optionally rendering the
tactile preview streams afterwards.

    python benchmark/run/collect_dataset.py --task stack_cube \
        --dataset ~/datasets/sim_stack_v3 --episodes 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sim.tasks.registry import TASKS

REPO = Path(__file__).resolve().parent.parent.parent
RUNNER = REPO / "benchmark" / "tasks" / "run_expert.py"


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect a v3 dataset for a task.")
    ap.add_argument("--task", default="stack_cube", choices=sorted(TASKS))
    ap.add_argument("--robot_usd", default="",
                    help="override the asset; default builds/uses the "
                         "one under assets/")
    ap.add_argument("--dataset", required=True, help="output dataset root")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="", help="report/frames dir "
                    "(default: <dataset>_run)")
    ap.add_argument("--render", action="store_true",
                    help="also write a video to <out>/<task>.mp4")
    ap.add_argument("--previews", action="store_true",
                    help="after recording, render preview videos of the tactile data")
    a, passthrough = ap.parse_known_args()

    out = a.out or (os.path.expanduser(a.dataset).rstrip("/") + "_run")
    inline_video = ["--video", os.path.join(out, f"{a.task}.mp4")] if a.render else []
    cmd = [sys.executable, str(RUNNER), "--task", a.task,
           "--headless", "--enable_cameras",
           *(["--robot_usd", a.robot_usd] if a.robot_usd else []),
           "--episodes", str(a.episodes),
           "--record_dataset", a.dataset,
           "--out", out,
           *inline_video, *passthrough]
    print("[collect_dataset]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc == 0 and a.previews:
        pcmd = [sys.executable, str(REPO / "benchmark/run/render_previews.py"),
                "--dataset", a.dataset]
        print("[collect_dataset]", " ".join(pcmd), flush=True)
        rc = subprocess.call(pcmd)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
