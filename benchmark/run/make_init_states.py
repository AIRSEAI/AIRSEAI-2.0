#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate frozen starts in vector environments, optionally checked by an expert."""

import argparse
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.collection_io import add_write_arguments
from sim.runtime import install_error_handler
from sim.tasks.registry import TASKS, load_task


def main():
    install_error_handler()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, choices=[key for key, entry in TASKS.items() if entry.manager_cfg]
    )
    parser.add_argument("--out", required=True, help="Frozen initial-state JSON output")
    parser.add_argument("--report-dir", default="", help="Candidate reports; default beside output JSON")
    parser.add_argument("--count", type=int, default=100, help="Number of accepted states, not attempts")
    parser.add_argument("--max-candidates", type=int, default=1000, help="Maximum number of candidate starts")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exclude", nargs="*", default=[], help="Frozen state files that must not overlap")
    parser.add_argument("--video", default="", help="Fixed-camera mosaic of candidate attempts")
    parser.add_argument(
        "--video-size", type=int, nargs=2, default=None, help="Review video width and height in pixels"
    )
    parser.add_argument("--robot_usd", default="")
    parser.add_argument("--grid", type=int, nargs=2, default=(288, 384))
    add_write_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    known, _ = parser.parse_known_args()
    if min(known.count, known.num_envs, known.max_candidates) < 1 or known.seed < 0:
        parser.error("counts and num_envs must be positive; seed must be nonnegative")
    if known.video_size is None:
        known.video_size = (640, 480) if known.num_envs >= 8 else (960, 720)
        parser.set_defaults(video_size=known.video_size)
    if min(known.video_size) < 2 or any(value % 2 for value in known.video_size):
        parser.error("video-size must contain positive even dimensions")
    if known.max_candidates < known.count:
        parser.error("max-candidates must be at least count")
    if known.write_queue_size < 1:
        parser.error("write_queue_size must be positive")
    destination = Path(known.out).expanduser().resolve()
    if destination.exists() or destination.with_suffix(".partial.json").exists():
        parser.error("out already exists; choose a new path")
    known.enable_cameras = True
    launcher = AppLauncher(known)
    try:
        # Task and simulation imports require Kit initialization.
        from sim.state_generation import generate_states

        task, pins = load_task(known.task)
        for key, knob in task.knobs.items():
            parser.add_argument(f"--{key}", type=knob.type, default=knob.default, help=knob.help)
        args = parser.parse_args()
        knobs = {key: getattr(args, key) for key in task.knobs} | pins
        args.states_out = str(destination)
        args.out = args.report_dir or str(destination.with_suffix("")) + "_run"
        args.episodes = args.max_candidates
        args.init_states = ""
        args.init_states_out = ""
        args.record_dataset = ""
        args.allow_empty = True
        return generate_states(args, task, knobs, launcher.app)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
