#!/usr/bin/env python3
"""Record keyboard, VR or scripted demonstrations with success labels.

    python benchmark/teleop/run_teleop.py --task stack_cube --episodes 5 \
        --dataset ~/stack_cube_teleop

"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.collection_io import add_write_arguments
from sim.runtime import install_error_handler
from sim.tasks.registry import TASKS as TASK_IDS

install_error_handler()

parser = argparse.ArgumentParser(description="Teleoperated demonstrations.")
parser.add_argument("--task", required=True, choices=sorted(TASK_IDS))
parser.add_argument(
    "--teleop_device",
    default="keyboard",
    choices=("keyboard", "vr", "scripted"),
    help="keyboard requires a viewer; scripted runs automatic controls",
)
parser.add_argument("--dataset", default="", help="write the recording here")
parser.add_argument("--out", default="", help="report dir, default <dataset>_run")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--num_envs", type=int, choices=(1,), default=1)
parser.add_argument("--init_states", default="", help="Consume supplied starts in order")
parser.add_argument("--seed", type=int, default=0, help="draw seed for the starting states")
parser.add_argument(
    "--horizon_s", type=float, default=None, help="seconds before an episode times out (default: the task's)"
)
parser.add_argument(
    "--pos_speed", type=float, default=0.08, help="m/s the command point moves at full deflection"
)
parser.add_argument("--rot_speed", type=float, default=0.6, help="rad/s the wrist turns at full deflection")
parser.add_argument("--jaw_speed", type=float, default=0.02, help="m/s the jaws open and close at")
parser.add_argument("--robot_usd", default="")
parser.add_argument("--grid", type=int, nargs=2, default=(288, 384))
parser.add_argument(
    "--previews",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="include derived tactile preview streams",
)
add_write_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
_known, _rest = parser.parse_known_args()
if _known.write_queue_size < 1:
    parser.error("write_queue_size must be positive")
if _known.episode_write_workers < 1:
    parser.error("episode_write_workers must be positive")
if _known.write_mode is None:
    parser.set_defaults(write_mode="background" if TASK_IDS[_known.task].manager_cfg else "sync")
if not TASK_IDS[_known.task].manager_cfg and _known.write_mode == "background":
    parser.error("Background writing requires a native vector environment; use --write_mode sync")
if _known.teleop_device == "vr":
    if not TASK_IDS[_known.task].manager_cfg:
        parser.error("VR requires a native vector environment")
    _known.xr = True
if _known.init_states and not TASK_IDS[_known.task].manager_cfg:
    parser.error("init_states requires a native vector environment")
if _known.episodes < 1:
    parser.error("episodes must be positive")
if _known.seed < 0:
    parser.error("seed must be nonnegative")
if _known.horizon_s is not None and _known.horizon_s <= 0:
    parser.error("horizon_s must be positive")
if min(_known.pos_speed, _known.rot_speed, _known.jaw_speed) <= 0:
    parser.error("control speeds must be positive")
if _known.teleop_device == "keyboard" and _known.headless:
    parser.error("keyboard collection requires an interactive viewer")
_known.enable_cameras = True
app_launcher = AppLauncher(_known)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import numpy as np

from sim.harness import CTRL_FPS, DT, Harness
from sim.recording import EpisodeRecorder
from sim.tasks.registry import load_scene, load_task
from sim.tasks.spec import RetryReset
from sim.teleop import Command, make_device

TASK, PINS = load_task(_known.task)
for kname, knob in TASK.knobs.items():
    parser.add_argument(f"--{kname}", type=knob.type, default=knob.default, help=knob.help)
args_cli = parser.parse_args()
KNOBS = {k: PINS.get(k, getattr(args_cli, k)) for k in TASK.knobs}

REC_EVERY = int(round(1 / (CTRL_FPS * DT)))  # sim steps per dataset frame


def main(cleanup: ExitStack) -> int:
    if TASK_IDS[args_cli.task].manager_cfg:
        from sim.collection import run_collection

        return run_collection(args_cli, TASK, KNOBS, simulation_app, teleop=True)
    out = args_cli.out
    if not out:
        if args_cli.dataset:
            out = args_cli.dataset.rstrip("/") + "_run"
        else:
            repo = Path(__file__).resolve().parents[2]
            practice = Path(os.environ.get("OUTPUT_ROOT", str(repo / "outputs"))).expanduser() / "practice"
            practice.mkdir(parents=True, exist_ok=True)
            out = tempfile.mkdtemp(prefix=f"{TASK.name}_teleop_", dir=practice)
    out = os.path.expanduser(out)
    out = Path(out).resolve()
    if args_cli.dataset:
        dataset = Path(args_cli.dataset).expanduser().resolve()
        if dataset == out or dataset in out.parents or out in dataset.parents:
            raise ValueError("Dataset and report directories must not overlap")
        if dataset.exists() and any(dataset.iterdir()):
            raise FileExistsError(f"Dataset directory is not empty: {dataset}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Report directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    horizon_s = args_cli.horizon_s or TASK.horizon_s

    preview = load_scene(args_cli.task).PREVIEW
    h = Harness(
        TASK.make_scene_cfg(KNOBS),
        device=args_cli.device,
        grid=tuple(args_cli.grid),
        robot_usd=args_cli.robot_usd,
        scene_cam=(preview.eye, preview.look_at),
    )
    TASK.apply_collision(h)
    h.start()
    if not args_cli.headless:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active GUI viewport is available")
        viewport.camera_path = Sdf.Path("/World/scene_cam")
        viewport.resolution = (960, 720)
    task_id = TASK.task_id(KNOBS)
    recorder = EpisodeRecorder(h, TASK, args_cli.dataset, fps=CTRL_FPS) if args_cli.dataset else None
    device = make_device(
        args_cli.teleop_device,
        TASK,
        h,
        knobs=KNOBS,
        pos_speed=args_cli.pos_speed,
        rot_speed=args_cli.rot_speed,
        jaw_speed=args_cli.jaw_speed,
    )
    print(f"TELEOP task {task_id}  device {args_cli.teleop_device}", flush=True)
    print(device.help_text(), flush=True)
    print(f"TELEOP report: {out / (TASK.name + '_teleop.json')}", flush=True)

    episodes, resets, n_saved, nominal = [], [], 0, False
    report_path = out / f"{TASK.name}_teleop.json"

    def write_report(status):
        report = {
            "task": task_id,
            "device": args_cli.teleop_device,
            "seed": args_cli.seed,
            "knobs": KNOBS,
            "fps": CTRL_FPS,
            "status": status,
            "requested_attempts": args_cli.episodes,
            "episodes": episodes,
            "resets": resets,
            "n_episodes": len(episodes),
            "n_kept": n_saved,
        }
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, default=float) + "\n")
        temporary.replace(report_path)

    write_report("running")
    status = "failed"

    def finish_recording():
        try:
            if recorder is not None:
                recorder.discard_episode()
                print(f"TELEOP dataset finalized: {recorder.finalize()}", flush=True)
                if n_saved and args_cli.previews:
                    script = Path(__file__).resolve().parents[1] / "run/render_previews.py"
                    subprocess.run([sys.executable, str(script), "--dataset", str(recorder.root)], check=True)
        except BaseException:
            write_report("failed")
            raise
        write_report(status)

    cleanup.callback(finish_recording)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Teleoperation interrupted by signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    cleanup.callback(signal.signal, signal.SIGTERM, previous)
    ep = 0
    while ep < args_cli.episodes and simulation_app.is_running():
        h.unlatch()
        rng = np.random.default_rng([args_cli.seed, TASK.reset_seed_index(ep)])
        ctx = None
        for _attempt in range(3):
            try:
                ctx = TASK.reset(h, rng, ep, KNOBS)
            except RetryReset as e:
                print(f"TELEOP ep{ep}: reset retry ({e})", flush=True)
                continue
            break
        if ctx is None:
            print(f"TELEOP ep{ep}: SKIPPED (no usable start)", flush=True)
            episodes.append(
                {
                    "episode": ep,
                    "status": "reset_failed",
                    "marked_success": False,
                    "criterion_success": None,
                    "saved": False,
                }
            )
            ep += 1
            write_report("running")
            continue
        if not nominal:
            h.initial_render()
            nominal = True
        expert_metadata = (
            TASK.prepare_expert(ctx, args_cli.seed, ep) if args_cli.teleop_device == "scripted" else {}
        )
        device.begin_episode(ctx)

        hist: dict = {"t": []}
        t_ep, rec_i = 0.0, 0
        cmd = Command(np.zeros(3), np.zeros(3), 0.0, ctx["side"])
        print(f"TELEOP ep{ep}: go ({horizon_s:.0f} s)", flush=True)
        while simulation_app.is_running():
            cmd = device.advance(t_ep)
            if cmd.reset_episode or cmd.discard or cmd.mark_success or cmd.finish:
                break
            device.apply(cmd)
            if recorder is not None and rec_i % REC_EVERY == 0:
                recorder.add_frame()
            rec_i += 1
            h.step()
            t_ep += DT

            sig = TASK.signals(h, ctx)
            device.after_step(sig)
            hist["t"].append(t_ep)
            for k, v in sig.items():
                hist.setdefault(k, []).append(v)
            hist.setdefault("state", []).append(h.state16()[0].cpu().numpy())
            hist.setdefault("action", []).append(h.action16_recorded())
            if t_ep >= horizon_s:
                break

        if cmd.reset_episode:
            if recorder is not None:
                recorder.discard_episode()
            resets.append({"episode": ep, "seconds": round(t_ep, 3)})
            write_report("running")
            continue
        if not simulation_app.is_running():
            if recorder is not None:
                recorder.discard_episode()
            episodes.append({"episode": ep, "status": "interrupted", "saved": False})
            break
        marked = bool(cmd.mark_success)
        rec = {
            "episode": ep,
            "arm": ctx["side"],
            "seconds": round(t_ep, 1),
            "marked_success": marked,
            "discarded": bool(cmd.discard),
            "criterion_success": None,
            "saved": False,
            "timeouts": getattr(device, "timeouts", []),
        }
        if expert_metadata:
            rec["expert"] = expert_metadata
        if cmd.discard or not hist["t"]:
            if recorder is not None:
                recorder.discard_episode()
            print(f"TELEOP ep{ep}: discarded", flush=True)
            episodes.append(rec)
            rec["status"] = "discarded" if cmd.discard else "empty"
            ep += 1
            write_report("running")
            continue

        trace = TASK.build_trace(h, ctx, hist)
        jaw_col = 7 if ctx["side"] == "right" else 15
        trace.extra.setdefault("jaw_command_mm", np.asarray(hist["action"])[:, jaw_col])
        verdict_of = TASK.verdict(trace, ctx, KNOBS)
        rec["verdict"] = verdict_of.as_dict()
        rec["criterion_success"] = bool(verdict_of.success)
        rec["status"] = "finished"
        keep = marked or verdict_of.success
        rec["success_source"] = (
            "both"
            if marked and verdict_of.success
            else "operator"
            if marked
            else "criterion"
            if verdict_of.success
            else "none"
        )
        if recorder is not None:
            if keep:
                stats = recorder.save_episode(
                    n_saved,
                    verdict_of.success,
                    is_intervention=args_cli.teleop_device == "keyboard",
                    annotation={
                        "attempt_id": ep,
                        "seed": args_cli.seed,
                        "device": args_cli.teleop_device,
                        "expert": expert_metadata,
                        "marked_success": marked,
                        "criterion_success": bool(verdict_of.success),
                        "success_source": rec["success_source"],
                    },
                )
                rec["dataset_episode_index"] = n_saved
                rec["saved"] = True
                n_saved += 1
                print(f"TELEOP ep{ep}: kept ({rec['success_source']}) {stats}", flush=True)
            else:
                n_drop = recorder.discard_episode()
                print(f"TELEOP ep{ep}: dropped ({n_drop} frames, no success)", flush=True)
        episodes.append(rec)
        ep += 1
        write_report("running")

    status = "complete" if ep == args_cli.episodes else "interrupted"
    print(f"\n===== {task_id.upper()} (TELEOP) =====", flush=True)
    print(f"  kept {n_saved}/{len(episodes)} episodes", flush=True)
    return 0 if status == "complete" else 130


if __name__ == "__main__":
    with ExitStack() as cleanup:
        rc = main(cleanup)
    sys.stdout.flush()
    os._exit(rc)
