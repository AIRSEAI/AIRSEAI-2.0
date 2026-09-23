#!/usr/bin/env python3
"""Evaluate a policy checkpoint on a benchmark task.

The task supplies reset, the scripted prologue, the handover gate and the
verdict; the observation keys come from the checkpoint.

    # policy served from a separate process (the default path)
    python benchmark/eval/evaluate.py --task blind_peg_insertion \\
        --policy-server 127.0.0.1:5555 --episodes 25

    # in-process, when one env has both Isaac and lerobot
    python benchmark/eval/evaluate.py --task stack_cube --policy <ckpt_dir> --episodes 10

    # loop test without a checkpoint: action = current state
    python benchmark/eval/evaluate.py --task stack_cube --policy hold --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

from sim.tasks.registry import TASKS as TASK_IDS

parser = argparse.ArgumentParser(description="Policy evaluation in Isaac.")
parser.add_argument("--task", required=True, choices=sorted(TASK_IDS))
parser.add_argument("--policy", default="",
                    help="lerobot pretrained dir (in-process), or 'hold' for the loop test")
parser.add_argument("--policy-server", default="",
                    help="host:port of a running benchmark/policy/policy_server.py")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=11, help="episode seed")
parser.add_argument("--horizon_s", type=float, default=None,
                    help="policy control time after handover (default: the task's)")
parser.add_argument("--out", default="")
parser.add_argument("--robot_usd", default="")
parser.add_argument("--grid", type=int, nargs=2, default=(288, 384))
parser.add_argument("--video_dir", default="",
                    help="if set, write per-episode mp4s (base, active wrist, "
                         "third-person) here")
AppLauncher.add_app_launcher_args(parser)
_known, _rest = parser.parse_known_args()
app_launcher = AppLauncher(_known)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import numpy as np
import torch

from sim.harness import (
    CTRL_FPS,
    DT,
    STEPS_PER_CTRL,
    Harness,
    HoldPosePolicy,
    RemotePolicy,
    load_policy,
)
from sim.tasks.registry import load_task
from sim.tasks.spec import RetryReset

TASK, PINS = load_task(_known.task)
for kname, knob in TASK.knobs.items():
    parser.add_argument(f"--{kname}", type=knob.type, default=knob.default, help=knob.help)
args_cli = parser.parse_args()
KNOBS = {k: getattr(args_cli, k) for k in TASK.knobs} | PINS   # the id's pins win


def run_prologue(h, ctx, sig) -> tuple[bool, dict]:
    """Run the scripted phases before the handover phase. Returns
    (handover_ok, last_signals)."""
    if TASK.handover_phase is None:
        return True, sig
    phase_names = [n for n, _ in TASK.phases]
    phase_idx = 0
    while phase_idx < len(TASK.phases):
        name, timeout = TASK.phases[phase_idx]
        if name == TASK.handover_phase:
            break
        t_phase = 0.0
        while True:
            target, grip, quat = TASK.phase_target(h, ctx, name, t_phase, sig, KNOBS)
            if name in getattr(TASK, "joint_phases", ()):
                h.step_joint_toward(target, grip)
            else:
                h.step_ik_toward(target, grip, quat)
            h.step()
            t_phase += DT
            sig = TASK.signals(h, ctx)
            if TASK.phase_done(h, ctx, name, t_phase, sig, target, KNOBS) or t_phase >= timeout:
                break
        print(f"EVAL prologue {name}: t={t_phase:.2f} tcp_err "
              f"{h.tcp_err(target) * 1e3:.1f} mm area "
              f"{sum(sig['areas'].values()):.0f} mm2", flush=True)
        jump = TASK.next_phase(ctx, name)
        phase_idx = phase_names.index(jump) if jump is not None else phase_idx + 1
    return TASK.handover_ok(h, ctx, sig), sig


def main() -> int:
    device = args_cli.device
    out = os.path.expanduser(args_cli.out or f"~/{TASK.name}_eval")
    os.makedirs(out, exist_ok=True)
    horizon_s = args_cli.horizon_s or TASK.horizon_s

    cfg = TASK.make_scene_cfg(KNOBS)
    h = Harness(cfg, device=device, grid=tuple(args_cli.grid), robot_usd=args_cli.robot_usd,
                scene_cam=TASK.scene_cam if args_cli.video_dir else None)
    TASK.apply_collision(h)
    h.start()
    task_id = TASK.task_id(KNOBS)

    # ---- policy ----
    if args_cli.policy_server:
        policy = RemotePolicy(args_cli.policy_server)
        src = args_cli.policy_server
    elif args_cli.policy == "hold":
        policy = HoldPosePolicy()
        src = "hold-pose test policy"
    elif args_cli.policy:
        policy = load_policy(os.path.expanduser(args_cli.policy), device)
        src = args_cli.policy
    else:
        raise SystemExit("give --policy <ckpt|hold> or --policy-server host:port")
    keys = list(policy.input_keys)
    print(f"EVAL task {task_id}  policy {getattr(policy, 'policy_type', type(policy).__name__)} "
          f"from {src}", flush=True)
    print(f"EVAL observation keys (from the checkpoint): {keys}", flush=True)

    episodes = []
    nominal_captured = False
    for ep in range(args_cli.episodes):
        # One RNG stream per episode; reset and prologue are retried together.
        ep_rng = np.random.default_rng([args_cli.seed, ep])
        ctx, handoff = None, False
        for _attempt in range(3):
            try:
                ctx = TASK.reset(h, ep_rng, ep, KNOBS)
            except RetryReset as e:
                print(f"EVAL ep{ep}: reset retry ({e})", flush=True)
                continue
            # Once, after the first reset that succeeds.
            if not nominal_captured:
                h.initial_render()
                nominal_captured = True
                # Check the checkpoint's keys once, after the nominal tactile
                # render exists.
                h.update_cams()
                h.build_obs(keys)
            sig = TASK.signals(h, ctx)
            handoff, sig = run_prologue(h, ctx, sig)
            if handoff:
                break
            print(f"EVAL ep{ep}: prologue handover failed, retrying", flush=True)
        if not handoff:
            print(f"EVAL ep{ep}: SKIPPED (prologue failed 3x)", flush=True)
            episodes.append({"episode": ep, "arm": ctx["side"] if ctx else None,
                             "success": False, "failure": "pregrasp", "metrics": {},
                             "wall_s": 0.0})
            continue
        policy.reset()

        n_ctrl = int(horizon_s * CTRL_FPS)
        t0 = time.perf_counter()
        # Per-step record of the task's signals.
        hist = {"t": []}
        frames = {"base": [], "wrist": [], "scene": []} if args_cli.video_dir else None
        stopped_early = False
        for step in range(n_ctrl):
            sig = TASK.signals(h, ctx)
            hist["t"].append(step / CTRL_FPS)
            for k, v in sig.items():
                hist.setdefault(k, []).append(v)

            h.update_cams(include_scene_cam=frames is not None)
            obs = h.build_obs(keys)
            if frames is not None:
                frames["base"].append(h.cam_rgb("base"))
                frames["wrist"].append(h.cam_rgb(f"{ctx['side']}_wrist"))
                frames["scene"].append(h.cam_rgb("scene"))
            with torch.inference_mode():
                a16 = policy.select_action(obs)
            for _ in range(STEPS_PER_CTRL):
                h.apply_action16(a16)
                h.step()

            if TASK.should_terminate(ctx, hist):
                stopped_early = True
                print(f"EVAL early-stop ep{ep} at {step / CTRL_FPS:.1f} s", flush=True)
                break

        trace = TASK.build_trace(h, ctx, hist)
        v = TASK.verdict(trace, ctx, KNOBS)
        rec = {"episode": ep, "arm": ctx["side"], "success": v.success,
               "failure": v.failure.value, "metrics": v.metrics,
               "stopped_early": stopped_early,
               "wall_s": round(time.perf_counter() - t0, 1)}
        rec.update(TASK.episode_extra(ctx, hist))
        episodes.append(rec)
        if frames is not None and frames["base"]:
            _write_videos(frames, os.path.expanduser(args_cli.video_dir), ep)
        print(f"EVAL episode {ep}: {'OK ' if v.success else 'FAIL'} {v.failure.value:14s} "
              f"arm={ctx['side']}", flush=True)

    n_ok = sum(bool(e["success"]) for e in episodes)
    report = {"task": task_id, "policy": src, "input_keys": keys, "seed": args_cli.seed,
              "knobs": KNOBS, "horizon_s": horizon_s, **TASK.report_extra(KNOBS),
              "episodes": episodes, "n_episodes": len(episodes), "n_success": n_ok,
              "success_rate": n_ok / max(len(episodes), 1)}
    with open(os.path.join(out, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"EVAL_DONE {n_ok}/{len(episodes)}  report: {out}/eval_report.json", flush=True)
    return 0


def _write_videos(frames: dict, vdir: str, ep: int) -> None:
    import subprocess
    os.makedirs(vdir, exist_ok=True)
    for view, fr in frames.items():
        if not fr:
            continue
        arr = np.stack(fr)
        _, hh, ww, _c = arr.shape
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{ww}x{hh}", "-r", str(CTRL_FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             os.path.join(vdir, f"episode_{ep:03d}_{view}.mp4")],
            input=arr.tobytes(), check=True)


if __name__ == "__main__":
    import traceback
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    finally:
        # Isaac's close() can hang; the report is on disk, so leave after 30 s.
        import threading
        threading.Thread(target=lambda: (__import__("time").sleep(30), os._exit(rc)),
                         daemon=True).start()
        simulation_app.close()
    os._exit(rc)
