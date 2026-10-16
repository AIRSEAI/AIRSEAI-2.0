#!/usr/bin/env python3
"""Run a task's scripted expert and, optionally, record a LeRobot v3 dataset.

    python benchmark/tasks/run_expert.py --task blind_peg_insertion \\
        --headless --enable_cameras --episodes 50 --record_dataset ~/peg_v3

"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time

from isaaclab.app import AppLauncher

from sim.collection_io import add_write_arguments
from sim.runtime import install_error_handler
from sim.tasks.registry import TASKS as TASK_IDS

install_error_handler()

parser = argparse.ArgumentParser(description="Scripted expert for a benchmark task.")
parser.add_argument("--task", required=True, choices=sorted(TASK_IDS))
parser.add_argument("--out", type=str, default="")
parser.add_argument("--robot_usd", type=str, default="")
parser.add_argument("--grid", type=int, nargs=2, default=(288, 384))
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--allow_empty", action="store_true", help="Allow no accepted vector episodes")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--episode_start", type=int, default=0)
parser.add_argument(
    "--record_dataset",
    type=str,
    default="",
    help="write a LeRobot v3.0 dataset here, plus a review mp4 per kept episode",
)
parser.add_argument("--video", type=str, default="", help="write a third-person mp4 of the run to this path")
parser.add_argument(
    "--video_tactile",
    action="store_true",
    help="show the four tactile streams beside the scene camera in the video",
)
parser.add_argument("--fps", type=int, choices=(20,), default=20)
parser.add_argument(
    "--dump_trace", action="store_true", help="write each episode's per-step signals to <out>/trace_epN.json"
)
parser.add_argument("--init_states", type=str, default="", help="load frozen initial states in episode order")
parser.add_argument(
    "--init_states_out", type=str, default="", help="save initial states with their expert results"
)
parser.add_argument(
    "--init_states_n",
    type=int,
    default=0,
    help="stop after this many solved initial states; 0 runs all attempts",
)
parser.add_argument(
    "--keep_unclean",
    action="store_true",
    help="also record episodes that are not demo_clean, with is_success set",
)
add_write_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
# Launch Kit before importing task-specific arguments.
_known, _rest = parser.parse_known_args()
if _known.write_queue_size < 1:
    parser.error("write_queue_size must be positive")
if _known.episode_write_workers < 1:
    parser.error("episode_write_workers must be positive")
if _known.write_mode is None:
    parser.set_defaults(write_mode="background" if TASK_IDS[_known.task].manager_cfg else "sync")
if not TASK_IDS[_known.task].manager_cfg and _known.write_mode == "background":
    parser.error("Background writing requires a native vector environment; use --write_mode sync")
if _known.num_envs < 1:
    parser.error("num_envs must be positive")
parser.set_defaults(video_size=(640, 480) if _known.num_envs >= 8 else (960, 720))
if _known.num_envs > 1 and not TASK_IDS[_known.task].manager_cfg:
    parser.error("This task has no native vector environment")
if TASK_IDS[_known.task].manager_cfg:
    _known.enable_cameras = True
app_launcher = AppLauncher(_known)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import numpy as np
import torch

from sim import init_state as init_state_io
from sim.envs.bimanual_tactile import REAL_HOME_JOINT_POS, TACTILE_SENSORS
from sim.harness import CTRL_FPS, DT, Harness
from sim.recording import EpisodeRecorder
from sim.tasks.registry import load_scene, load_task
from sim.tasks.spec import RetryReset

TASK, PINS = load_task(_known.task)
for kname, knob in TASK.knobs.items():
    parser.add_argument(f"--{kname}", type=knob.type, default=knob.default, help=knob.help)
args_cli = parser.parse_args()
KNOBS = {k: getattr(args_cli, k) for k in TASK.knobs} | PINS  # the id's pins win

REC_EVERY = int(round(1 / (CTRL_FPS * DT)))  # sim steps per dataset frame


def _joint_margin(h):
    """[joint index, margin in degrees] of the active arm joint closest to a
    position limit. Trace-only; separates an arm stalled on a limit from a
    slipping grasp."""
    try:
        jids = h.ch["arm"].joint_ids
        q = h.robot.data.joint_pos[0, jids]
        lim = h.robot.data.soft_joint_pos_limits[0, jids]
        m = torch.minimum(q - lim[:, 0], lim[:, 1] - q)
        i = int(torch.argmin(m))
        return [i, round(float(m[i]) * 57.2958, 1)]
    except Exception:
        return [-1, 0.0]


# HOME in the 16-dim layout's 14 arm slots, degrees: right joints 1-7, left 1-7.
HOME_ARM_DEG = np.asarray(
    [
        math.degrees(REAL_HOME_JOINT_POS[f"openarm_{a}_joint{i}"])
        for a in ("right", "left")
        for i in range(1, 8)
    ]
)
ARM_SLOTS = np.r_[0:7, 8:15]
GRIP_SLOTS = [7, 15]


def _arm_extremes(h, acc: dict) -> None:
    """Fold the active arm's wrist angles and joint-limit margins into `acc`:
    the peak of joints 6 and 7, and the joint that came closest to a limit."""
    jids = h.ch["arm"].joint_ids
    q = h.robot.data.joint_pos[0, jids]
    lim = h.robot.data.soft_joint_pos_limits[0, jids]
    deg = (q * 57.2958).tolist()
    margin = (torch.minimum(q - lim[:, 0], lim[:, 1] - q) * 57.2958).tolist()
    acc["j6_peak_deg"] = max(acc.get("j6_peak_deg", 0.0), abs(deg[5]))
    acc["j7_peak_deg"] = max(acc.get("j7_peak_deg", 0.0), abs(deg[6]))
    i = int(np.argmin(margin))
    if margin[i] < acc.get("margin_min_deg", float("inf")):
        acc["margin_min_deg"] = margin[i]
        acc["margin_min_joint"] = f"{h.ch['side']}_joint{i + 1}"


def _git_commit() -> str:
    """The revision the states were generated at, or "" outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    if TASK_IDS[args_cli.task].manager_cfg:
        from sim.collection import run_collection

        return run_collection(args_cli, TASK, KNOBS, simulation_app)
    out = os.path.expanduser(args_cli.out or f"~/{TASK.name}_out")
    os.makedirs(out, exist_ok=True)

    cfg = TASK.make_scene_cfg(KNOBS)
    # The scene camera also feeds the per-episode review video of a recording.
    want_scene_cam = bool(args_cli.video or args_cli.record_dataset)
    preview = load_scene(args_cli.task).PREVIEW
    h = Harness(
        cfg,
        device=args_cli.device,
        grid=tuple(args_cli.grid),
        robot_usd=args_cli.robot_usd,
        scene_cam=(preview.eye, preview.look_at) if want_scene_cam else None,
    )
    TASK.apply_collision(h)
    h.start()
    task_id = TASK.task_id(KNOBS)
    print(f"STEP task {task_id} knobs {KNOBS}", flush=True)

    # ---- the recording (schema and frames shared with the teleoperator) ----
    recorder = None
    if args_cli.record_dataset:
        recorder = EpisodeRecorder(h, TASK, args_cli.record_dataset, fps=CTRL_FPS)
        print(f"STEP recording LeRobot v3 dataset -> {args_cli.record_dataset}", flush=True)

    vid = None
    if args_cli.video:
        from pathlib import Path

        from sim.video import Mp4Writer, compose_tactile_frame

        vp = Path(os.path.expanduser(args_cli.video))
        vid = Mp4Writer(vp, args_cli.fps)
        print(
            f"STEP recording video -> {vp}{' (scene + 4 tactile)' if args_cli.video_tactile else ''}",
            flush=True,
        )
    every = max(1, round(60 / args_cli.fps))
    frame_i = 0
    # Short labels for the tactile tiles, in the order the grid shows them.
    _tile_label = {
        "right_left_finger_tactile": "R-mid",
        "right_right_finger_tactile": "R-idx",
        "left_left_finger_tactile": "L-mid",
        "left_right_finger_tactile": "L-idx",
    }

    def video_frame(caption: str) -> np.ndarray:
        scene = h.cam_rgb("scene")
        if not args_cli.video_tactile:
            return scene
        from sim.tactile import contact_area

        tiles = []
        for pn in TACTILE_SENSORS:
            d = h.sensors[pn].data
            depth_mm = float(d.penetration_depth[0].max()) * 1e3 if d.penetration_depth is not None else 0.0
            area = (
                float(contact_area(d.penetration_depth, *h.pitch)["area_mm2"][0])
                if d.penetration_depth is not None
                else 0.0
            )
            tiles.append((_tile_label[pn], h.tacrgb_u8(pn), depth_mm, area, pad_shear(pn), pad_axial(pn)))
        return compose_tactile_frame(scene, tiles, caption)

    def pad_shear(pn: str) -> float:
        """Summed tangential force magnitude on one pad, x1e3 (the `ff_shear`
        scale). Where an axial load on a gripped peg shows up."""
        d = h.sensors[pn].data
        sh = getattr(d, "tactile_shear_force", None)
        if sh is None:
            return 0.0
        return float(torch.linalg.norm(sh[0], dim=-1).sum()) * 1e3

    def pad_axial(pn: str) -> float:
        """Signed world-vertical resultant on one pad (shear + normal, rotated
        to the world), x1e3, for the video header. Free-hanging it is the
        pad's share of the held object's weight; it reverses on contact."""
        d = h.sensors[pn].data
        nf = getattr(d, "tactile_normal_force", None)
        sh = getattr(d, "tactile_shear_force", None)
        qw = getattr(d, "tactile_points_quat_w", None)
        if None in (nf, sh, qw):
            return 0.0
        from sim.tactile.quantities import quat_apply

        local = torch.cat([sh, nf.unsqueeze(-1)], dim=-1)
        fw = torch.nan_to_num(quat_apply(qw, local)).sum(dim=1)[0]
        return float(fw[2]) * 1e3

    def pad_depth_area(pn: str) -> tuple[float, float]:
        from sim.tactile import contact_area

        d = h.sensors[pn].data
        if d.penetration_depth is None:
            return 0.0, 0.0
        return (
            float(d.penetration_depth[0].max()) * 1e3,
            float(contact_area(d.penetration_depth, *h.pitch)["area_mm2"][0]),
        )

    def review_close(ep_index: int | None) -> None:
        if recorder is not None:
            recorder.review_close(ep_index)

    episodes = []
    nominal_captured = False
    n_saved = 0  # dataset episode index of the next kept episode
    # Frozen initial states: read them instead of drawing, or snapshot the
    # draws so they can be read back later. Never both.
    frozen = (
        init_state_io.load(os.path.expanduser(args_cli.init_states))["states"]
        if args_cli.init_states
        else None
    )
    snapshots: list[dict] = []
    n_episodes = len(frozen) if frozen is not None else args_cli.episodes
    if frozen is not None:
        print(f"STEP initial states: {len(frozen)} from {args_cli.init_states}", flush=True)
    if args_cli.episode_start < 0 or (frozen is not None and args_cli.episode_start):
        raise ValueError("episode_start requires fresh states and a nonnegative value")
    for ep in range(args_cli.episode_start, args_cli.episode_start + n_episodes):
        # ---- reset ----
        ep_rng = np.random.default_rng([args_cli.seed, TASK.reset_seed_index(ep)])

        # ---- reset AND prologue, retried as a unit ----
        # Retry reset and prologue together, up to three times.
        ctx = None
        snap = None
        pregrasp_failed = False
        for attempt in range(3):
            ctx = None
            review_close(None)  # an abandoned attempt keeps no video
            try:
                # Retries restore the same frozen state or draw a new random state.
                ctx = (
                    init_state_io.restore(h, TASK, frozen[ep], KNOBS)
                    if frozen is not None
                    else TASK.reset(h, ep_rng, ep, KNOBS)
                )
            except RetryReset as e:
                print(f"STEP ep{ep}: reset retry ({e})", flush=True)
                continue
            # Once, after the first reset that succeeds: a reset that asks for
            # a retry leaves the scene in no state to render the baseline from.
            if not nominal_captured:
                h.initial_render()
                nominal_captured = True
                print("STEP nominal tactile captured", flush=True)
            if args_cli.init_states_out:
                # Taken here: the scene has settled and the expert has not
                # moved yet, so this is what an episode starts from.
                snap = init_state_io.snapshot(h, TASK, ctx, ep)

            expert_metadata = TASK.prepare_expert(ctx, args_cli.seed, ep)
            rec = {"episode": ep, "arm": ctx["side"], "timeouts": []}
            if expert_metadata:
                rec["expert"] = expert_metadata
            arm_acc: dict = {}
            # Record every key the task's `signals` produces, so task-specific
            # quantities reach `build_trace`.
            hist = {"t": []}
            recording = TASK.handover_phase is None  # t=0 tasks record from the start
            rec_i = 0  # sim steps since recording started
            pregrasp_failed = False
            t_ep = 0.0
            trace_rows = []
            sig = TASK.signals(h, ctx)

            phase_names = [n for n, _ in TASK.phases]
            phase_idx = 0
            while phase_idx < len(TASK.phases):
                name, timeout = TASK.phases[phase_idx]
                t_phase = 0.0
                while True:
                    target, grip, quat = TASK.phase_target(h, ctx, name, t_phase, sig, KNOBS)
                    # A task may cap how far ahead of the TCP the command sits,
                    # per phase. Optional: a task that does not define
                    # `speed_for` drives uncapped.
                    v, w = getattr(TASK, "speed_for", lambda _n: (None, None))(name)
                    if name in getattr(TASK, "joint_phases", ()):
                        # Joint-space phase: `target` is a joint dict, not a
                        # world point. Used for a move that re-poses the arm
                        # (HOME -> working configuration), which IK cannot do.
                        h.step_joint_toward(target, grip)
                    else:
                        h.step_ik_toward(target, grip, quat, max_speed=v, max_ang_speed=w)
                    # A dataset frame is taken after the command is set and
                    # before the physics step, so the first frame of a t = 0
                    # episode is the settled HOME.
                    if recording:
                        if rec_i % REC_EVERY == 0:
                            _arm_extremes(h, arm_acc)
                            if "start_state" not in rec:
                                rec["start_state"] = h.to16(h.robot.data.joint_pos[0])
                            if recorder is not None:
                                recorder.add_frame()
                        rec_i += 1
                    h.step()
                    TASK.after_control_step(h, ctx, DT)
                    if vid is not None:
                        h.scene_cam.update(DT)
                        if frame_i % every == 0:
                            vid.add(
                                video_frame(
                                    f"ep{ep} {ctx['side']}  {name}  t {t_ep:5.1f}s"
                                    + (f"  wp {ctx.get('wp_idx', 0) + 1}" if name == "search" else "")
                                    + (
                                        f"  depth {ctx.get('depth_now', 0.0) * 1e3:4.1f} mm"
                                        if name in ("search", "insert")
                                        else ""
                                    )
                                )
                            )
                    frame_i += 1
                    t_phase += DT
                    t_ep += DT

                    sig = TASK.signals(h, ctx)
                    if args_cli.dump_trace:
                        # A joint-space phase's target is a joint dict, not a point.
                        in_joint_space = isinstance(target, dict)
                        row = {
                            "t": round(t_ep, 3),
                            "phase": name,
                            "lat": round(float(sig.get("lat", 0.0)), 4),
                            "depth_mm": round(float(sig.get("depth", 0.0)) * 1e3, 3),
                            "tip_above_mm": round(float(sig.get("tip_above_mm", 0.0)), 3),
                            "tilt": round(float(sig.get("tilt", 0.0)), 2),
                            "axial": round(float(sig.get("axial", 0.0)), 1),
                            "contact": bool(ctx.get("wp_contact_t") is not None),
                            "wp": ctx.get("wp_idx"),
                            # Arm-side diagnostics: is the TCP where it was
                            # told to go, and is any arm joint on a limit?
                            "tcp_z": round(float(h.tcp_pos()[2]), 4),
                            "tgt_z": (float("nan") if in_joint_space else round(float(target[2]), 4)),
                            "tcp_err_mm": (
                                float("nan") if in_joint_space else round(float(h.tcp_err(target)) * 1e3, 2)
                            ),
                            "jl": _joint_margin(h),
                        }
                        # Every other scalar the task publishes this step.
                        for k, v in sig.items():
                            if k not in row and isinstance(v, (int, float, bool)):
                                row[k] = round(float(v), 5)
                        for pn in TACTILE_SENSORS:
                            dmm, a = pad_depth_area(pn)
                            row[_tile_label[pn]] = [round(dmm, 3), round(a, 1), round(pad_shear(pn), 2)]
                        trace_rows.append(row)
                    if recording:
                        hist["t"].append(t_ep)
                        for k, v in sig.items():
                            hist.setdefault(k, []).append(v)

                    done = TASK.phase_done(h, ctx, name, t_phase, sig, target, KNOBS)
                    if done or t_phase >= timeout:
                        if t_phase >= timeout and not done:
                            rec["timeouts"].append(name)
                        break

                # Where next: the task may send the machine back to an
                # earlier phase (a retry); otherwise the next in the list.
                jump = TASK.next_phase(ctx, name)
                phase_idx = phase_names.index(jump) if jump is not None else phase_idx + 1
                nxt = phase_names[phase_idx] if phase_idx < len(phase_names) else None

                # Handover: the prologue is over. Validate before the recorded
                # episode starts; a failed grasp skips the episode rather than
                # polluting the dataset with an unwinnable start.
                if not recording and nxt == TASK.handover_phase:
                    if TASK.handover_ok(h, ctx, sig):
                        recording = True
                        rec_i = 0
                        t_ep = 0.0
                        print(
                            f"STEP ep{ep}: recording from {nxt} (area {sum(sig['areas'].values()):.0f} mm2)",
                            flush=True,
                        )
                    else:
                        pregrasp_failed = True
                        print(
                            f"STEP ep{ep}: PREGRASP FAILED (area "
                            f"{sum(sig['areas'].values()):.0f} mm2) -- skipping episode",
                            flush=True,
                        )
                        break

            # A prologue that reached the handover is the episode we keep;
            # anything else goes round again.
            rec["prologue_attempts"] = attempt + 1
            if not pregrasp_failed:
                break
            print(f"STEP ep{ep}: prologue handover failed, retrying", flush=True)

        if ctx is None:
            # Reset failed on all three attempts; the episode never started.
            review_close(None)
            episodes.append(
                {
                    "episode": ep,
                    "arm": None,
                    "verdict": {"success": False, "failure": "reset", "metrics": {}, "notes": []},
                }
            )
            continue
        if pregrasp_failed or not recording:
            review_close(None)
            rec.pop("start_state", None)
            print(f"STEP ep{ep}: SKIPPED (prologue failed 3x)", flush=True)
            rec["verdict"] = {"success": False, "failure": "pregrasp", "metrics": {}, "notes": []}
            episodes.append(rec)
            if snap is not None:
                snap["meta"] = {"expert_ok": False, "failure": "pregrasp", "t_solve_s": None}
                snapshots.append(snap)
            continue

        rec["arm_extremes"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in arm_acc.items()}
        st = rec.pop("start_state", None)
        if st is not None and TASK.handover_phase is None:
            rec["start_pose_err_deg"] = round(float(np.abs(st[ARM_SLOTS] - HOME_ARM_DEG).max()), 2)
            rec["start_grip_mm"] = [round(float(st[i]), 2) for i in GRIP_SLOTS]
        if args_cli.dump_trace:
            with open(os.path.join(out, f"trace_ep{ep}.json"), "w") as f:
                json.dump({"episode": ep, "arm": ctx["side"], "rows": trace_rows}, f)
        trace = TASK.build_trace(h, ctx, hist)
        v = TASK.verdict(trace, ctx, KNOBS)
        rec["verdict"] = v.as_dict()
        rec.update(TASK.episode_extra(ctx, hist))
        # A demonstration worth training on: succeeded, and did not have to
        # be rescued from a stall on the way. Tasks that do not report
        # `insert_stalled` get `demo_clean == success`.
        rec["demo_clean"] = bool(
            v.success and not rec.get("insert_stalled", False) and rec.get("prologue_attempts", 1) == 1
        )
        if snap is not None:
            snap["meta"] = {
                "expert_ok": bool(v.success),
                "failure": v.failure.value,
                "t_solve_s": round(t_ep, 2),
                "demo_clean": bool(rec["demo_clean"]),
            }
            snapshots.append(snap)
        if recorder is not None:
            if rec["demo_clean"] or args_cli.keep_unclean:
                stats = recorder.save_episode(
                    n_saved,
                    v.success,
                    annotation={"attempt_id": ep, "seed": args_cli.seed, "expert": expert_metadata},
                )
                rec["dataset_episode_index"] = n_saved
                n_saved += 1
                print(f"STEP dataset episode {ep}: {stats}", flush=True)
            else:
                n_drop = recorder.discard_episode()
                print(
                    f"STEP dataset episode {ep}: DROPPED ({n_drop} frames; "
                    f"success={v.success} stalled={rec.get('insert_stalled')} "
                    f"attempts={rec.get('prologue_attempts')})",
                    flush=True,
                )
        episodes.append(rec)
        ext = rec["arm_extremes"]
        print(
            f"STEP episode {ep}: {'OK ' if v.success else 'FAIL'} {v.failure.value:14s} "
            f"arm={ctx['side']} timeouts={rec['timeouts']} "
            f"start_err={rec.get('start_pose_err_deg')}deg grip0={rec.get('start_grip_mm')}mm "
            f"j6max={ext.get('j6_peak_deg')} j7max={ext.get('j7_peak_deg')} "
            f"margin={ext.get('margin_min_deg')}@{ext.get('margin_min_joint')}",
            flush=True,
        )
        n_solved = sum(x["meta"]["expert_ok"] for x in snapshots)
        if args_cli.init_states_n and n_solved >= args_cli.init_states_n:
            print(f"STEP initial states: {n_solved} solved, stopping early", flush=True)
            break

    if args_cli.init_states_out:
        # Only the states the expert solved are offered as an evaluation set;
        # the rest are kept beside them, because a start no expert can solve
        # says something about the task, not about a policy.
        kept = [s for s in snapshots if s["meta"]["expert_ok"]]
        rejected = [s for s in snapshots if not s["meta"]["expert_ok"]]
        for i, st in enumerate(kept):
            st["id"] = i
        path = init_state_io.save(
            os.path.expanduser(args_cli.init_states_out),
            task=task_id,
            states=kept,
            rejected=rejected,
            generator={
                "seed": args_cli.seed,
                "candidates": len(snapshots),
                "knobs": KNOBS,
                "commit": _git_commit(),
                "date": time.strftime("%F"),
            },
        )
        print(f"STEP initial states: {len(kept)} kept, {len(rejected)} rejected -> {path}", flush=True)

    if vid is not None:
        stats = vid.close()
        print(f"STEP wrote {args_cli.video} ({stats['frames']} frames)", flush=True)
    if recorder is not None:
        print(f"STEP dataset finalized: {recorder.finalize()}", flush=True)

    n_ok = sum(bool(e["verdict"]["success"]) for e in episodes)
    n_clean = sum(bool(e.get("demo_clean")) for e in episodes)
    report = {
        "task": task_id,
        "seed": args_cli.seed,
        "knobs": KNOBS,
        "n_clean": n_clean,
        "keep_unclean": bool(args_cli.keep_unclean),
        **TASK.report_extra(KNOBS),
        "episodes": episodes,
        "n_episodes": len(episodes),
        "n_success": n_ok,
        "checks": {
            "expert_success_rate": {
                "pass": bool(n_ok >= 0.9 * len(episodes)),
                "min_rate": 0.9,
                "n": n_ok,
                "of": len(episodes),
            }
        },
    }
    with open(os.path.join(out, TASK.report_name), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n===== {task_id.upper()} (SCRIPTED) =====", flush=True)
    print(f"  success {n_ok}/{len(episodes)}  (expert needs >= {math.ceil(0.9 * len(episodes))})", flush=True)
    print(
        f"  clean demonstrations {n_clean}/{len(episodes)}"
        + ("" if args_cli.keep_unclean else "  (only these were written to the dataset)"),
        flush=True,
    )
    print("CHECK_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    import traceback

    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    finally:
        # Isaac's teardown can wedge; the result is already on disk, so give
        # close() 30 s and then leave regardless.
        import threading

        threading.Thread(target=lambda: (__import__("time").sleep(30), os._exit(rc)), daemon=True).start()
        simulation_app.close()
    os._exit(rc)
