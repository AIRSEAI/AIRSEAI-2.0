#!/usr/bin/env python3
"""Run a task's scripted expert and, optionally, record a LeRobot v3 dataset.

Task-agnostic: the phases, handover gate and verdict come from the task's
`TASK` object (`src/sim/tasks/<task>.py`), and the Isaac plumbing from
`sim.harness`; the evaluator drives the same `TASK` up to the handover point.

    python benchmark/tasks/run_expert.py --task blind_peg_insertion \\
        --headless --enable_cameras --episodes 50 --record_dataset ~/peg_v3

Normally run through `benchmark/run/collect_dataset.py`; run it directly to debug.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

from sim.tasks.registry import TASKS as TASK_IDS

parser = argparse.ArgumentParser(description="Scripted expert for a benchmark task.")
parser.add_argument("--task", required=True, choices=sorted(TASK_IDS))
parser.add_argument("--out", type=str, default="")
parser.add_argument("--robot_usd", type=str, default="")
parser.add_argument("--grid", type=int, nargs=2, default=(288, 384))
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--record_dataset", type=str, default="",
                    help="write a LeRobot v3.0 dataset here, plus a review mp4 per kept episode")
parser.add_argument("--video", type=str, default="",
                    help="write a third-person mp4 of the run to this path")
parser.add_argument("--video_tactile", action="store_true",
                    help="show the four tactile streams beside the scene camera in the video")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--dump_trace", action="store_true",
                    help="write each episode's per-step signals to <out>/trace_epN.json")
parser.add_argument("--keep_unclean", action="store_true",
                    help="also record episodes that are not demo_clean, with is_success set")
AppLauncher.add_app_launcher_args(parser)
# The task's knobs come from its module, and the task modules import isaaclab,
# which needs Kit running. So: parse what is known, launch, register the
# knobs, then parse for real.
_known, _rest = parser.parse_known_args()
app_launcher = AppLauncher(_known)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import numpy as np
import torch

from sim.envs.bimanual_tactile import REAL_HOME_JOINT_POS, SENSOR_FEATURE_NAME, TACTILE_SENSORS
from sim.harness import CTRL_FPS, DT, Harness
from sim.tasks.registry import load_task
from sim.tasks.spec import RetryReset

TASK, PINS = load_task(_known.task)
for kname, knob in TASK.knobs.items():
    parser.add_argument(f"--{kname}", type=knob.type, default=knob.default, help=knob.help)
args_cli = parser.parse_args()
KNOBS = {k: getattr(args_cli, k) for k in TASK.knobs} | PINS   # the id's pins win

REC_EVERY = int(round(1 / (CTRL_FPS * DT)))     # sim steps per dataset frame


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
HOME_ARM_DEG = np.asarray([math.degrees(REAL_HOME_JOINT_POS[f"openarm_{a}_joint{i}"])
                           for a in ("right", "left") for i in range(1, 8)])
ARM_SLOTS = np.r_[0:7, 8:15]
GRIP_SLOTS = [7, 15]
# The third-person review video is written at the dataset rate.
REVIEW_DIRNAME = ("review", "third_person")


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


def main() -> int:
    out = os.path.expanduser(args_cli.out or f"~/{TASK.name}_out")
    os.makedirs(out, exist_ok=True)

    cfg = TASK.make_scene_cfg(KNOBS)
    # The scene camera also feeds the per-episode review video of a recording.
    want_scene_cam = bool(args_cli.video or args_cli.record_dataset)
    h = Harness(cfg, device=args_cli.device, grid=tuple(args_cli.grid),
                robot_usd=args_cli.robot_usd,
                scene_cam=TASK.scene_cam if want_scene_cam else None)
    TASK.apply_collision(h)
    h.start()
    task_id = TASK.task_id(KNOBS)
    print(f"STEP task {task_id} knobs {KNOBS}", flush=True)

    # ---- dataset writer (schema is task-independent) ----
    writer = None
    if args_cli.record_dataset:
        from lerobot_tactile_ext.writer import LeRobotTactileWriter
        from tactile_codec import PROFILES
        rows, cols = h.rows, h.cols
        feats = {f"observation.images.{k}": {"dtype": "video", "shape": (480, 640, 3)}
                 for k in ("base", "left_wrist", "right_wrist")}
        for pn in TACTILE_SENSORS:
            sname = SENSOR_FEATURE_NAME[pn]
            feats[f"observation.images.{sname}_tactile"] = {
                "dtype": "video", "shape": (288, 384, 3)}
            # Native TacSL force-field channels: depth is the normal component
            # (F_n = k_n*d, k_n = 1.0); ff_shear is `tactile_shear_force` x1e3.
            feats[f"observation.depth.{sname}_tactile"] = {
                "dtype": "data", "shape": (rows, cols, 1),
                "profile": PROFILES["sim_v1_depth"]}
            feats[f"observation.ff_shear.{sname}_tactile"] = {
                "dtype": "data", "shape": (rows, cols, 2),
                "profile": PROFILES["sim_v1_shear"]}
        names = [f"{a}_joint_{i}.pos" for a in ("right", "left") for i in range(1, 8)]
        names = names[:7] + ["right_gripper.pos"] + names[7:] + ["left_gripper.pos"]
        feats["observation.state"] = {"dtype": "float32", "shape": (16,), "names": names}
        feats["action"] = {"dtype": "float32", "shape": (16,), "names": names}
        for k in ("is_intervention", "control_mode", "is_success"):
            feats[k] = {"dtype": "int64", "shape": (1,)}
        if not TASK.instruction:
            raise SystemExit(
                f"task '{TASK.name}' has no `instruction`; the dataset needs "
                f"a language label")
        writer = LeRobotTactileWriter(os.path.expanduser(args_cli.record_dataset),
                                      fps=CTRL_FPS, features=feats,
                                      task=TASK.instruction)
        print(f"STEP recording LeRobot v3 dataset -> {args_cli.record_dataset}", flush=True)

    vid = None
    if args_cli.video:
        from pathlib import Path

        from sim.video import Mp4Writer, compose_tactile_frame
        vp = Path(os.path.expanduser(args_cli.video))
        vid = Mp4Writer(vp, args_cli.fps)
        print(f"STEP recording video -> {vp}"
              f"{' (scene + 4 tactile)' if args_cli.video_tactile else ''}", flush=True)
    every = max(1, round(60 / args_cli.fps))
    frame_i = 0
    # Short labels for the tactile tiles, in the order the grid shows them.
    _tile_label = {"right_left_finger_tactile": "R-mid", "right_right_finger_tactile": "R-idx",
                   "left_left_finger_tactile": "L-mid", "left_right_finger_tactile": "L-idx"}

    def video_frame(caption: str) -> np.ndarray:
        scene = h.cam_rgb("scene")
        if not args_cli.video_tactile:
            return scene
        from sim.tactile import contact_area
        tiles = []
        for pn in TACTILE_SENSORS:
            d = h.sensors[pn].data
            depth_mm = (float(d.penetration_depth[0].max()) * 1e3
                        if d.penetration_depth is not None else 0.0)
            area = (float(contact_area(d.penetration_depth, *h.pitch)["area_mm2"][0])
                    if d.penetration_depth is not None else 0.0)
            tiles.append((_tile_label[pn], h.tacrgb_u8(pn), depth_mm, area, pad_shear(pn),
                          pad_axial(pn)))
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
        return (float(d.penetration_depth[0].max()) * 1e3,
                float(contact_area(d.penetration_depth, *h.pitch)["area_mm2"][0]))

    # ---- third-person review video: one mp4 per kept dataset episode ----
    # Named by the DATASET episode index, which skips dropped episodes, so
    # stage 3 can pair each video with the episode it converts.
    review = {"dir": None, "writer": None, "path": None}
    if writer is not None and h.scene_cam is not None:
        from pathlib import Path

        from sim.video import Mp4Writer
        root = Path(os.path.expanduser(args_cli.record_dataset))
        review["dir"] = root.joinpath(*REVIEW_DIRNAME)

    def review_add(frame: np.ndarray) -> None:
        if review["dir"] is None:
            return
        if review["writer"] is None:
            review["path"] = review["dir"] / "_pending.mp4"
            review["writer"] = Mp4Writer(review["path"], CTRL_FPS)
        review["writer"].add(frame)

    def review_close(ep_index: int | None) -> None:
        """Keep the pending video as `episode_<ep_index>.mp4`, or drop it."""
        if review["writer"] is None:
            return
        review["writer"].close()
        if ep_index is None:
            review["path"].unlink(missing_ok=True)
        else:
            review["path"].rename(review["dir"] / f"episode_{ep_index:06d}.mp4")
        review["writer"] = None

    def record_frame() -> np.ndarray:
        # Taken before the physics step, so no camera has been updated yet.
        h.update_cams(include_scene_cam=True)
        fr = {f"observation.images.{k}": h.cam_rgb(k) for k in ("base", "left_wrist", "right_wrist")}
        fr["observation.state"] = h.to16(h.robot.data.joint_pos[0])
        fr["action"] = h.action16_recorded()
        for pn in TACTILE_SENSORS:
            sname = SENSOR_FEATURE_NAME[pn]
            pen, sh = h.tactile_raw(pn)
            fr[f"observation.images.{sname}_tactile"] = h.tacrgb_u8(pn)
            fr[f"observation.depth.{sname}_tactile"] = pen
            fr[f"observation.ff_shear.{sname}_tactile"] = sh
        writer.add_frame(fr)
        if h.scene_cam is not None:
            review_add(h.cam_rgb("scene"))
        return fr["observation.state"]

    episodes = []
    nominal_captured = False
    n_saved = 0                      # dataset episode index of the next kept episode
    for ep in range(args_cli.episodes):
        # ---- reset ----
        # One RNG stream per episode, seeded (seed, ep): episode N gets the same
        # draw whatever extra draws (`RetryReset`) earlier episodes consumed.
        ep_rng = np.random.default_rng([args_cli.seed, ep])

        # ---- reset AND prologue, retried as a unit ----
        # Neither a bad draw nor a failed scripted grasp is charged to the
        # episode. A retry draws again from the already-advanced `ep_rng`.
        ctx = None
        pregrasp_failed = False
        for attempt in range(3):
            ctx = None
            review_close(None)       # an abandoned attempt keeps no video
            try:
                ctx = TASK.reset(h, ep_rng, ep, KNOBS)
            except RetryReset as e:
                print(f"STEP ep{ep}: reset retry ({e})", flush=True)
                continue
            # Once, after the first reset that succeeds: a reset that asks for
            # a retry leaves the scene in no state to render the baseline from.
            if not nominal_captured:
                h.initial_render()
                nominal_captured = True
                print("STEP nominal tactile captured", flush=True)

            rec = {"episode": ep, "arm": ctx["side"], "timeouts": []}
            arm_acc: dict = {}
            # Record every key the task's `signals` produces, so task-specific
            # quantities reach `build_trace`.
            hist = {"t": []}
            recording = TASK.handover_phase is None  # t=0 tasks record from the start
            rec_i = 0                # sim steps since recording started
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
                        h.step_ik_toward(target, grip, quat,
                                         max_speed=v, max_ang_speed=w)
                    # A dataset frame is taken after the command is set and
                    # before the physics step, so the first frame of a t = 0
                    # episode is the settled HOME.
                    if recording:
                        if rec_i % REC_EVERY == 0:
                            _arm_extremes(h, arm_acc)
                            if "start_state" not in rec:
                                rec["start_state"] = h.to16(h.robot.data.joint_pos[0])
                            if writer is not None:
                                record_frame()
                        rec_i += 1
                    h.step()
                    if vid is not None:
                        h.scene_cam.update(DT)
                        if frame_i % every == 0:
                            vid.add(video_frame(
                                f"ep{ep} {ctx['side']}  {name}  t {t_ep:5.1f}s"
                                + (f"  wp {ctx.get('wp_idx', 0) + 1}" if name == "search" else "")
                                + (f"  depth {ctx.get('depth_now', 0.0) * 1e3:4.1f} mm"
                                   if name in ("search", "insert") else "")))
                    frame_i += 1
                    t_phase += DT
                    t_ep += DT

                    sig = TASK.signals(h, ctx)
                    if args_cli.dump_trace:
                        # A joint-space phase's target is a joint dict, not a point.
                        in_joint_space = isinstance(target, dict)
                        row = {"t": round(t_ep, 3), "phase": name,
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
                               "tgt_z": (float("nan") if in_joint_space
                                         else round(float(target[2]), 4)),
                               "tcp_err_mm": (float("nan") if in_joint_space
                                              else round(float(h.tcp_err(target)) * 1e3, 2)),
                               "jl": _joint_margin(h)}
                        # Every other scalar the task publishes this step.
                        for k, v in sig.items():
                            if k not in row and isinstance(v, (int, float, bool)):
                                row[k] = round(float(v), 5)
                        for pn in TACTILE_SENSORS:
                            dmm, a = pad_depth_area(pn)
                            row[_tile_label[pn]] = [round(dmm, 3), round(a, 1),
                                                    round(pad_shear(pn), 2)]
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
                        print(f"STEP ep{ep}: recording from {nxt} "
                              f"(area {sum(sig['areas'].values()):.0f} mm2)", flush=True)
                    else:
                        pregrasp_failed = True
                        print(f"STEP ep{ep}: PREGRASP FAILED (area "
                              f"{sum(sig['areas'].values()):.0f} mm2) -- skipping episode",
                              flush=True)
                        break


            # A prologue that reached the handover is the episode we keep;
            # anything else goes round again.
            rec["prologue_attempts"] = attempt + 1
            if not pregrasp_failed:
                break
            print(f"STEP ep{ep}: prologue handover failed, retrying", flush=True)

        if ctx is None:
            # Three draws and the task rejected all of them. Not the same thing
            # as a failed grasp: the episode never started.
            review_close(None)
            episodes.append({"episode": ep, "arm": None,
                             "verdict": {"success": False, "failure": "reset",
                                         "metrics": {}, "notes": []}})
            continue
        if pregrasp_failed or not recording:
            review_close(None)
            rec.pop("start_state", None)
            print(f"STEP ep{ep}: SKIPPED (prologue failed 3x)", flush=True)
            rec["verdict"] = {"success": False, "failure": "pregrasp", "metrics": {}, "notes": []}
            episodes.append(rec)
            continue

        rec["arm_extremes"] = {k: (round(v, 1) if isinstance(v, float) else v)
                               for k, v in arm_acc.items()}
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
        rec["demo_clean"] = bool(v.success and not rec.get("insert_stalled", False)
                                 and rec.get("prologue_attempts", 1) == 1)
        if writer is not None:
            if rec["demo_clean"] or args_cli.keep_unclean:
                stats = writer.save_episode(scalars={"is_intervention": 0, "control_mode": 0,
                                                     "is_success": int(v.success)})
                review_close(n_saved)
                rec["dataset_episode_index"] = n_saved
                n_saved += 1
                print(f"STEP dataset episode {ep}: {stats}", flush=True)
            else:
                n_drop = writer.discard_episode()
                review_close(None)
                print(f"STEP dataset episode {ep}: DROPPED ({n_drop} frames; "
                      f"success={v.success} stalled={rec.get('insert_stalled')} "
                      f"attempts={rec.get('prologue_attempts')})", flush=True)
        episodes.append(rec)
        ext = rec["arm_extremes"]
        print(f"STEP episode {ep}: {'OK ' if v.success else 'FAIL'} {v.failure.value:14s} "
              f"arm={ctx['side']} timeouts={rec['timeouts']} "
              f"start_err={rec.get('start_pose_err_deg')}deg grip0={rec.get('start_grip_mm')}mm "
              f"j6max={ext.get('j6_peak_deg')} j7max={ext.get('j7_peak_deg')} "
              f"margin={ext.get('margin_min_deg')}@{ext.get('margin_min_joint')}", flush=True)

    if vid is not None:
        stats = vid.close()
        print(f"STEP wrote {args_cli.video} ({stats['frames']} frames)", flush=True)
    if writer is not None:
        print(f"STEP dataset finalized: {writer.finalize()}", flush=True)

    n_ok = sum(bool(e["verdict"]["success"]) for e in episodes)
    n_clean = sum(bool(e.get("demo_clean")) for e in episodes)
    report = {"task": task_id, "seed": args_cli.seed, "knobs": KNOBS,
              "n_clean": n_clean, "keep_unclean": bool(args_cli.keep_unclean),
              **TASK.report_extra(KNOBS),
              "episodes": episodes, "n_episodes": len(episodes), "n_success": n_ok,
              "checks": {"expert_success_rate": {"pass": bool(n_ok >= 0.9 * len(episodes)),
                                                 "min_rate": 0.9,
                                                 "n": n_ok, "of": len(episodes)}}}
    with open(os.path.join(out, TASK.report_name), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n===== {task_id.upper()} (SCRIPTED) =====", flush=True)
    print(f"  success {n_ok}/{len(episodes)}  (expert needs >= {math.ceil(0.9 * len(episodes))})",
          flush=True)
    print(f"  clean demonstrations {n_clean}/{len(episodes)}"
          + ("" if args_cli.keep_unclean else "  (only these were written to the dataset)"),
          flush=True)
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
        threading.Thread(target=lambda: (__import__("time").sleep(30), os._exit(rc)),
                         daemon=True).start()
        simulation_app.close()
    os._exit(rc)
