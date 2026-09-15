"""handover_cup_transfer expert.

Phases::

    home -> above_bar -> descend -> close_left -> lift -> turn_flat -> present
        -> above_bar2 -> descend2 -> close_right -> release_left -> part
        -> turn_up -> above_cup -> lower -> release -> retreat
    above_bar2               policy takes over here
"""

from __future__ import annotations

import numpy as _np
import torch as _torch

from sim.criteria.probes import axis_from_quat as _axis
from sim.criteria.probes import seated_in_cup as _seat
from sim.criteria.probes import tilt_deg as _tilt
from sim.criteria.probes import tip_position as _tip
from sim.criteria.tasks import handover_cup_transfer as _verdict
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.handover_cup_transfer import (
    CUP_BORE_R,
    MIN_SEAT,
)
from sim.envs.bimanual_tactile import TABLE_TOP_Z, TCP_JAW_L
from sim.envs.handover_cup_transfer import (
    BAR_FOOT_Z,
    BAR_LENGTH,
    BAR_SECTION,
    CUP_RIM_Z,
    CUP_XY,
    DEST,
    SOURCE,
    SPAWN_JITTER,
    SPAWN_YAW_JITTER,
    HandoverCupTransferSceneCfg,
)
from sim.tactile import apply_ccd, apply_sdf_collision, apply_triangle_mesh_collision
from sim.tasks.scene_utils import _mat_from_quat as _mat
from sim.tasks.scene_utils import align_frame_quat as _frame
from sim.tasks.spec import Knob, RetryReset, Task

# Controller parameters.
# Jaw command while approaching, metres per jaw: a 14 mm gel gap for the 5 mm face.
OPEN_WIDTH = 0.010
# Where each gel grips, metres up the bar from its foot. Limits: the bar may
# stand at most ~58 mm above a gel before it meets the palm, two blades side by
# side need ~35 mm, and each pad centre stays >= 12 mm from either bar end.
LEFT_GRASP_H = 0.120

RIGHT_GRASP_H = 0.085

# Palm direction for a side grasp of the upright bar, per arm: each palm
# points to its own side, so the two grippers take opposite sides of the bar.
PALM_TARGET = {"left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0)}

# How far out along its own side a hand waits before closing on the bar,
# metres, so the fingertips come in along the jaw axis rather than across it.
SIDE_STANDOFF = 0.070

CARRY_SPEED = 0.20

CARRY_ANG_SPEED = 90.0

GRIP_SEPARATION = LEFT_GRASP_H - RIGHT_GRASP_H

JAW_TARGET = (1.0, 0.0, 0.0)

# The bar's world axis: upright in both cups, along -y for the exchange so two
# top-down grippers are separated along the bar rather than stacked.
BAR_UP_W = (0.0, 0.0, 1.0)

BAR_FLAT_W = (0.0, -1.0, 0.0)

# Commanded approach speed, m/s: the IK is given a point this far ahead of the
# TCP rather than the goal, so a distant goal does not set the initial rate.
APPROACH_SPEED = 0.10

# Wrist speed limit, deg/s; capped together with the position limit.
APPROACH_ANG_SPEED = 40.0

HANDOVER_XY = (0.26, 0.0)

HANDOVER_FOOT_Z = TABLE_TOP_Z + 0.075   # the bar's FOOT during the exchange

# Bar foot height while carried: just clear of the rim. The gel rides a grasp
# height above this, so a larger value pushes the TCP out of reach.
CARRY_Z = CUP_RIM_Z + 0.010

# Released with its foot just inside the destination bore; it drops the rest.
RELEASE_BELOW_RIM = 0.005

CLOSE_WIDTH = 0.0015                 # commanded past contact; the pads stop it



def _bar_pq(h):
    bar = h.scene["bar"]
    return (bar.data.root_pos_w[0].detach().cpu().numpy(),
            bar.data.root_quat_w[0].detach().cpu().numpy())


def _bar_axis(h):
    return _axis(_bar_pq(h)[1], (0.0, 0.0, 1.0))


def _bar_foot(h):
    p, q = _bar_pq(h)
    return _tip(p, q, BAR_LENGTH)


def _axis_err(h, want) -> float:
    """How far the bar's axis is from `want`, degrees. A line, so unsigned."""
    return float(_np.degrees(_np.arccos(_np.clip(
        abs(float(_bar_axis(h) @ _np.asarray(want))), 0.0, 1.0))))


def _hold_bar(h, bar_axis_target):
    """Wrist orientation for a hand that IS holding the bar: jaws across world
    x, the bar's own axis on `bar_axis_target`.

    With something in the jaws the wrist is fully determined by the bar's
    axis; this is what turns the bar flat for the exchange and upright again
    for the drop.
    """
    q_now = h.robot.data.body_state_w[0:1, h.ch["ee_body"], 3:7]
    R = _mat(q_now[0].detach().cpu().numpy())
    q = _frame(q_now[0].detach().cpu().numpy(),
               _bar_axis(h), R @ _np.asarray(TCP_JAW_L),
               bar_axis_target, JAW_TARGET)
    return _torch.as_tensor(q, dtype=_torch.float32, device=q_now.device)


class HandoverCupTransferTask(Task):
    name = "handover_cup_transfer"
    instruction = "hand the bar to the other arm and seat it in the cup"
    report_name = "handover_cup_transfer_scripted.json"
    knobs = {
        "cup_jitter": Knob(0.010,
                           "per-episode cup position jitter, metres"),
        "spawn_yaw_jitter": Knob(SPAWN_YAW_JITTER,
                                 "the bar's yaw in the source cup, degrees"),
        "left_grasp_h": Knob(LEFT_GRASP_H,
                             "where the giving gel grips, metres up from the "
                             "bar's foot"),
        "right_grasp_h": Knob(RIGHT_GRASP_H,
                              "where the receiving gel grips, metres up from the "
                              "bar's foot"),
        "min_seat": Knob(MIN_SEAT, "success bar: bar foot this far below the "
                                   "destination rim, metres"),
    }
    phases = [
        # -- left: top-down out of the source cup ----------------------------
        ("home", 4.0), ("above_bar", 4.0), ("descend", 3.0), ("close_left", 3.5),
        ("lift", 3.0),
        # -- turn the bar flat and present it --------------------------------
        ("turn_flat", 5.0), ("present", 4.0),
        # -- right: top-down onto the horizontal bar, 35 mm along it ---------
        ("above_bar2", 4.0), ("descend2", 3.0), ("close_right", 3.5),
        ("release_left", 1.0), ("part", 3.0),
        # -- right: upright again, into the destination cup ------------------
        ("turn_up", 5.0), ("above_cup", 4.0), ("lower", 3.5), ("release", 1.0),
        ("retreat", 2.5),
    ]
    handover_phase = "above_bar2"
    horizon_s = 35.0
    scene_cam = ((0.95, -0.62, 0.52), (0.26, 0.0, TABLE_TOP_Z + 0.06))

    _LEFT_PHASES = ("home", "above_bar", "descend", "close_left", "lift",
                    "turn_flat", "present", "part")
    # Speed-capped phases: only those that end beside something. On the long
    # moves the cap would keep the arm from arriving within the timeout.
    _SLOW_PHASES = ("descend", "close_left", "descend2", "close_right",
                    "lower", "release", "release_left")

    def _arm_for(self, name: str) -> str:
        return "left" if name in self._LEFT_PHASES else "right"

    def speed_for(self, name: str):
        return ((APPROACH_SPEED, APPROACH_ANG_SPEED) if name in self._SLOW_PHASES
                else (None, None))

    # -- scene ----------------------------------------------------------------
    def make_scene_cfg(self, knobs: dict):
        return HandoverCupTransferSceneCfg(num_envs=1, env_spacing=3.0)

    def apply_collision(self, h) -> None:
        apply_ccd("/World/envs/env_0/bar")
        n_sdf = len(apply_sdf_collision("/World/envs/env_0/bar"))
        n_cup = sum(len(apply_triangle_mesh_collision(f"/World/envs/env_0/{c}"))
                    for c in ("cup_src", "cup_dst"))
        print(f"STEP bar {BAR_LENGTH * 1e3:.0f} mm ({BAR_SECTION[0] * 1e3:.0f} x "
              f"{BAR_SECTION[1] * 1e3:.0f} mm), bores "
              f"{CUP_BORE_R[SOURCE] * 2e3:.0f}/{CUP_BORE_R[DEST] * 2e3:.0f} mm; "
              f"top-down grips {LEFT_GRASP_H * 1e3:.0f}/{RIGHT_GRASP_H * 1e3:.0f} mm "
              f"up the bar, {GRIP_SEPARATION * 1e3:.0f} mm apart; sdf on bar: "
              f"{n_sdf}; cup meshes: {n_cup}", flush=True)

    # -- episode --------------------------------------------------------------
    def reset(self, h, rng, ep, knobs) -> dict:
        h.unlatch()
        h.reset_robot()
        h.select_arm("left")

        jit = float(knobs["cup_jitter"])
        cups = {}
        for key, prim in ((SOURCE, "cup_src"), (DEST, "cup_dst")):
            xy = _np.array(CUP_XY[key]) + (rng.uniform(-jit, jit, 2) if jit else 0.0)
            h.place(h.scene[prim], xy, TABLE_TOP_Z)
            cups[key] = (float(xy[0]), float(xy[1]))

        bar = h.scene["bar"]
        yj = float(knobs["spawn_yaw_jitter"])
        for attempt in range(4):
            b_xy = _np.array(cups[SOURCE]) + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
            yaw = _np.radians(rng.uniform(-yj, yj)) / 2.0
            h.place(bar, b_xy, BAR_FOOT_Z + BAR_LENGTH / 2 + 0.001,
                    quat=(float(_np.cos(yaw)), 0.0, 0.0, float(_np.sin(yaw))))
            for _ in range(40):
                h.step_ik_toward(h.tcp_pos(), OPEN_WIDTH)
                h.step()
            if _tilt(_bar_pq(h)[1]) < 6.0:
                break
            print(f"STEP ep{ep}: bar did not seat (tilt {_tilt(_bar_pq(h)[1]):.1f} "
                  f"deg), re-dropping (attempt {attempt + 1})", flush=True)
        else:
            raise RetryReset("bar would not stand in the source cup")

        p, _q = _bar_pq(h)
        return {
            "side": "right",
            "cups": cups, "bar_xy0": (float(p[0]), float(p[1])),
            "left_h": float(knobs["left_grasp_h"]),
            "right_h": float(knobs["right_grasp_h"]),
            "grasp_h": None, "flat_quat": None, "up_quat": None,
            "handed_over": False, "seat_now": 0.0,
            "sep_at_transfer_mm": None, "tilt_at_transfer_deg": None,
            "flat_err_deg": None, "up_err_deg": None,
            "area_by_phase": {},
        }

    # -- probes ---------------------------------------------------------------
    def _seat(self, h, ctx) -> float:
        p, q = _bar_pq(h)
        return _seat(p, q, length=BAR_LENGTH, cup_xy=ctx["cups"][DEST],
                     bore_r=CUP_BORE_R[DEST], rim_z=CUP_RIM_Z)

    def signals(self, h, ctx) -> dict:
        p, q = _bar_pq(h)
        s = self._seat(h, ctx)
        ctx["seat_now"] = s
        return {"areas": h.pad_areas(), "seat": s, "tilt": _tilt(q),
                "bar_pos": p.copy()}

    def _grasp_h(self, ctx) -> float:
        """How far the right gel is from the bar's foot ALONG THE BAR, measured
        once the receiving grasp closed and nominal until then."""
        return ctx["grasp_h"] if ctx.get("grasp_h") else ctx["right_h"]

    # -- phase machine --------------------------------------------------------
    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        h.select_arm(self._arm_for(name))
        dst = ctx["cups"][DEST]
        b_xy = ctx["bar_xy0"]
        lh, rh = ctx["left_h"], ctx["right_h"]
        grip, quat = CLOSE_WIDTH, None

        # -- left: top-down pick ----------------------------------------------
        if name == "home":
            grip = OPEN_WIDTH
            tgt = [b_xy[0], b_xy[1], BAR_FOOT_Z + lh + 0.090]
        elif name == "above_bar":
            grip = OPEN_WIDTH
            tgt = [b_xy[0], b_xy[1], BAR_FOOT_Z + lh + 0.045]
        elif name == "descend":
            grip = OPEN_WIDTH
            tgt = [b_xy[0], b_xy[1], BAR_FOOT_Z + lh]
        elif name == "close_left":
            grip = OPEN_WIDTH + (CLOSE_WIDTH - OPEN_WIDTH) * min(1.0, t_phase / 1.2)
            tgt = [b_xy[0], b_xy[1], BAR_FOOT_Z + lh]
        elif name == "lift":
            tgt = [b_xy[0], b_xy[1], CARRY_Z + lh]

        # -- turn the bar flat, then present it --------------------------------
        elif name == "turn_flat":
            quat = _hold_bar(h, BAR_FLAT_W)
            tgt = [HANDOVER_XY[0], HANDOVER_XY[1], HANDOVER_FOOT_Z + lh]
        elif name == "present":
            # Wrist frozen at the turn's result. Rebuilding it from the bar's
            # measured axis each step is a feedback loop: slip moves the
            # wrist, which slips the bar further.
            quat = ctx["flat_quat"]
            tgt = [HANDOVER_XY[0], HANDOVER_XY[1], HANDOVER_FOOT_Z + lh]

        # -- right: top-down onto the horizontal bar ---------------------------
        elif name in ("above_bar2", "descend2", "close_right"):
            # `right_h` along the bar from its MEASURED foot, and the bar is
            # flat, so that offset is in y rather than z. Aimed at the bar as
            # it actually is: the left arm's IK lands where it lands.
            foot = _bar_foot(h)
            at = foot + _bar_axis(h) * rh
            up = {"above_bar2": 0.055, "descend2": 0.0, "close_right": 0.0}[name]
            grip = (OPEN_WIDTH if name != "close_right"
                    else OPEN_WIDTH + (CLOSE_WIDTH - OPEN_WIDTH) * min(1.0, t_phase / 1.2))
            tgt = [float(at[0]), float(at[1]), float(at[2]) + up]
        elif name == "release_left":
            # Nothing moves: both arms latched, the left simply opens.
            h.hold_grip("left", OPEN_WIDTH)
            tp = h.tcp_pos()
            tgt = [tp[0], tp[1], float(tp[2])]
        elif name == "part":
            grip = OPEN_WIDTH
            tp = h.tcp_pos()
            tgt = [float(tp[0]), float(tp[1]) + 0.12, float(tp[2]) + 0.07]

        # -- right: upright, then into the cup ---------------------------------
        elif name == "turn_up":
            quat = _hold_bar(h, BAR_UP_W)
            tgt = [HANDOVER_XY[0], HANDOVER_XY[1], CARRY_Z + self._grasp_h(ctx)]
        elif name == "above_cup":
            quat = ctx["up_quat"]
            tgt = [dst[0], dst[1], CARRY_Z + self._grasp_h(ctx)]
        elif name == "lower":
            quat = ctx["up_quat"]
            tgt = [dst[0], dst[1],
                   CUP_RIM_Z - RELEASE_BELOW_RIM + self._grasp_h(ctx)]
        elif name == "release":
            grip = OPEN_WIDTH
            tp = h.tcp_pos()
            tgt = [tp[0], tp[1], float(tp[2])]
        else:                                          # retreat
            grip = OPEN_WIDTH
            tp = h.tcp_pos()
            tgt = [float(tp[0]), float(tp[1]),
                   CARRY_Z + self._grasp_h(ctx) + 0.06]
        return tgt, grip, quat

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        areas = sig["areas"]
        ctx["area_by_phase"][name] = max(ctx["area_by_phase"].get(name, 0.0),
                                         sum(areas.values()))
        left_pads = ("left_left_finger_tactile", "left_right_finger_tactile")
        right_pads = ("right_left_finger_tactile", "right_right_finger_tactile")

        if name in ("home", "above_bar", "lift", "above_bar2", "above_cup",
                    "part", "retreat"):
            return h.tcp_err(target) < 0.015
        if name in ("descend", "descend2"):
            # Tolerance OR time: the grasp does not need tight convergence,
            # and a tighter tolerance than the arm reaches would time out.
            return h.tcp_err(target) < 0.012 or t_phase > 2.0
        if name == "close_left":
            return t_phase > 1.5 and sum(areas[p] for p in left_pads) > 3.0
        if name in ("turn_flat", "turn_up"):
            # Judged on the bar's measured axis, the controlled quantity, not
            # on the wrist command.
            want = BAR_FLAT_W if name == "turn_flat" else BAR_UP_W
            err = _axis_err(h, want)
            if t_phase > 1.0 and err < 10.0 and h.tcp_err(target) < 0.015:
                key = "flat_quat" if name == "turn_flat" else "up_quat"
                ctx[key] = h.robot.data.body_state_w[0, h.ch["ee_body"], 3:7].clone()
                ctx["flat_err_deg" if name == "turn_flat" else "up_err_deg"] = round(err, 2)
                return True
            return False
        if name == "present":
            if h.tcp_err(target) < 0.010:
                h.latch()
                return True
            return False
        if name == "close_right":
            if t_phase > 1.3 and sum(areas[p] for p in right_pads) > 3.0:
                h.latch()
                ctx["handed_over"] = True
                foot = _bar_foot(h)
                ctx["grasp_h"] = float(_np.linalg.norm(h.tcp_pos() - foot))
                lg, rg = h.gel_centres("left"), h.gel_centres("right")
                if lg is not None and rg is not None:
                    ctx["sep_at_transfer_mm"] = round(float(_np.linalg.norm(
                        0.5 * (lg[0] + lg[1]) - 0.5 * (rg[0] + rg[1]))) * 1e3, 1)
                ctx["tilt_at_transfer_deg"] = round(sig["tilt"], 2)
                return True
            return False
        if name == "release_left":
            return t_phase > 0.6
        if name == "lower":
            return h.tcp_err(target) < 0.006
        if name == "release":
            return t_phase > 0.6
        return False

    def handover_ok(self, h, ctx, sig) -> bool:
        """The left must be holding the bar, flat, at the transfer pose."""
        left = ("left_left_finger_tactile", "left_right_finger_tactile")
        return (sum(sig["areas"][p] for p in left) > 3.0
                and _axis_err(h, BAR_FLAT_W) < 15.0)

    def should_terminate(self, ctx, hist) -> bool:
        from sim.harness import CTRL_FPS
        s = hist.get("seat", [])
        n = 2 * CTRL_FPS
        return len(s) >= n and all(x >= MIN_SEAT for x in s[-n:])

    # -- scoring --------------------------------------------------------------
    def build_trace(self, h, ctx, hist) -> _Trace:
        from sim.envs.bimanual_tactile import TACTILE_SENSORS
        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={"bar": _np.asarray(hist["bar_pos"])},
            extra={"seat_depth_m": _np.asarray(hist["seat"]),
                   "bar_tilt_deg": _np.asarray(hist["tilt"])},
        )

    def verdict(self, trace, ctx, knobs):
        from sim.harness import ARM_PADS
        return _verdict(trace, pads=ARM_PADS[ctx["side"]],
                        min_seat_m=float(knobs["min_seat"]))

    def episode_extra(self, ctx, hist) -> dict:
        return {
            "cup_src_xy": list(ctx["cups"][SOURCE]),
            "cup_dst_xy": list(ctx["cups"][DEST]),
            "handed_over": bool(ctx["handed_over"]),
            "turn_flat_err_deg": ctx["flat_err_deg"],
            "turn_up_err_deg": ctx["up_err_deg"],
            "gel_separation_at_transfer_mm": ctx["sep_at_transfer_mm"],
            "bar_tilt_at_transfer_deg": ctx["tilt_at_transfer_deg"],
            "right_grasp_h_mm": (round(ctx["grasp_h"] * 1e3, 1)
                                 if ctx["grasp_h"] else None),
            "final_seat_mm": (round(float(hist["seat"][-1]) * 1e3, 2)
                              if hist.get("seat") else 0.0),
            "max_seat_mm": (round(float(max(hist["seat"])) * 1e3, 2)
                            if hist.get("seat") else 0.0),
            "max_tilt_deg": (round(float(max(hist["tilt"])), 2)
                             if hist.get("tilt") else 0.0),
            "area_peak_by_phase": {k: round(v, 1) for k, v in ctx["area_by_phase"].items()},
        }

    def report_extra(self, knobs) -> dict:
        return {"bar_section_m": list(BAR_SECTION), "bar_length_m": BAR_LENGTH,
                "cup_bore_r_m": dict(CUP_BORE_R), "cup_rim_z_m": CUP_RIM_Z,
                "grasp_heights_m": [float(knobs["left_grasp_h"]),
                                    float(knobs["right_grasp_h"])],
                "handover_foot_z_m": HANDOVER_FOOT_Z,
                "min_seat_m": float(knobs["min_seat"]),
                "cup_jitter_m": float(knobs["cup_jitter"])}


TASK = HandoverCupTransferTask()
