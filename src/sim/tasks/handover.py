"""handover expert.

Phases::

    home_l -> above_sleeve -> orient_l -> descend_l -> close_l -> pinch -> draw
        -> carry_in -> present -> home_r -> standoff -> orient_r -> open_r
        -> slide_on -> receive -> release -> part -> carry_out -> above_target
        -> set_gap -> align -> seat -> open -> retreat
    close_l -> above_sleeve  no grasp: re-open and try again, at most twice
    draw -> retreat          the sleeve stayed in the source cup
    receive -> retreat       no full receiving grasp: the giver keeps the sleeve
"""

from __future__ import annotations

import numpy as _np

from sim.criteria.probes import axis_from_quat, seated_in_cup, tilt_deg
from sim.criteria.tasks import handover as _verdict
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.handover import (
    DEFAULT_TARGET_BORE,
    GAP_MIN,
    MIN_SEAT,
    REST_GAP,
    SLEEVE_LENGTH,
    SPAWN_GAP,
    gap_window,
)
from sim.envs.bimanual_tactile import TABLE_TOP_Z, TACTILE_SENSORS, gel_gap
from sim.envs.handover import (
    CUP_FLOOR_Z,
    CUP_JITTER,
    CUP_RIM_Z,
    SHELL_PRIMS,
    SHELL_QUAT,
    SLEEVE_DAMPING,
    SLEEVE_SPAWN_Z,
    SLEEVE_STIFFNESS,
    SOURCE,
    TARGET,
    HandoverSceneCfg,
    sample_layout,
)
from sim.tactile import apply_sdf_collision, apply_triangle_mesh_collision
from sim.tasks.scene_utils import (
    enable_cross_arm_collision,
    join_split_shells,
    oblique_grasp_quat,
    retarget_pads,
)
from sim.tasks.spec import Knob, RetryReset, Task

__all__ = ["TASK", "HandoverTask"]

DT = 1.0 / 60.0
# PhysX SDF grid per half-shell. The thin-walled shells cook at this size;
# the tactile sensors' SDF queries fault on the finer default.
SDF_RESOLUTION = 128
SIDE_PADS = {"right": TACTILE_SENSORS[:2], "left": TACTILE_SENSORS[2:]}
GIVER, RECEIVER = "left", "right"
# The sleeve's split axis in a shell's own frame.
SPLIT_AXIS_L = (1.0, 0.0, 0.0)
# Sign of the joint's rest position that opens the shells: the joint's
# position grows as the +x shell moves toward the -x one, so opening is
# negative.
REST_SIGN = -1.0

# Controller parameters.
# Jaw command while approaching and releasing, metres per jaw: a 46 mm gel
# gap over the 38-40 mm sleeve.
OPEN_WIDTH = 0.026
# Jaw command while the receiver turns its wrist beside the held sleeve: open
# just enough for the jaw axis to be measured, so fingers not yet across the
# sleeve do not sweep into it.
ORIENT_WIDTH = 0.009
# Runaway floor for the closing ramp, metres per jaw: reaching it means the
# sleeve is not between the jaws.
GRIP_MIN_WIDTH = 0.006

# --- grasp geometry --------------------------------------------------------
# Gel height on the sleeve, metres above its lower rim. Each hand holds as
# near the sleeve's middle as the other allows, so it hangs nearly upright
# in the gel: the giver above the source rim, the receiver above the target
# rim once seated, the two far enough apart that the hands' fingers do not
# touch.
GIVER_GRASP_H = 0.095
RECEIVER_GRASP_H = 0.045
# Blade heading per arm, degrees in the world xy plane: 30 degrees in toward
# the centreline from straight ahead, where the wrist's sixth joint keeps its
# margin; heading straight across puts it on its limit.
GRASP_YAW_DEG = {"left": -30.0, "right": 30.0}
# The pick comes straight down onto the standing sleeve from this far above
# its grasp point, metres.
PICK_ABOVE = 0.030
# The receiver turns its wrist this far back from its grasp point along the
# horizontal part of its blade, at the grasp's height, then slides in level:
# its fingers stay below the giver's, which reach down toward the sleeve
# from above.
RECEIVE_BACKOFF = 0.080
HOME_ABOVE_PREGRASP = 0.030   # m above the pre-grasp point where `home` ends
# How far the hand rises to draw the sleeve out of the source cup, beyond the
# cup's depth, metres; the sleeve slips a few millimetres in the grasp as its
# spring re-opens it out of the bore. The draw is complete once the sleeve's
# lower rim is DRAWN_CLEAR above the cup's rim.
DRAW_CLEAR = 0.008
DRAWN_CLEAR = 0.003
# Where the upright sleeve's axis stands during the exchange, metres: on the
# centreline, where both arms keep joint margin, with its lower rim this far
# above the table.
EXCHANGE_XY = (0.22, 0.0)
EXCHANGE_BOTTOM_ABOVE_TABLE = 0.015
# The carried sleeve's lower rim this far above the target rim before it is
# lowered in.
SEAT_APPROACH_CLEAR = 0.012
# How a hand withdraws after letting go, metres: straight back along the
# horizontal part of its blade, so its open fingers slide off the sleeve's
# sides; rising instead drags a seated sleeve out of the cup, and drives the
# giver's wrist onto a joint limit.
PART_BACK = 0.060
PART_UP = 0.005
# How long the jaws keep opening after the pads have lifted, before the hand
# moves, s.
RELEASE_OPEN_S = 0.30

# --- grip ------------------------------------------------------------------
# A grasp closes fast until both pads touch. The giver then pinches past that
# contact so the sleeve narrows off the source bore before it is drawn; the
# receiver holds where it touched until the giver lets go, then servos the
# shells' MEASURED separation to its target, letting the spring open the
# sleeve into the target's window.
CLOSE_RATE_FAST = 0.010       # m/s per jaw, until both pads touch
PINCH_RATE = 0.004            # m/s per jaw
# The gel gives under the squeeze, so jaw travel past contact closes the
# shells by far less; the pinch runs on the MEASURED separation, with the
# jaw command stopped this far past contact.
PINCH_MAX = 0.005             # m per jaw
GAP_RATE = 0.004              # m/s per jaw, the servo's slew limit
GAP_GAIN = 0.3                # jaw command change per metre of separation error, per step
GAP_TOL = 0.0005              # m; the separation counts as held inside this
AREA_CONTACT = 5.0            # mm2 per pad; below this a pad is not bearing load
OPEN_RATE = 0.040             # m/s per jaw
# How long a grasp holds, with both pads bearing, before it counts as closed, s.
GRIP_HOLD_S = 0.30
RECEIVE_HOLD_S = 0.20
# Contact area each receiving pad must bear before the giver lets go, mm2.
RECEIVE_AREA_MM2 = 60.0

# --- motion ----------------------------------------------------------------
# Commanded TCP speed limits, m/s. The command point advances from the
# previous command, so a distant goal does not set the initial rate.
FREE_SPEED = 0.25
APPROACH_SPEED = 0.10         # the last few centimetres into a grasp
DRAW_SPEED = 0.03             # drawing the sleeve out of the source bore
# The giver carries the sleeve by its upper part; slow enough that it does
# not swing out of upright in the gel.
CARRY_SPEED = 0.08
# The receiver carries the sleeve by its lower part only, so it moves slowly
# and keeps the grip it took until the sleeve is over the target.
CARRY_HELD_SPEED = 0.06
SEAT_SPEED = 0.02             # lowering over the key

# --- phase completion ------------------------------------------------------
ORIENT_TOL_DEG = 3.0          # wrist within this of its grasp orientation
# Measured gel gap, m, before the grasp orientation is computed: the jaw axis
# has no reliable sign while the gels overlap.
LEVEL_MIN_GEL_GAP = 0.010
PRESENT_MAX_TILT_DEG = 10.0   # the giver must hold the sleeve this upright to present it
SEAT_XY_TOL = 0.004           # m; the seat probe's allowance off the cup's axis
# Aligning the sleeve's lower rim over the target's centre before seating:
# the key's ends leave about a millimetre either way along the split. Pose IK
# settles a few millimetres short and stays there, so the commanded xy carries
# the accumulated miss; completion reads the sleeve's MEASURED rim centre.
ALIGN_XY_TOL = 0.001          # m
XY_BIAS_GAIN = 0.05           # per step
XY_BIAS_MAX = 0.012           # m
# The sleeve counts as over the target while its axis is inside the bore and
# its lower rim at most this far above the rim, metres.
OVER_TARGET_CLEAR = 0.030

# --- episode ---------------------------------------------------------------
SETTLE_STEPS = 30             # steps held at HOME before t = 0
SPAWN_MAX_TILT_DEG = 5.0      # a sleeve leaning further after the settle is re-drawn
MAX_REGRASPS = 2


def _shell_pq(h, prim: str):
    obj = h.scene[prim]
    return (obj.data.root_pos_w[0].detach().cpu().numpy().copy(),
            obj.data.root_quat_w[0].detach().cpu().numpy().copy())


def _sleeve_pq(h):
    """The sleeve's pose: midway between its shells, oriented as shell a."""
    pa, qa = _shell_pq(h, SHELL_PRIMS[0])
    pb, _qb = _shell_pq(h, SHELL_PRIMS[1])
    return 0.5 * (pa + pb), qa


def _sleeve_point(h, above_rim: float) -> _np.ndarray:
    """World position on the sleeve's axis `above_rim` metres above its lower rim."""
    p, q = _sleeve_pq(h)
    up = axis_from_quat(q, (0.0, 0.0, 1.0))
    up = up if up[2] >= 0.0 else -up
    return p + up * (above_rim - SLEEVE_LENGTH / 2.0)


def _gap(h) -> float:
    """Shell separation along the sleeve's split axis, metres."""
    pa, qa = _shell_pq(h, SHELL_PRIMS[0])
    pb, _qb = _shell_pq(h, SHELL_PRIMS[1])
    return float(_np.dot(pa - pb, axis_from_quat(qa, SPLIT_AXIS_L)))


def _blade(pitch_deg: float, yaw_deg: float) -> _np.ndarray:
    """Unit blade direction at the grasp, from its pitch and heading."""
    th, psi = _np.radians(pitch_deg), _np.radians(yaw_deg)
    return _np.array([_np.cos(th) * _np.cos(psi), _np.cos(th) * _np.sin(psi), -_np.sin(th)])


def _wrist_err_deg(h, quat) -> float:
    """Angle between the TCP orientation and a locked orientation."""
    if quat is None:
        return 180.0
    d = abs(float(_np.dot(h.tcp_quat(), _np.asarray(quat, dtype=float))))
    return float(_np.degrees(2.0 * _np.arccos(min(1.0, d))))


class HandoverTask(Task):
    name = "handover"
    instruction = "move the sleeve from the orange cup into the teal cup, handing it between the arms"
    report_name = "handover_scripted.json"
    knobs = {
        "target_bore_mm": Knob(DEFAULT_TARGET_BORE * 1e3,
                               "bore of the target cup, mm; one of the generated set"),
        "pick_pinch_mm": Knob(0.5, "shell gap closed below the source fit before the draw, mm"),
        "receive_gap_mm": Knob(0.0, "shell gap held for the seat, mm; 0 is the window middle"),
        "grasp_pitch_deg": Knob(30.0, "blade pitch below horizontal at the grasps, deg"),
        "cup_jitter": Knob(CUP_JITTER, "per-episode cup position jitter, metres"),
    }
    phases = [
        # -- left: out of the source cup, across to the exchange --------------
        ("home_l", 4.0), ("above_sleeve", 2.5), ("orient_l", 3.0), ("descend_l", 3.0),
        ("close_l", 3.0), ("pinch", 3.0), ("draw", 3.0), ("carry_in", 5.0),
        ("present", 1.0),
        # -- right: onto the lower part; the left lets go and backs off -------
        ("home_r", 4.0), ("standoff", 2.5), ("orient_r", 3.0), ("open_r", 1.0), ("slide_on", 2.5),
        ("receive", 3.0), ("release", 1.0), ("part", 2.5),
        # -- right: open the sleeve into the window, seat it over the key ------
        ("carry_out", 4.0), ("above_target", 4.0), ("set_gap", 2.0), ("align", 3.0),
        ("seat", 5.0), ("open", 1.5),
        ("retreat", 2.0),
    ]
    # The policy takes control at t = 0: no prologue, the whole episode is
    # recorded, HOME included.
    handover_phase = None
    horizon_s = 45.0
    scene_cam = ((0.95, -0.50, 0.60), (0.22, 0.0, TABLE_TOP_Z + 0.06))

    _ARM = {"home_l": "left", "above_sleeve": "left", "orient_l": "left",
            "descend_l": "left", "close_l": "left", "pinch": "left", "draw": "left",
            "carry_in": "left", "present": "left",
            "home_r": "right", "standoff": "right", "orient_r": "right", "open_r": "right",
            "slide_on": "right", "receive": "right", "release": "right",
            "part": "left",
            "carry_out": "right", "above_target": "right", "set_gap": "right", "align": "right",
            "seat": "right",
            "open": "right", "retreat": "right"}
    # Phases whose goal is read once, when they start: those that begin from
    # wherever the arm is, and the receiver's approach, which would otherwise
    # chase the sleeve if it moved.
    _FROZEN_GOAL = ("close_l", "pinch", "draw", "home_r", "standoff", "orient_r",
                    "open_r", "slide_on", "receive", "part", "set_gap")

    # ---- scene -------------------------------------------------------------
    def make_scene_cfg(self, knobs: dict):
        cfg = HandoverSceneCfg(num_envs=1, env_spacing=3.0)
        self._bore = cfg.set_target_bore(float(knobs["target_bore_mm"]) * 1e-3)
        return cfg

    def apply_collision(self, h) -> None:
        n_sdf = sum(len(apply_sdf_collision(f"/World/envs/env_0/{p}", sdf_resolution=SDF_RESOLUTION))
                    for p in SHELL_PRIMS)
        n_tri = sum(len(apply_triangle_mesh_collision(f"/World/envs/env_0/{c}"))
                    for c in ("cup_source", "cup_target"))
        # The two hands work on one sleeve a few centimetres apart: let them
        # collide with each other.
        n_pairs = enable_cross_arm_collision("/World/envs/env_0/Robot")
        # The shells' spring, authored with the scene: a joint that first
        # appears at a reset kicks the standing sleeve out of its cup.
        self._author_joint()
        lo, hi = gap_window(self._bore)
        print(f"STEP cross-arm link pairs colliding: {n_pairs}", flush=True)
        print(f"STEP sdf on shells: {n_sdf}; cup meshes: {n_tri}; target bore "
              f"{self._bore * 1e3:.1f} mm, gap window {lo * 1e3:.2f}-{hi * 1e3:.2f} mm", flush=True)

    # ---- episode -----------------------------------------------------------
    def _author_joint(self) -> None:
        """The spring between the shells, relaxed at REST_GAP and stopped at
        GAP_MIN, before `sim.reset()` so it is built with the scene."""
        a, b = (f"/World/envs/env_0/{p}" for p in SHELL_PRIMS)
        opening = REST_GAP - SPAWN_GAP
        closing = SPAWN_GAP - GAP_MIN
        lo, hi = sorted((REST_SIGN * opening, -REST_SIGN * closing))
        join_split_shells(a, b, stiffness=SLEEVE_STIFFNESS, damping=SLEEVE_DAMPING,
                          break_force=float("inf"), travel=opening, gap=SPAWN_GAP,
                          axis="X", name="squash_joint", rest=REST_SIGN * opening,
                          limits=(lo, hi))

    def _retarget(self, h, side: str) -> None:
        """Each of `side`'s pads senses the shell it faces: the pad further
        along the split axis, toward shell a, senses shell a."""
        names = list(h.robot.data.body_names)
        _pa, qa = _shell_pq(h, SHELL_PRIMS[0])
        split = axis_from_quat(qa, SPLIT_AXIS_L)
        along = [float(h.robot.data.body_state_w[0, names.index(f"openarm_{p}_finger"), 0:3]
                       .detach().cpu().numpy() @ split)
                 for p in (f"{side}_left", f"{side}_right")]
        first_plus = along[0] > along[1]
        pads = SIDE_PADS[side]
        retarget_pads({pads[0]: h.sensors[pads[0]]}, SHELL_PRIMS[0] if first_plus else SHELL_PRIMS[1])
        retarget_pads({pads[1]: h.sensors[pads[1]]}, SHELL_PRIMS[1] if first_plus else SHELL_PRIMS[0])

    def reset(self, h, rng, ep, knobs) -> dict:
        h.unlatch()
        h.select_arm(GIVER)
        layout = sample_layout(rng, float(knobs["cup_jitter"]))
        for key, prim in ((SOURCE, "cup_source"), (TARGET, "cup_target")):
            h.place(h.scene[prim], layout[key], TABLE_TOP_Z)
        sx, sy = layout[SOURCE]
        h.place(h.scene[SHELL_PRIMS[0]], (sx, sy + SPAWN_GAP / 2), SLEEVE_SPAWN_Z, quat=SHELL_QUAT)
        h.place(h.scene[SHELL_PRIMS[1]], (sx, sy - SPAWN_GAP / 2), SLEEVE_SPAWN_Z, quat=SHELL_QUAT)
        # As at scene build: the right pads each sense one shell, the left
        # pads the same shell. Rebuilt every reset.
        retarget_pads({SIDE_PADS["right"][0]: h.sensors[SIDE_PADS["right"][0]]}, SHELL_PRIMS[0])
        retarget_pads({SIDE_PADS["right"][1]: h.sensors[SIDE_PADS["right"][1]]}, SHELL_PRIMS[1])
        retarget_pads({p: h.sensors[p] for p in SIDE_PADS["left"]}, SHELL_PRIMS[0])
        h.reset_robot()
        # Hold HOME exactly, jaws shut, while the sleeve settles against the
        # source bore: the first recorded frame is the robot's HOME.
        h.hold_default(SETTLE_STEPS)
        _p, q = _sleeve_pq(h)
        if tilt_deg(q) > SPAWN_MAX_TILT_DEG:
            raise RetryReset(f"sleeve settled at {tilt_deg(q):.1f} deg in the source cup")
        lo, hi = gap_window(self._bore)
        want = float(knobs["receive_gap_mm"]) * 1e-3
        return {
            "side": RECEIVER,
            "layout": layout,
            "pitch": float(knobs["grasp_pitch_deg"]),
            "pinch": float(knobs["pick_pinch_mm"]) * 1e-3,
            "gap_target": want if want > 0.0 else 0.5 * (lo + hi),
            "phase_i": 0,
            "grip": {"left": 0.0, "right": 0.0},
            "contact": {"left": None, "right": None},   # measured jaw at first contact
            "held_at": {"left": None, "right": None},
            "received": False,
            "drawn": False,
            "quat": {"left": None, "right": None},      # locked grasp orientation
            "cmd_pos": None,
            "goal": None, "prev_goal": None,
            # The held sleeve's grasp point relative to the TCP, measured once
            # each hand has the sleeve.
            "grasp_off": {"left": None, "right": None},
            # How far the sleeve's lower rim hangs below the receiver's TCP,
            # measured once the receiver holds it alone: the seat's depth is
            # set from where the sleeve actually sits in the gel.
            "rim_below": None,
            "regrasps": 0,
            "release_pos": None,
            "xy_bias": _np.zeros(2),
            "gap_after_pinch": None,
            "gap_at_present": None,
            "grip_log": [],
        }

    # ---- probes ------------------------------------------------------------
    def signals(self, h, ctx) -> dict:
        areas = h.pad_areas()
        p, q = _sleeve_pq(h)
        dst = ctx["layout"][TARGET]
        bottom_pt = _sleeve_point(h, 0.0)
        bottom = float(bottom_pt[2])
        over = bool(_np.linalg.norm(p[:2] - _np.asarray(dst)) < self._bore / 2
                    and bottom < CUP_RIM_Z[TARGET] + OVER_TARGET_CLEAR)
        return {
            "areas": areas,
            "side_area": {s: min(areas[x] for x in SIDE_PADS[s]) for s in SIDE_PADS},
            "sleeve_pos": p,
            "tilt": tilt_deg(q),
            "gap": _gap(h),
            "bottom_z": bottom,
            "bottom_xy": bottom_pt[:2].copy(),
            "seat": seated_in_cup(p, q, length=SLEEVE_LENGTH, cup_xy=dst,
                                   rim_z=CUP_RIM_Z[TARGET], xy_tol=SEAT_XY_TOL),
            "over_target": 1.0 if over else 0.0,
            "phase_i": ctx["phase_i"],
        }

    def _touch(self, h, ctx, sig, side: str) -> float | None:
        """Close `side`'s jaws fast until both pads touch; the jaw command, or
        None once contact has been recorded."""
        g = ctx["grip"][side]
        if ctx["contact"][side] is None:
            if sig["side_area"][side] > AREA_CONTACT:
                ctx["contact"][side] = h.grip_measured() if h.ch["side"] == side else g
                return g
            return max(GRIP_MIN_WIDTH, g - CLOSE_RATE_FAST * DT)
        return None

    def _servo_gap(self, ctx, sig, side: str, target: float) -> float:
        """Jaw command that moves the MEASURED shell separation toward `target`."""
        step = float(_np.clip(GAP_GAIN * (sig["gap"] - target), -GAP_RATE * DT, GAP_RATE * DT))
        return min(OPEN_WIDTH, max(GRIP_MIN_WIDTH, ctx["grip"][side] - step))

    def _held_for(self, ctx, sig, side: str, t_phase: float, on: bool) -> float:
        """Seconds `on` has held with both of `side`'s pads bearing; zero until then."""
        if not (on and sig["side_area"][side] > AREA_CONTACT):
            ctx["held_at"][side] = None
            return 0.0
        if ctx["held_at"][side] is None:
            ctx["held_at"][side] = t_phase
        return t_phase - ctx["held_at"][side]

    def _receiver_grasp(self, h, ctx):
        """(grasp point, pre-grasp point) for the receiver, read from the sleeve."""
        grasp = _sleeve_point(h, RECEIVER_GRASP_H)
        return grasp, grasp - _blade(0.0, GRASP_YAW_DEG[RECEIVER]) * RECEIVE_BACKOFF

    def _over(self, ctx, side: str, axis_xy) -> list:
        """TCP xy that puts the held sleeve's axis over `axis_xy`."""
        off = ctx["grasp_off"][side]
        off = _np.zeros(2) if off is None else off
        return [float(axis_xy[0] - off[0]), float(axis_xy[1] - off[1])]

    def _advance(self, h, ctx, goal, speed: float) -> list:
        """Advance the commanded point toward `goal` by at most `speed` * DT."""
        if ctx["cmd_pos"] is None:
            ctx["cmd_pos"] = h.tcp_pos().astype(float)
        delta = _np.asarray(goal, dtype=float) - ctx["cmd_pos"]
        dist = float(_np.linalg.norm(delta))
        step = speed * DT
        ctx["cmd_pos"] = (ctx["cmd_pos"] + delta * (step / dist) if dist > step
                          else _np.asarray(goal, dtype=float))
        return ctx["cmd_pos"].tolist()

    # ---- phases ------------------------------------------------------------
    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        arm = self._ARM[name]
        names = [n for n, _ in self.phases]
        if t_phase == 0.0:
            h.latch()                      # the arm that just moved holds where it is
            ctx["cmd_pos"] = None
            ctx["phase_i"] = names.index(name)
            ctx["prev_goal"], ctx["goal"] = ctx["goal"], None
        h.select_arm(arm)
        if t_phase == 0.0 and name in ("descend_l", "slide_on"):
            # The wrist is placed, so which pad faces which shell is settled
            # for the rest of the episode.
            self._retarget(h, arm)
        grip = ctx["grip"]
        quat = ctx["quat"][arm]
        src, dst = ctx["layout"][SOURCE], ctx["layout"][TARGET]
        speed = FREE_SPEED
        tcp = h.tcp_pos()
        pick_z = CUP_FLOOR_Z + GIVER_GRASP_H
        rim_below = RECEIVER_GRASP_H if ctx["rim_below"] is None else ctx["rim_below"]

        # -- left: pick from the source cup ---------------------------------
        if name == "home_l":
            # Jaws still shut, as HOME leaves them; `above_sleeve` opens them.
            grip["left"] = 0.0
            tgt = [src[0], src[1], pick_z + PICK_ABOVE + HOME_ABOVE_PREGRASP]
        elif name == "above_sleeve":
            grip["left"] = OPEN_WIDTH
            tgt = [src[0], src[1], pick_z + PICK_ABOVE]
        elif name == "orient_l":
            grip["left"] = OPEN_WIDTH
            quat = self._orient(h, ctx, "left")
            tgt = [src[0], src[1], pick_z + PICK_ABOVE]
        elif name == "descend_l":
            grip["left"] = OPEN_WIDTH
            speed = APPROACH_SPEED
            p = _sleeve_point(h, GIVER_GRASP_H)
            tgt = [float(p[0]), float(p[1]), pick_z]
        elif name == "close_l":
            g = self._touch(h, ctx, sig, "left")
            grip["left"] = grip["left"] if g is None else g
            tgt = list(tcp)
        elif name == "pinch":
            # Squeeze until the shells close below their fit in the bore, so
            # the draw does not drag the sleeve along the wall.
            floor = max(GRIP_MIN_WIDTH, ctx["contact"]["left"] - PINCH_MAX)
            if sig["gap"] > SPAWN_GAP - ctx["pinch"]:
                grip["left"] = max(floor, grip["left"] - PINCH_RATE * DT)
            tgt = list(tcp)
        elif name == "draw":
            # Up by the cup's depth and the clearance from wherever the hand
            # took the sleeve.
            speed = DRAW_SPEED
            tgt = [float(tcp[0]), float(tcp[1]),
                   float(tcp[2]) + (CUP_RIM_Z[SOURCE] - CUP_FLOOR_Z) + DRAW_CLEAR]
        elif name == "carry_in":
            speed = CARRY_SPEED
            xy = self._over(ctx, "left", EXCHANGE_XY)
            tgt = [xy[0], xy[1], TABLE_TOP_Z + EXCHANGE_BOTTOM_ABOVE_TABLE + GIVER_GRASP_H]
        elif name == "present":
            tgt = ctx["prev_goal"]

        # -- right: take the lower part -----------------------------------------
        elif name == "home_r":
            grip["right"] = 0.0
            _g, pre = self._receiver_grasp(h, ctx)
            tgt = pre.tolist()
        elif name in ("standoff", "orient_r", "open_r"):
            grip["right"] = OPEN_WIDTH if name == "open_r" else ORIENT_WIDTH
            if name == "orient_r":
                quat = self._orient(h, ctx, "right")
            tgt = self._receiver_grasp(h, ctx)[1].tolist()
        elif name == "slide_on":
            grip["right"] = OPEN_WIDTH
            speed = APPROACH_SPEED
            tgt = self._receiver_grasp(h, ctx)[0].tolist()
        elif name == "receive":
            # Close to contact and hold there: the giver still has the
            # sleeve pinched, so the receiver takes it as it is.
            g = self._touch(h, ctx, sig, "right")
            grip["right"] = grip["right"] if g is None else g
            tgt = list(tcp)
        elif name == "release":
            grip["left"] = min(OPEN_WIDTH, grip["left"] + OPEN_RATE * DT)
            h.hold_grip("left", grip["left"])
            tgt = ctx["prev_goal"]
        elif name == "part":
            grip["left"] = OPEN_WIDTH
            h.hold_grip("right", grip["right"])
            back = -_blade(0.0, GRASP_YAW_DEG["left"]) * PART_BACK
            tgt = [float(tcp[0] + back[0]), float(tcp[1] + back[1]), float(tcp[2]) + PART_UP]

        # -- right: open the sleeve into the window, seat it ----------------------
        elif name in ("carry_out", "above_target"):
            # Held as taken; the separation is set once over the target.
            speed = CARRY_HELD_SPEED
            z = CUP_RIM_Z[TARGET] + SEAT_APPROACH_CLEAR + rim_below
            if name == "carry_out":
                tgt = [float(tcp[0]), float(tcp[1]), max(float(tcp[2]), z)]
            else:
                xy = self._over(ctx, "right", dst)
                tgt = [xy[0], xy[1], z]
        elif name == "set_gap":
            # Stationary over the target, the receiver lets the spring open
            # the sleeve to its target separation.
            grip["right"] = self._servo_gap(ctx, sig, "right", ctx["gap_target"])
            tgt = ctx["prev_goal"]
        elif name in ("align", "seat"):
            # Put the sleeve's lower rim centre over the target's centre from
            # its measured position, holding the separation meanwhile.
            grip["right"] = self._servo_gap(ctx, sig, "right", ctx["gap_target"])
            miss = _np.asarray(dst) - sig["bottom_xy"]
            ctx["xy_bias"] = _np.clip(ctx["xy_bias"] + XY_BIAS_GAIN * miss, -XY_BIAS_MAX, XY_BIAS_MAX)
            xy = self._over(ctx, "right", dst)
            xy = [xy[0] + float(ctx["xy_bias"][0]), xy[1] + float(ctx["xy_bias"][1])]
            if name == "align":
                tgt = [xy[0], xy[1], ctx["prev_goal"][2]]
            else:
                speed = SEAT_SPEED
                tgt = [xy[0], xy[1], CUP_FLOOR_Z + rim_below]
        elif name == "open":
            grip["right"] = OPEN_WIDTH
            if ctx["release_pos"] is None:
                ctx["release_pos"] = [float(v) for v in tcp]
            tgt = list(ctx["release_pos"])
        else:                                          # retreat
            grip["right"] = OPEN_WIDTH
            rel = ctx["release_pos"] or [float(v) for v in tcp]
            back = -_blade(0.0, GRASP_YAW_DEG["right"]) * PART_BACK
            tgt = [rel[0] + float(back[0]), rel[1] + float(back[1]), rel[2] + PART_UP]

        if name not in self._FROZEN_GOAL or ctx["goal"] is None:
            ctx["goal"] = [float(v) for v in tgt]
        return self._advance(h, ctx, ctx["goal"], speed), grip[arm], quat

    def _orient(self, h, ctx, side: str):
        """Lock `side`'s grasp orientation once its jaws are open enough for
        the jaw axis to be meaningful; keep it for the rest of the episode."""
        if ctx["quat"][side] is None and gel_gap(h.grip_measured()) >= LEVEL_MIN_GEL_GAP:
            ctx["quat"][side] = oblique_grasp_quat(
                h.tcp_quat(), h.jaw_axis(), h.approach_axis(),
                ctx["pitch"], GRASP_YAW_DEG[side])
        return ctx["quat"][side]

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        arm = self._ARM[name]
        if name in ("close_l", "pinch", "draw", "receive", "release", "carry_out", "align", "seat") \
                and int(round(t_phase / DT)) % 3 == 0:
            ctx["grip_log"].append([
                name, round(t_phase, 3), round(ctx["grip"][arm] * 1e3, 3),
                round(h.grip_measured() * 1e3, 3),
                [round(sig["areas"][p], 1) for p in SIDE_PADS[arm]],
                round(sig["gap"] * 1e3, 3), round(sig["tilt"], 2),
                round(sig["bottom_z"] * 1e3, 1),
                [round(float(v) * 1e3, 2) for v in (_np.asarray(ctx["layout"][TARGET]) - sig["bottom_xy"])]])
        goal = ctx["goal"]
        if name in ("home_l", "above_sleeve", "home_r", "standoff", "part", "retreat"):
            return h.tcp_err(goal) < 0.015
        if name in ("orient_l", "orient_r"):
            done = (ctx["quat"][arm] is not None and t_phase > 0.3
                    and _wrist_err_deg(h, ctx["quat"][arm]) < ORIENT_TOL_DEG
                    and h.tcp_err(goal) < 0.015)
            return done
        if name == "descend_l":
            # Close to the grasp height: the draw's rise is measured from here.
            return h.tcp_err(goal) < 0.002 or t_phase > 2.5
        if name == "open_r":
            return h.grip_measured() >= OPEN_WIDTH - 0.001
        if name == "slide_on":
            return h.tcp_err(goal) < 0.005 or t_phase > 2.0
        if name == "close_l":
            if ctx["contact"]["left"] is None:
                return ctx["grip"]["left"] <= GRIP_MIN_WIDTH + 1e-9
            return self._held_for(ctx, sig, "left", t_phase, True) >= GRIP_HOLD_S
        if name == "pinch":
            done = self._held_for(ctx, sig, "left", t_phase,
                                  sig["gap"] <= SPAWN_GAP - ctx["pinch"]) >= GRIP_HOLD_S
            if done:
                ctx["gap_after_pinch"] = round(sig["gap"] * 1e3, 3)
            return done
        if name == "draw":
            done = sig["bottom_z"] > CUP_RIM_Z[SOURCE] + DRAWN_CLEAR
            if done:
                ctx["drawn"] = True
                ctx["grasp_off"]["left"] = sig["sleeve_pos"][:2] - h.tcp_pos()[:2]
            return done
        if name in ("carry_in", "carry_out", "above_target"):
            return h.tcp_err(goal) < 0.010
        if name == "set_gap":
            return self._held_for(ctx, sig, "right", t_phase,
                                  abs(sig["gap"] - ctx["gap_target"]) <= GAP_TOL) >= GRIP_HOLD_S
        if name == "present":
            if t_phase > 0.5:
                ctx["gap_at_present"] = round(sig["gap"] * 1e3, 3)
                return True
            return False
        if name == "receive":
            if ctx["contact"]["right"] is None:
                return ctx["grip"]["right"] <= GRIP_MIN_WIDTH + 1e-9
            if sig["side_area"]["right"] < RECEIVE_AREA_MM2:
                return False
            done = self._held_for(ctx, sig, "right", t_phase, True) >= RECEIVE_HOLD_S
            if done:
                ctx["received"] = True
            return done
        if name == "release":
            done = (max(sig["areas"][p] for p in SIDE_PADS["left"]) < 1.0
                    and t_phase > RELEASE_OPEN_S)
            if done:
                ctx["grasp_off"]["right"] = sig["sleeve_pos"][:2] - h.tcp_pos()[:2]
                ctx["rim_below"] = float(h.tcp_pos()[2] - sig["bottom_z"])
            return done
        if name == "align":
            miss = float(_np.linalg.norm(_np.asarray(ctx["layout"][TARGET]) - sig["bottom_xy"]))
            return self._held_for(ctx, sig, "right", t_phase, miss <= ALIGN_XY_TOL) >= GRIP_HOLD_S
        if name == "seat":
            return sig["seat"] >= MIN_SEAT + 0.002 or h.tcp_err(goal) < 0.003
        if name == "open":
            return (max(sig["areas"][p] for p in SIDE_PADS["right"]) < 1.0
                    and t_phase > RELEASE_OPEN_S)
        return False

    def next_phase(self, ctx, name):
        if name == "close_l" and ctx["contact"]["left"] is None and ctx["regrasps"] >= MAX_REGRASPS:
            # Never touched the sleeve: nothing to pinch or hand over.
            return "retreat"
        if name == "draw" and not ctx["drawn"]:
            # The sleeve did not come out of the source cup: nothing to hand
            # over, and the receiver must not reach for it there.
            return "retreat"
        if name == "receive" and not ctx["received"]:
            # The receiver did not get a full grasp: the giver keeps the
            # sleeve and the episode ends there, rather than dropping it.
            return "retreat"
        if name == "close_l":
            held = ctx["contact"]["left"] is not None and ctx["grip"]["left"] > GRIP_MIN_WIDTH + 1e-9
            if not held and ctx["regrasps"] < MAX_REGRASPS:
                ctx["regrasps"] += 1
                ctx["contact"]["left"] = None
                ctx["held_at"]["left"] = None
                ctx["grip"]["left"] = OPEN_WIDTH
                return "above_sleeve"
        return None

    def handover_ok(self, h, ctx, sig) -> bool:
        """The giver holds the sleeve upright, clear of the source cup."""
        return sig["side_area"]["left"] > AREA_CONTACT and sig["tilt"] < PRESENT_MAX_TILT_DEG

    def should_terminate(self, ctx, hist) -> bool:
        from sim.harness import CTRL_FPS
        seat, areas = hist.get("seat", []), hist.get("areas", [])
        n = CTRL_FPS
        if len(seat) < n:
            return False
        return (all(s >= MIN_SEAT for s in seat[-n:])
                and all(max(a.values()) < AREA_CONTACT for a in areas[-n:]))

    # ---- scoring -----------------------------------------------------------
    def build_trace(self, h, ctx, hist) -> _Trace:
        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={"sleeve": _np.asarray(hist["sleeve_pos"])},
            extra={"tilt_deg": _np.asarray(hist["tilt"]),
                   "gap_m": _np.asarray(hist["gap"]),
                   "seat_depth_m": _np.asarray(hist["seat"]),
                   "over_target": _np.asarray(hist["over_target"])},
            phase=_np.asarray(hist["phase_i"]),
            phase_names=tuple(n for n, _ in self.phases),
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(trace, bore_m=self._bore)

    def episode_extra(self, ctx, hist) -> dict:
        gap = _np.asarray(hist["gap"])
        return {
            "cup_source_xy": [round(v, 4) for v in ctx["layout"][SOURCE]],
            "cup_target_xy": [round(v, 4) for v in ctx["layout"][TARGET]],
            "gap_target_mm": round(ctx["gap_target"] * 1e3, 3),
            "gap_after_pinch_mm": ctx["gap_after_pinch"],
            "gap_at_present_mm": ctx["gap_at_present"],
            "gap_min_mm": round(float(gap.min()) * 1e3, 3) if gap.size else None,
            "gap_max_mm": round(float(gap.max()) * 1e3, 3) if gap.size else None,
            "regrasps": ctx["regrasps"],
            "received": ctx["received"],
            "drawn": ctx["drawn"],
            "contact_mm": {s: (None if c is None else round(c * 1e3, 2))
                           for s, c in ctx["contact"].items()},
            "final_seat_mm": (round(float(hist["seat"][-1]) * 1e3, 2) if hist.get("seat") else None),
            # phase, t, grip cmd mm, grip measured mm, area per pad mm2, gap mm, tilt deg, bottom z mm,
            # target centre minus rim centre xy mm
            "grip_log": ctx["grip_log"],
        }

    def report_extra(self, knobs) -> dict:
        lo, hi = gap_window(self._bore)
        return {"target_bore_m": self._bore, "gap_window_m": [lo, hi],
                "pick_pinch_mm": float(knobs["pick_pinch_mm"]),
                "receive_gap_mm": float(knobs["receive_gap_mm"]),
                "grasp_pitch_deg": float(knobs["grasp_pitch_deg"]),
                "cup_jitter_m": float(knobs["cup_jitter"])}


TASK = HandoverTask()
