"""cap_unscrew expert.

Phases::

    home_l -> above_bottle -> orient_l -> approach_l -> close_l
        -> home_r -> above_cap -> orient_r -> descend_r -> close_r
        -> twist -> test_lift -> open_r -> unwind
        -> lift -> carry -> above_tray -> lower -> open -> retreat_r
        -> open_l -> retreat_l
    unwind -> close_r        the cap did not come up: grip again and turn once more
    test_lift -> lift        the cap came up with the hand: it is off the thread
    test_lift -> open        the stroke budget is spent: let go and give up
    close_l, close_r         no contact: re-open and approach again, at most twice
"""

from __future__ import annotations

import copy
import math

import numpy as _np

from sim.criteria.cap_unscrew import cap_centre, in_tray
from sim.criteria.cap_unscrew import verdict as _verdict
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.cap_unscrew import BOTTLE_H, CAP_BORE_DEPTH, CAP_H, TRAY_HEIGHT
from sim.envs.bimanual_tactile import TABLE_TOP_Z, TACTILE_SENSORS, gel_gap
from sim.envs.cap_unscrew import (
    BOTTLE_JITTER,
    BOTTLE_SPAWN_Z,
    BOTTLE_SPIN_DAMPING,
    CAP_SPAWN_Z,
    SPINDLE_Z,
    THREAD_ANGLE_SWEEP,
    THREAD_DAMPING,
    TRAY_JITTER,
    CapUnscrewSceneCfg,
    sample_layout,
)
from sim.tactile import apply_sdf_collision, apply_triangle_mesh_collision
from sim.tasks.scene_utils import (
    enable_cross_arm_collision,
    join_spin,
    oblique_grasp_quat,
    set_joint_enabled,
)
from sim.tasks.spec import Knob, RetryReset, Task

__all__ = ["TASK", "CapUnscrewTask"]

DT = 1.0 / 60.0
ENV = "/World/envs/env_0"
# PhysX SDF grid per sensed part.
SDF_RESOLUTION = 128
SIDE_PADS = {"right": TACTILE_SENSORS[:2], "left": TACTILE_SENSORS[2:]}
TURNER, HOLDER = "right", "left"
THREAD_JOINT_NAME = "thread_joint"
THREAD_JOINT = f"{ENV}/{THREAD_JOINT_NAME}"

# Controller parameters.
# Jaw command while approaching and releasing, metres per jaw: a 46 mm gel
# gap over the 38 mm body and the 34 mm cap.
OPEN_WIDTH = 0.026
# Runaway floor for the closing ramp, metres per jaw, per hand: reaching it
# means the part is not between the jaws.
GRIP_MIN_WIDTH = {"left": 0.016, "right": 0.014}

# --- grasp geometry --------------------------------------------------------
# Gel height on the bottle, metres above its base: the body's middle.
HOLD_GRASP_H = 0.045
# Blade heading per hand, degrees in the world xy plane: in toward the
# centreline, where the arm keeps its joint margin. The turner turns further
# in from there, the way the thread opens, and heading plus stroke must stay
# inside the band where neither end of the stroke sits on a joint limit.
HOLD_YAW_DEG = -45.0
TWIST_FROM_DEG = 30.0
# The pre-grasp point is this far back from the grasp point along the blade,
# metres; `home_*` ends this far above the pre-grasp point.
APPROACH_BACKOFF = 0.060
HOME_ABOVE_PREGRASP = 0.030

# --- grip ------------------------------------------------------------------
# A grasp closes fast until both pads touch, then squeezes this far past the
# MEASURED contact position, so the pads carry the thread's torque.
CLOSE_RATE_FAST = 0.010  # m/s per jaw
CLOSE_RATE_SLOW = 0.002  # m/s per jaw, past contact
SQUEEZE = {"left": 0.0025, "right": 0.0025}  # m per jaw past contact
AREA_CONTACT = 5.0  # mm2 per pad; below this a pad is not bearing load
OPEN_RATE = 0.040  # m/s per jaw
GRIP_HOLD_S = 0.30  # s the squeeze holds before a grasp counts as closed
RELEASE_OPEN_S = 0.30  # s the jaws keep opening after the pads lift

# --- turning ---------------------------------------------------------------
# One stroke turns the whole hand this far about the vertical through the
# cap's axis, degrees, at TWIST_RATE_DEG: the heading band the arm's reach
# allows, turned slowly enough for the gels to carry the thread's torque.
STROKE_DEG = 30.0
TWIST_RATE_DEG = 25.0  # deg/s
UNWIND_RATE_DEG = 60.0  # deg/s, jaws open
MAX_STROKES = 8
# After each stroke the hand rises LIFT_TEST for LIFT_TEST_S; the cap is off
# the thread when it rises LIFT_SEEN with it. The commanded rise is several
# times LIFT_SEEN, which the IK lag and the gels' give eat into.
LIFT_TEST = 0.008  # m
LIFT_SEEN = 0.0015  # m
LIFT_TEST_S = 1.0  # s

# --- carrying --------------------------------------------------------------
# The freed cap rises until its skirt is this far above the neck's top.
LIFT_CLEAR = 0.020  # m
# Heading the cap is carried to the tray at, degrees: the one it was gripped
# at. The wrist turns back to it at CARRY_TURN_RATE_DEG, a rate the gels can
# hold the cap through.
CARRY_YAW_DEG = TWIST_FROM_DEG
CARRY_TURN_RATE_DEG = 25.0  # deg/s
# The cap's skirt above the tray's rim on the way over, and when the hand
# lets go: a hand holding the cap by its middle cannot reach the floor
# without putting its fingers inside the tray.
TRAY_CLEAR = 0.015  # m
RELEASE_ABOVE_RIM = 0.010  # m
# Jaw command while letting go, metres per jaw: the gripper's limit, so the
# fingers clear the cap as the hand withdraws.
RELEASE_WIDTH = 0.027
# How far a hand withdraws after letting go, metres: the turner straight up
# out of the tray, whose fingers straddle the cap, the holder back along its
# blade, away from the bottle.
PART_UP = 0.060
PART_BACK = 0.060

# --- motion ----------------------------------------------------------------
FREE_SPEED = 0.25  # m/s
RETREAT_SPEED = 0.05  # leaving a cap standing loose in the tray
APPROACH_SPEED = 0.10  # the last few centimetres into a grasp
LIFT_SPEED = 0.03  # drawing the cap up off the neck
CARRY_SPEED = 0.08
LOWER_SPEED = 0.06

# --- phase completion ------------------------------------------------------
ORIENT_TOL_DEG = 3.0  # wrist within this of its commanded orientation
TWIST_TOL_DEG = 4.0
# Measured gel gap, m, before the grasp orientation is computed: the jaw axis
# has no reliable sign while the gels overlap.
LEVEL_MIN_GEL_GAP = 0.010

# --- episode ---------------------------------------------------------------
SETTLE_STEPS = 30  # steps held at HOME before t = 0
SPAWN_MAX_DRIFT = 0.002  # m; a cap that moved further off the neck is re-drawn
MAX_REGRASPS = 2


def _pq(h, prim: str):
    obj = h.scene[prim]
    return (
        obj.data.root_pos_w[0].detach().cpu().numpy().copy(),
        obj.data.root_quat_w[0].detach().cpu().numpy().copy(),
    )


def _qmul(a, b) -> _np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return _np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _rz(deg: float) -> _np.ndarray:
    """Quaternion of a turn about world z."""
    h = math.radians(deg) / 2.0
    return _np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def _yaw_of(q_ref, q) -> float:
    """Turn of `q` relative to `q_ref` about `q_ref`'s z, degrees in (-180, 180]."""
    w, x, y, z = q_ref
    r = _qmul(_np.array([w, -x, -y, -z]), _np.asarray(q, dtype=float))
    return math.degrees(2.0 * math.atan2(r[3], r[0]))


def _unwrap(prev: float, raw: float) -> float:
    """Continue an accumulated angle with a new reading in (-180, 180]."""
    d = (raw - prev + 180.0) % 360.0 - 180.0
    return prev + d


def _blade(pitch_deg: float, yaw_deg: float) -> _np.ndarray:
    """Unit blade direction at a grasp, from its pitch and heading."""
    th, psi = math.radians(pitch_deg), math.radians(yaw_deg)
    return _np.array([math.cos(th) * math.cos(psi), math.cos(th) * math.sin(psi), -math.sin(th)])


def _wrist_err_deg(h, quat) -> float:
    """Angle between the TCP orientation and a commanded orientation."""
    if quat is None:
        return 180.0
    d = abs(float(_np.dot(h.tcp_quat(), _np.asarray(quat, dtype=float))))
    return float(_np.degrees(2.0 * _np.arccos(min(1.0, d))))


class CapUnscrewTask(Task):
    name = "cap_unscrew"
    instruction = "unscrew the cap from the bottle and put the cap in the tray"
    report_name = "cap_unscrew_scripted.json"
    knobs = {
        "thread_angle_deg": Knob(
            0.0, f"turn that frees the cap, deg; 0 draws one of {THREAD_ANGLE_SWEEP} per episode"
        ),
        "grasp_pitch_deg": Knob(45.0, "blade pitch below horizontal at both grasps, deg"),
        "bottle_jitter": Knob(BOTTLE_JITTER, "per-episode bottle position jitter, metres"),
        "tray_jitter": Knob(TRAY_JITTER, "per-episode tray position jitter, metres"),
    }
    phases = [
        # -- left: hold the bottle ---------------------------------------------
        ("home_l", 4.0),
        ("above_bottle", 2.5),
        ("orient_l", 3.0),
        ("approach_l", 3.0),
        ("close_l", 3.0),
        # -- right: turn the cap until it comes free -----------------------------
        ("home_r", 4.0),
        ("above_cap", 2.5),
        ("orient_r", 3.0),
        ("descend_r", 3.0),
        ("close_r", 3.0),
        ("twist", 4.0),
        ("test_lift", 1.5),
        ("open_r", 1.5),
        ("unwind", 3.0),
        # -- right: into the tray ------------------------------------------------
        ("lift", 3.0),
        ("carry", 4.0),
        ("above_tray", 3.0),
        ("lower", 4.0),
        ("open", 1.5),
        ("retreat_r", 3.0),
        # -- left: let the bottle go ---------------------------------------------
        ("open_l", 1.5),
        ("retreat_l", 2.0),
    ]
    handover_phase = None
    horizon_s = 60.0
    scene_cam = ((0.80, -0.55, 0.55), (0.22, -0.06, TABLE_TOP_Z + 0.08))

    _ARM = {
        "home_l": "left",
        "above_bottle": "left",
        "orient_l": "left",
        "approach_l": "left",
        "close_l": "left",
        "home_r": "right",
        "above_cap": "right",
        "orient_r": "right",
        "descend_r": "right",
        "close_r": "right",
        "twist": "right",
        "test_lift": "right",
        "open_r": "right",
        "unwind": "right",
        "lift": "right",
        "carry": "right",
        "above_tray": "right",
        "lower": "right",
        "open": "right",
        "retreat_r": "right",
        "open_l": "left",
        "retreat_l": "left",
    }
    # Phases whose goal is read once, when they start.
    _FROZEN_GOAL = ("close_l", "close_r", "test_lift", "lift", "open", "retreat_r", "open_l", "retreat_l")

    # ---- scene -------------------------------------------------------------
    def make_scene_cfg(self, knobs: dict):
        return CapUnscrewSceneCfg(num_envs=1, env_spacing=3.0)

    def apply_collision(self, h, *, env_path=ENV) -> dict:
        meshes = {
            p: apply_sdf_collision(f"{env_path}/{p}", sdf_resolution=SDF_RESOLUTION)
            for p in ("bottle", "cap")
        }
        meshes["tray"] = apply_triangle_mesh_collision(f"{env_path}/tray")
        # The two hands work on one bottle a few centimetres apart: let them
        # collide with each other.
        n_pairs = enable_cross_arm_collision(f"{env_path}/Robot")
        # Both joints are authored with the scene: a joint that first appears
        # at a reset kicks its bodies.
        join_spin(
            f"{env_path}/spindle",
            f"{env_path}/bottle",
            local_pos0=(0.0, 0.0, BOTTLE_SPAWN_Z - SPINDLE_Z),
            damping=BOTTLE_SPIN_DAMPING,
            name="spin_joint",
        )
        join_spin(
            f"{env_path}/bottle",
            f"{env_path}/cap",
            local_pos0=(0.0, 0.0, CAP_SPAWN_Z - BOTTLE_SPAWN_Z),
            damping=THREAD_DAMPING,
            name=THREAD_JOINT_NAME,
        )
        print(f"STEP cross-arm link pairs colliding: {n_pairs}", flush=True)
        return meshes

    # ---- episode -----------------------------------------------------------

    settle_seconds = SETTLE_STEPS * DT

    def sample_init_state(self, robot, rng, state_id, knobs):
        layout = sample_layout(rng, float(knobs["bottle_jitter"]), float(knobs["tray_jitter"]))
        angle = float(knobs["thread_angle_deg"])
        if angle <= 0:
            angle = float(THREAD_ANGLE_SWEEP[rng.integers(len(THREAD_ANGLE_SWEEP))])
        objects = {
            name: [*layout["bottle"], z, 1.0, 0.0, 0.0, 0.0]
            for name, z in (("spindle", SPINDLE_Z), ("bottle", BOTTLE_SPAWN_Z), ("cap", CAP_SPAWN_Z))
        }
        objects["tray"] = [*layout["tray"], TABLE_TOP_Z, 1.0, 0.0, 0.0, 0.0]
        return dict(
            id=state_id,
            robot=copy.deepcopy(robot),
            objects=objects,
            task_vars={"side": TURNER, "thread_angle": angle},
        )

    def validate_init_state(self, state):
        variables = state["task_vars"]
        angle = float(variables.get("thread_angle", float("nan")))
        if variables.get("side") != TURNER or not _np.isfinite(angle) or angle <= 0:
            raise ValueError("Cap state requires the turner arm and a positive thread angle")

    def begin_restored_episode(self, h, state, knobs):
        self.validate_init_state(state)
        h.unlatch()
        h.select_arm(HOLDER)
        layout = {key: _pq(h, key)[0][:2].copy() for key in ("bottle", "tray")}
        return self._new_ctx(h, layout, state["task_vars"]["thread_angle"], knobs)

    def init_state_objects(self):
        return ("spindle", "bottle", "cap", "tray")

    def snapshot_init_state(self, h, ctx):
        return {"side": TURNER, "thread_angle": ctx["thread_angle"]}

    def restore_init_state(self, h, state, knobs):
        set_joint_enabled(THREAD_JOINT, True)
        h.hold_default(SETTLE_STEPS)
        return self.begin_restored_episode(h, state, knobs)

    def reset(self, h, rng, ep, knobs) -> dict:
        h.unlatch()
        h.select_arm(HOLDER)
        layout = sample_layout(rng, float(knobs["bottle_jitter"]), float(knobs["tray_jitter"]))
        want = float(knobs["thread_angle_deg"])
        thread = want if want > 0.0 else float(THREAD_ANGLE_SWEEP[rng.integers(len(THREAD_ANGLE_SWEEP))])
        bx, by = layout["bottle"]
        h.place(h.scene["spindle"], (bx, by), SPINDLE_Z)
        h.place(h.scene["bottle"], (bx, by), BOTTLE_SPAWN_Z)
        h.place(h.scene["cap"], (bx, by), CAP_SPAWN_Z)
        h.place(h.scene["tray"], layout["tray"], TABLE_TOP_Z)
        set_joint_enabled(THREAD_JOINT, True)
        h.reset_robot()
        # Hold HOME exactly, jaws shut, while the parts settle: the first
        # recorded frame is the robot's HOME.
        h.hold_default(SETTLE_STEPS)
        cp, _cq = _pq(h, "cap")
        drift = abs(cp[2] - CAP_SPAWN_Z)
        if drift > SPAWN_MAX_DRIFT:
            raise RetryReset(f"cap settled {drift * 1e3:.1f} mm off the neck")
        return self._new_ctx(h, layout, thread, knobs)

    def _new_ctx(self, h, layout, thread, knobs):
        _bp, bq = _pq(h, "bottle")
        return {
            "side": TURNER,
            "layout": layout,
            "thread_angle": thread,
            "thread_joint": f"{getattr(h, 'env_path', ENV)}/{THREAD_JOINT_NAME}",
            "pitch": float(knobs["grasp_pitch_deg"]),
            "phase_i": 0,
            # The world's own state: the thread's accumulated turn, the
            # bottle's spin, and whether the thread has let go. The
            # controller reads none of these.
            "thread_deg": 0.0,
            "bottle_deg": 0.0,
            "bottle_q0": bq,
            "freed": False,
            "grip": {"left": 0.0, "right": 0.0},
            "contact": {"left": None, "right": None},  # measured jaw at first contact
            "held_at": {"left": None, "right": None},
            "quat": {"left": None, "right": None},  # locked grasp orientation
            "grasp_pt": None,  # the turner's grasp point
            "twist_cmd": 0.0,  # deg turned in this stroke
            "strokes": 0,
            "cap_z_turned": None,  # cap height after a stroke
            "cap_up": False,  # the lift test saw the cap rise
            "cap_below": None,  # TCP height above the cap's skirt
            "grasp_off": None,  # cap axis minus TCP, xy
            "release_pos": {"left": None, "right": None},
            "regrasps": 0,
            "cmd_pos": None,
            "goal": None,
            "prev_goal": None,
        }

    # ---- probes ------------------------------------------------------------
    def signals(self, h, ctx) -> dict:
        areas = h.pad_areas()
        cp, cq = _pq(h, "cap")
        bp, bq = _pq(h, "bottle")
        # The thread: the cap's turn on the bottle, accumulated, and the
        # joint let go once it reaches the drawn angle. World behaviour, run
        # here so the expert and the evaluator share it.
        ctx["thread_deg"] = _unwrap(ctx["thread_deg"], _yaw_of(bq, cq))
        ctx["bottle_deg"] = _unwrap(ctx["bottle_deg"], _yaw_of(ctx["bottle_q0"], bq))
        if not ctx["freed"] and ctx["thread_deg"] >= ctx["thread_angle"]:
            set_joint_enabled(ctx["thread_joint"], False)
            ctx["freed"] = True
        tray = (*ctx["layout"]["tray"], TABLE_TOP_Z)
        return {
            "areas": areas,
            "side_area": {s: min(areas[x] for x in SIDE_PADS[s]) for s in SIDE_PADS},
            "cap_pos": cp,
            "cap_quat": cq,
            "bottle_pos": bp,
            "cap_z": float(cp[2]),
            "thread_deg": float(ctx["thread_deg"]),
            "bottle_deg": float(ctx["bottle_deg"]),
            "freed": 1.0 if ctx["freed"] else 0.0,
            "in_tray": 1.0 if in_tray(cap_centre(cp, cq), tray) else 0.0,
            "phase_i": ctx["phase_i"],
        }

    def _close(self, h, ctx, sig, side: str) -> float:
        """Jaw command for a squeezing grasp: fast until both pads touch, then
        slowly down to SQUEEZE past the measured contact position."""
        g = ctx["grip"][side]
        if ctx["contact"][side] is None:
            if sig["side_area"][side] > AREA_CONTACT:
                ctx["contact"][side] = h.grip_measured()
                return g
            return max(GRIP_MIN_WIDTH[side], g - CLOSE_RATE_FAST * h.control_dt)
        floor = max(GRIP_MIN_WIDTH[side], ctx["contact"][side] - SQUEEZE[side])
        return max(floor, g - CLOSE_RATE_SLOW * h.control_dt)

    def _held_for(self, ctx, sig, side: str, t_phase: float, on: bool) -> float:
        """Seconds `on` has held with both of `side`'s pads bearing; zero until then."""
        if not (on and sig["side_area"][side] > AREA_CONTACT):
            ctx["held_at"][side] = None
            return 0.0
        if ctx["held_at"][side] is None:
            ctx["held_at"][side] = t_phase
        return t_phase - ctx["held_at"][side]

    def _hold_point(self, ctx):
        """(grasp point, pre-grasp point) on the bottle's body for the holder."""
        bx, by = ctx["layout"]["bottle"]
        g = _np.array([bx, by, BOTTLE_SPAWN_Z + HOLD_GRASP_H])
        return g, g - _blade(ctx["pitch"], HOLD_YAW_DEG) * APPROACH_BACKOFF

    def _cap_point(self, h, ctx):
        """(grasp point, pre-grasp point) at the cap's centre for the turner."""
        if ctx["grasp_pt"] is None:
            cp, cq = _pq(h, "cap")
            g = cap_centre(cp, cq)
        else:
            g = _np.asarray(ctx["grasp_pt"], dtype=float)
        return g, g - _blade(ctx["pitch"], TWIST_FROM_DEG) * APPROACH_BACKOFF

    def _advance(self, h, ctx, goal, speed: float) -> list:
        """Advance the commanded point toward `goal` by at most `speed` * h.control_dt."""
        if ctx["cmd_pos"] is None:
            ctx["cmd_pos"] = h.tcp_pos().astype(float)
        delta = _np.asarray(goal, dtype=float) - ctx["cmd_pos"]
        dist = float(_np.linalg.norm(delta))
        step = speed * h.control_dt
        ctx["cmd_pos"] = (
            ctx["cmd_pos"] + delta * (step / dist) if dist > step else _np.asarray(goal, dtype=float)
        )
        return ctx["cmd_pos"].tolist()

    def _turned(self, ctx, deg: float):
        """The turner's grasp orientation turned `deg` about the vertical."""
        return _qmul(_rz(deg), ctx["quat"]["right"])

    # ---- phases ------------------------------------------------------------
    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        arm = self._ARM[name]
        names = [n for n, _ in self.phases]
        if t_phase == 0.0:
            h.latch()  # the arm that just moved holds where it is
            ctx["cmd_pos"] = None
            ctx["phase_i"] = names.index(name)
            ctx["prev_goal"], ctx["goal"] = ctx["goal"], None
        h.select_arm(arm)
        grip = ctx["grip"]
        quat = ctx["quat"][arm]
        speed = FREE_SPEED
        tcp = h.tcp_pos()

        # -- left: hold the bottle ------------------------------------------------
        if name in ("home_l", "above_bottle", "orient_l", "approach_l"):
            g, pre = self._hold_point(ctx)
            if name == "home_l":
                grip["left"] = 0.0
                tgt = (pre + [0.0, 0.0, HOME_ABOVE_PREGRASP]).tolist()
            elif name == "approach_l":
                grip["left"] = OPEN_WIDTH
                speed = APPROACH_SPEED
                tgt = g.tolist()
            else:
                grip["left"] = OPEN_WIDTH
                if name == "orient_l":
                    quat = self._orient(h, ctx, "left", HOLD_YAW_DEG)
                tgt = pre.tolist()
        elif name == "close_l":
            grip["left"] = self._close(h, ctx, sig, "left")
            tgt = list(tcp)

        # -- right: turn the cap ---------------------------------------------------
        elif name in ("home_r", "above_cap", "orient_r", "descend_r"):
            g, pre = self._cap_point(h, ctx)
            if name == "home_r":
                grip["right"] = 0.0
                tgt = (pre + [0.0, 0.0, HOME_ABOVE_PREGRASP]).tolist()
            elif name == "descend_r":
                grip["right"] = OPEN_WIDTH
                speed = APPROACH_SPEED
                if ctx["grasp_pt"] is None:
                    ctx["grasp_pt"] = g.tolist()
                tgt = g.tolist()
            else:
                grip["right"] = OPEN_WIDTH
                if name == "orient_r":
                    quat = self._orient(h, ctx, "right", TWIST_FROM_DEG)
                tgt = pre.tolist()
        elif name == "close_r":
            grip["right"] = self._close(h, ctx, sig, "right")
            tgt = list(ctx["grasp_pt"])
        elif name == "twist":
            if t_phase == 0.0:
                ctx["twist_cmd"] = 0.0
                ctx["strokes"] += 1
            ctx["twist_cmd"] = min(STROKE_DEG, ctx["twist_cmd"] + TWIST_RATE_DEG * h.control_dt)
            quat = self._turned(ctx, ctx["twist_cmd"])
            tgt = list(ctx["grasp_pt"])
        elif name == "test_lift":
            if t_phase == 0.0:
                ctx["cap_z_turned"] = sig["cap_z"]
            quat = self._turned(ctx, ctx["twist_cmd"])
            tgt = (_np.asarray(ctx["grasp_pt"]) + [0.0, 0.0, LIFT_TEST]).tolist()
        elif name == "open_r":
            grip["right"] = min(OPEN_WIDTH, grip["right"] + OPEN_RATE * h.control_dt)
            quat = self._turned(ctx, ctx["twist_cmd"])
            tgt = list(ctx["grasp_pt"])
        elif name == "unwind":
            grip["right"] = OPEN_WIDTH
            ctx["twist_cmd"] = max(0.0, ctx["twist_cmd"] - UNWIND_RATE_DEG * h.control_dt)
            quat = self._turned(ctx, ctx["twist_cmd"])
            tgt = list(ctx["grasp_pt"])

        # -- right: into the tray ----------------------------------------------------
        elif name == "lift":
            speed = LIFT_SPEED
            quat = self._turned(ctx, ctx["twist_cmd"])
            skirt_up = (BOTTLE_SPAWN_Z + BOTTLE_H - CAP_BORE_DEPTH) + LIFT_CLEAR
            tgt = [float(tcp[0]), float(tcp[1]), float(tcp[2]) + max(0.0, skirt_up - sig["cap_z"])]
        elif name in ("carry", "above_tray", "lower"):
            speed = CARRY_SPEED if name != "lower" else LOWER_SPEED
            step = CARRY_TURN_RATE_DEG * h.control_dt
            back = CARRY_YAW_DEG - TWIST_FROM_DEG
            ctx["twist_cmd"] = (
                max(back, ctx["twist_cmd"] - step)
                if ctx["twist_cmd"] > back
                else min(back, ctx["twist_cmd"] + step)
            )
            quat = self._turned(ctx, ctx["twist_cmd"])
            tx, ty = ctx["layout"]["tray"]
            off = ctx["grasp_off"] if ctx["grasp_off"] is not None else _np.zeros(2)
            below = ctx["cap_below"] if ctx["cap_below"] is not None else CAP_H / 2
            # The cap crosses to the tray above the neck it came off, which
            # stands higher than the tray's rim.
            transit = (
                max(BOTTLE_SPAWN_Z + BOTTLE_H + LIFT_CLEAR, TABLE_TOP_Z + TRAY_HEIGHT + TRAY_CLEAR) + below
            )
            if name == "carry":
                tgt = [float(tcp[0]), float(tcp[1]), max(float(tcp[2]), transit)]
            elif name == "above_tray":
                tgt = [tx - float(off[0]), ty - float(off[1]), transit]
            else:
                tgt = [
                    tx - float(off[0]),
                    ty - float(off[1]),
                    TABLE_TOP_Z + TRAY_HEIGHT + RELEASE_ABOVE_RIM + below,
                ]
        elif name == "open":
            grip["right"] = min(RELEASE_WIDTH, grip["right"] + OPEN_RATE * h.control_dt)
            if ctx["release_pos"]["right"] is None:
                ctx["release_pos"]["right"] = [float(v) for v in tcp]
            quat = h.tcp_quat()
            tgt = list(ctx["release_pos"]["right"])
        elif name == "retreat_r":
            speed = RETREAT_SPEED
            grip["right"] = RELEASE_WIDTH
            rel = ctx["release_pos"]["right"] or [float(v) for v in tcp]
            tgt = [rel[0], rel[1], rel[2] + PART_UP]

        # -- left: let the bottle go -------------------------------------------------
        elif name == "open_l":
            grip["left"] = min(RELEASE_WIDTH, grip["left"] + OPEN_RATE * h.control_dt)
            if ctx["release_pos"]["left"] is None:
                ctx["release_pos"]["left"] = [float(v) for v in tcp]
            tgt = list(ctx["release_pos"]["left"])
        else:  # retreat_l
            grip["left"] = RELEASE_WIDTH
            rel = ctx["release_pos"]["left"] or [float(v) for v in tcp]
            back = -_blade(ctx["pitch"], HOLD_YAW_DEG) * PART_BACK
            tgt = [rel[i] + float(back[i]) for i in range(3)]

        if name not in self._FROZEN_GOAL or ctx["goal"] is None:
            ctx["goal"] = [float(v) for v in tgt]
        return self._advance(h, ctx, ctx["goal"], speed), grip[arm], quat

    def _orient(self, h, ctx, side: str, yaw_deg: float):
        """Lock `side`'s grasp orientation once its jaws are open enough for
        the jaw axis to be meaningful; keep it for the rest of the episode."""
        if ctx["quat"][side] is None and gel_gap(h.grip_measured()) >= LEVEL_MIN_GEL_GAP:
            ctx["quat"][side] = oblique_grasp_quat(
                h.tcp_quat(), h.jaw_axis(), h.approach_axis(), ctx["pitch"], yaw_deg
            )
        return ctx["quat"][side]

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        arm = self._ARM[name]
        goal = ctx["goal"]
        if name in (
            "home_l",
            "above_bottle",
            "home_r",
            "above_cap",
            "carry",
            "above_tray",
            "retreat_r",
            "retreat_l",
        ):
            return h.tcp_err(goal) < (0.010 if name in ("carry", "above_tray") else 0.015)
        if name in ("orient_l", "orient_r"):
            return (
                ctx["quat"][arm] is not None
                and t_phase > 0.3
                and _wrist_err_deg(h, ctx["quat"][arm]) < ORIENT_TOL_DEG
                and h.tcp_err(goal) < 0.015
            )
        if name in ("approach_l", "descend_r"):
            return h.tcp_err(goal) < 0.003 or t_phase > 2.5
        if name in ("close_l", "close_r"):
            if ctx["contact"][arm] is None:
                return ctx["grip"][arm] <= GRIP_MIN_WIDTH[arm] + 1e-9
            # The squeeze is judged on the command: the compressed gels hold
            # the measured jaw short of it.
            floor = max(GRIP_MIN_WIDTH[arm], ctx["contact"][arm] - SQUEEZE[arm])
            squeezed = ctx["grip"][arm] <= floor + 1e-9
            return self._held_for(ctx, sig, arm, t_phase, squeezed) >= GRIP_HOLD_S
        if name == "twist":
            return (
                ctx["twist_cmd"] >= STROKE_DEG
                and _wrist_err_deg(h, self._turned(ctx, STROKE_DEG)) < TWIST_TOL_DEG
            )
        if name == "test_lift":
            if t_phase < LIFT_TEST_S:
                return False
            ctx["cap_up"] = sig["cap_z"] - ctx["cap_z_turned"] >= LIFT_SEEN
            return True
        if name == "open_r":
            return (
                h.grip_measured() >= OPEN_WIDTH - 0.002
                and t_phase > RELEASE_OPEN_S
                and h.tcp_err(goal) < 0.004
            )
        if name == "unwind":
            return ctx["twist_cmd"] <= 0.0 and _wrist_err_deg(h, self._turned(ctx, 0.0)) < ORIENT_TOL_DEG
        if name == "lift":
            done = h.tcp_err(goal) < 0.004
            if done:
                cp, cq = _pq(h, "cap")
                ctx["cap_below"] = float(h.tcp_pos()[2] - cp[2])
                ctx["grasp_off"] = cap_centre(cp, cq)[:2] - h.tcp_pos()[:2]
            return done
        if name == "lower":
            # No time fallback: the cap is let go at this height, and a
            # release further up bounces it out of the tray.
            return h.tcp_err(goal) < 0.004
        if name in ("open", "open_l"):
            return max(sig["areas"][p] for p in SIDE_PADS[arm]) < 1.0 and t_phase > RELEASE_OPEN_S
        return False

    def next_phase(self, ctx, name):
        if name in ("close_l", "close_r"):
            side = self._ARM[name]
            if ctx["contact"][side] is None:
                if ctx["regrasps"] < MAX_REGRASPS:
                    ctx["regrasps"] += 1
                    ctx["held_at"][side] = None
                    ctx["grip"][side] = OPEN_WIDTH
                    return "above_bottle" if side == "left" else "above_cap"
                return "open_l" if side == "left" else "open"
        if name == "test_lift":
            if ctx["cap_up"]:
                return "lift"
            if ctx["strokes"] >= MAX_STROKES:
                return "open"
            return None  # open_r, then unwind
        if name == "unwind":
            # Grip again from the same place: a new contact is measured.
            ctx["contact"]["right"] = None
            ctx["held_at"]["right"] = None
            return "close_r"
        return None

    def handover_ok(self, h, ctx, sig) -> bool:
        """Policy control starts at t = 0 from HOME: nothing to check."""
        return True

    def should_terminate(self, ctx, hist) -> bool:
        """The cap has been in the tray for a second: the task is done."""
        from sim.harness import CTRL_FPS

        in_tray_hist = hist.get("in_tray", [])
        n = CTRL_FPS
        return len(in_tray_hist) >= n and all(v > 0.5 for v in in_tray_hist[-n:])

    # ---- scoring -----------------------------------------------------------
    def build_trace(self, h, ctx, hist) -> _Trace:
        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={"cap": _np.asarray(hist["cap_pos"]), "bottle": _np.asarray(hist["bottle_pos"])},
            object_quat={"cap": _np.asarray(hist["cap_quat"])},
            extra={
                "thread_deg": _np.asarray(hist["thread_deg"]),
                "bottle_deg": _np.asarray(hist["bottle_deg"]),
            },
            phase=_np.asarray(hist["phase_i"]),
            phase_names=tuple(n for n, _ in self.phases),
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(
            trace,
            tray_pos=(*ctx["layout"]["tray"], TABLE_TOP_Z),
            turner_pads=SIDE_PADS[TURNER],
            holder_pads=SIDE_PADS[HOLDER],
        )

    def episode_extra(self, ctx, hist) -> dict:
        return {
            "bottle_xy": [round(v, 4) for v in ctx["layout"]["bottle"]],
            "tray_xy": [round(v, 4) for v in ctx["layout"]["tray"]],
            "thread_angle_deg": ctx["thread_angle"],
            "strokes": ctx["strokes"],
            "freed": ctx["freed"],
            "regrasps": ctx["regrasps"],
            "contact_mm": {s: (None if c is None else round(c * 1e3, 2)) for s, c in ctx["contact"].items()},
        }

    def report_extra(self, knobs) -> dict:
        return {
            "thread_angle_deg": float(knobs["thread_angle_deg"]),
            "thread_angle_sweep_deg": list(THREAD_ANGLE_SWEEP),
            "grasp_pitch_deg": float(knobs["grasp_pitch_deg"]),
            "bottle_jitter_m": float(knobs["bottle_jitter"]),
            "tray_jitter_m": float(knobs["tray_jitter"]),
            "stroke_deg": STROKE_DEG,
        }


TASK = CapUnscrewTask()
