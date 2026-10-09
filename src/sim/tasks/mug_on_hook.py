"""mug_on_hook expert.

Phases::

    home -> above_mug -> orient -> descend -> close -> lift -> carry
        -> align -> thread -> seat -> settle -> release -> retreat -> park
    seat                     ends on a sustained gel shear contact signature
"""

from __future__ import annotations

import copy
import math
from collections import deque
from statistics import median

import numpy as _np
import torch as _torch

from sim.criteria.mug_on_hook import eye_margin as _eye_margin
from sim.criteria.mug_on_hook import verdict as _verdict
from sim.criteria.probes import axis_from_quat as _axis
from sim.criteria.probes import tilt_deg as _tilt
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.mug_on_hook import (
    EYE_FLOOR_Z,
    EYE_HEIGHT,
    EYE_X,
    MUG_OUTER_R,
    peg_axis_z,
    peg_span,
)
from sim.envs.bimanual_tactile import RIGHT_ARM_SENSORS as _GRASP_PADS
from sim.envs.bimanual_tactile import TABLE_TOP_Z, gel_gap
from sim.envs.mug_on_hook import (
    MUG_SPAWN_Z,
    MugOnHookSceneCfg,
    sample_layout,
)
from sim.tactile import apply_sdf_collision, apply_triangle_mesh_collision
from sim.tasks.scene_utils import oblique_grasp_quat
from sim.tasks.spec import Knob, RetryReset, Task

__all__ = [
    "TASK",
    "MugOnHookTask",
]

# ---- controller parameters --------------------------------------------------
DT = 1.0 / 60.0
HOVER = 0.100  # m, transit height of the TCP above the table
GRASP_Z = 0.040  # m above the mug base, below the handle
OPEN_WIDTH = 0.027  # m per jaw
CLOSE_WIDTH = 0.019  # m per jaw
ORIENT_TOL_DEG = 3.0
APPROACH_BACKOFF = 0.060  # m back along the blade to the pre-grasp point
RETREAT_BACKOFF = 0.090  # m back along the blade once the mug is hung
HOME_ABOVE_PREGRASP = 0.030
APPROACH_SPEED = 0.10  # m/s, the command point's speed into the grasp
LEVEL_MIN_GEL_GAP = 0.010  # m, jaw axis is meaningless while the gels overlap

HANG_ALONG = 0.053  # m from the rail face to the release target
APPROACH_GAP = 0.015  # m beyond the peg tip before threading
THREAD_SPEED = 0.035  # m/s
SEAT_SPEED = 0.025  # m/s
THREAD_TOL = 0.005  # m
SETTLE_STEPS = 30
MAX_SPAWN_TILT_DEG = 5.0
CONTACT_AREA = 5.0  # mm2, summed across the grasp pads

EYE_LOCAL = _np.array([MUG_OUTER_R + EYE_X / 2.0, 0.0, EYE_FLOOR_Z + EYE_HEIGHT / 2.0])


class _ShearContact:
    """Compare shear against a recent free-carry baseline, rejecting spikes."""

    def __init__(self, ratio: float = 1.25, hold_s: float = 0.15):
        if not math.isfinite(ratio) or ratio <= 1.0 or not math.isfinite(hold_s) or hold_s <= 0.0:
            raise ValueError("Contact needs a shear ratio above one and a positive hold time")
        self.threshold = ratio
        # Hysteresis keeps sustained contact from resetting on small oscillations.
        self.reset_threshold = 1.0 + 0.4 * (ratio - 1.0)
        self.hold_s = hold_s
        self.free = deque(maxlen=7)
        self.baseline = None
        self.elapsed = 0.0
        self.ratio = 1.0

    def observe_free(self, shear: float, valid: bool = True) -> None:
        if valid and math.isfinite(shear) and shear > 0.0:
            self.free.append(float(shear))
        else:
            self.free.clear()

    def start(self) -> None:
        if len(self.free) < 5:
            raise RuntimeError("mug_on_hook needs five valid gel shear samples before seating")
        self.baseline = median(self.free)

    def update(self, shear: float, dt: float, valid: bool = True, held: bool = True) -> bool:
        if self.baseline is None:
            raise RuntimeError("Gel shear baseline has not been established")
        if not valid or not held or not math.isfinite(shear):
            self.elapsed = 0.0
            self.ratio = 0.0
            return False
        self.ratio = float(shear) / self.baseline
        threshold = self.reset_threshold if self.elapsed > 0.0 else self.threshold
        self.elapsed = self.elapsed + dt if self.ratio >= threshold else 0.0
        return self.elapsed + 1e-9 >= self.hold_s


class MugOnHookTask(Task):
    name = "mug_on_hook"
    instruction = "hang the mug on the hook"
    report_name = "mug_on_hook_scripted.json"
    knobs = {
        "open_width": Knob(OPEN_WIDTH, "jaw command while approaching (m)"),
        "close_width": Knob(CLOSE_WIDTH, "jaw command while holding (m)"),
        "grasp_pitch_deg": Knob(45.0, "blade pitch below horizontal at the grasp, deg"),
        "grasp_bias_mm": Knob(6.0, "half-width of the hidden x offset of the mug in the jaws (mm)"),
        "align_err_x_mm": Knob(
            1.0, "half-width of the hidden sideways offset of the aim inside the eye (mm)"
        ),
        "align_err_z_mm": Knob(0.0, "optional extra height error of the aim inside the eye (mm)"),
        "eye_gap_mm": Knob(EYE_X * 1e3, "clear width of the handle's eye (mm); the generated mug must match"),
        "seat_shear_ratio": Knob(1.25, "gel shear / free-carry baseline required for seating"),
        "seat_drop_mm": Knob(26.0, "maximum descent while looking for support contact (mm)"),
    }
    phases = [
        ("home", 3.0),
        ("above_mug", 2.0),
        ("orient", 2.0),
        ("descend", 2.5),
        ("close", 1.0),
        ("lift", 1.5),
        ("carry", 2.5),
        ("align", 2.5),
        ("thread", 3.0),
        ("seat", 2.2),
        ("settle", 0.8),
        ("release", 0.6),
        ("retreat", 1.2),
        ("park", 3.0),
    ]
    handover_phase = None  # the policy drives from t = 0, the robot's HOME
    horizon_s = 60.0
    scene_cam = ((0.72, 0.10, 0.44), (0.265, -0.170, TABLE_TOP_Z + 0.07))

    # ---- scene -------------------------------------------------------------
    def make_scene_cfg(self, knobs):
        return MugOnHookSceneCfg(num_envs=1, env_spacing=3.0)

    def apply_collision(self, h, *, env_path="/World/envs/env_0") -> dict:
        meshes = {"mug": apply_sdf_collision(f"{env_path}/mug", sdf_resolution=128)}
        meshes["hook"] = apply_triangle_mesh_collision(f"{env_path}/hook")
        return meshes

    # ---- episode -----------------------------------------------------------

    settle_seconds = SETTLE_STEPS * DT

    def sample_init_state(self, robot, rng, state_id, knobs):
        layout = sample_layout(rng)
        yaw = _np.radians(layout["mug_yaw"])
        objects = {
            "hook": [*layout["rail"], layout["rail_z"], 1.0, 0.0, 0.0, 0.0],
            "mug": [*layout["mug"], MUG_SPAWN_Z, float(_np.cos(yaw / 2)), 0.0, 0.0, float(_np.sin(yaw / 2))],
        }
        variables = {"side": "right"}
        for name, knob in (
            ("bias", "grasp_bias_mm"),
            ("aim_x", "align_err_x_mm"),
            ("aim_z", "align_err_z_mm"),
        ):
            variables[name] = float(rng.uniform(-1, 1) * float(knobs[knob]) * 1e-3)
        return dict(id=state_id, robot=copy.deepcopy(robot), objects=objects, task_vars=variables)

    def validate_init_state(self, state):
        variables = state["task_vars"]
        if variables.get("side") != "right":
            raise ValueError("mug_on_hook requires side=right")
        offsets = _np.asarray([variables.get(key) for key in ("bias", "aim_x", "aim_z")], dtype=float)
        if not _np.isfinite(offsets).all():
            raise ValueError("Mug offsets must be finite")

    def begin_restored_episode(self, h, state, knobs):
        self.validate_init_state(state)
        h.select_arm("right")
        rail, _ = self._pose(h, "hook")
        variables = state["task_vars"]
        return self._new_ctx(h, rail[:3], variables["bias"], variables["aim_x"], variables["aim_z"], knobs)

    def init_state_objects(self):
        return ("hook", "mug")

    def snapshot_init_state(self, h, ctx):
        return {key: ctx[key] for key in ("side", "bias", "aim_x", "aim_z")}

    def restore_init_state(self, h, state, knobs):
        h.hold_default(SETTLE_STEPS)
        return self.begin_restored_episode(h, state, knobs)

    def reset(self, h, rng, ep, knobs) -> dict:
        h.select_arm("right")
        layout = sample_layout(rng)
        yaw = _np.radians(layout["mug_yaw"])
        h.place(h.scene["hook"], layout["rail"], layout["rail_z"])
        h.place(
            h.scene["mug"],
            layout["mug"],
            MUG_SPAWN_Z,
            quat=(float(_np.cos(yaw / 2)), 0.0, 0.0, float(_np.sin(yaw / 2))),
        )
        h.reset_robot()
        h.hold_default(SETTLE_STEPS)

        _, mug_quat = self._pose(h, "mug")
        if _tilt(mug_quat) > MAX_SPAWN_TILT_DEG:
            raise RetryReset("the mug did not settle upright on the table")

        bias = float(rng.uniform(-1.0, 1.0) * float(knobs["grasp_bias_mm"]) * 1e-3)
        aim_x = float(rng.uniform(-1.0, 1.0) * float(knobs["align_err_x_mm"]) * 1e-3)
        aim_z = float(rng.uniform(-1.0, 1.0) * float(knobs["align_err_z_mm"]) * 1e-3)
        return self._new_ctx(h, (*layout["rail"], layout["rail_z"]), bias, aim_x, aim_z, knobs)

    def _new_ctx(self, h, rail_xyz, bias, aim_x, aim_z, knobs):
        mug_pos, _ = self._pose(h, "mug")
        rail = _np.asarray(rail_xyz, dtype=float)
        span = peg_span()
        peg_root = _np.array([rail[0], rail[1], rail[2] + peg_axis_z(0.0)])
        peg_tip = _np.array([rail[0], rail[1] + span, rail[2] + peg_axis_z(span)])
        mid = 0.5 * (HANG_ALONG + span)

        return {
            "side": "right",
            "rail_z_delta": float(rail[2] - TABLE_TOP_Z),
            "mug0": mug_pos,
            "bias": bias,
            "aim_x": aim_x,
            "aim_z": aim_z,
            "peg_root": peg_root,
            "peg_tip": peg_tip,
            # Search for the unknown fixture height through tactile seating.
            "carry_eye_z": TABLE_TOP_Z + peg_axis_z(mid),
            "hang_y": float(rail[1] + HANG_ALONG),
            "stage_y": float(peg_tip[1] + APPROACH_GAP),
            "level_quat": None,
            "approach_cmd": None,
            "seat_ref": None,
            "seat_cmd": None,
            "seat_tcp": None,
            "seat_travel": 0.0,
            "thread_cmd": None,
            "threaded": False,
            "thread_target_reached": False,
            "seated": False,
            "held_area": [],
            "feedback": _ShearContact(ratio=float(knobs["seat_shear_ratio"])),
        }

    @staticmethod
    def _pose(h, name):
        d = h.scene[name].data
        return (d.root_pos_w[0].detach().cpu().numpy().copy(), d.root_quat_w[0].detach().cpu().numpy().copy())

    @staticmethod
    def _tactile_load(h) -> dict:
        """Sum gel shear magnitudes across the grasp pads, scaled by 1000."""
        shear = 0.0
        valid = True
        for pad in _GRASP_PADS:
            data = h.sensors[pad].data
            # Reject incomplete or non-finite sensor samples.
            nf = getattr(data, "tactile_normal_force", None)
            sh = getattr(data, "tactile_shear_force", None)
            qw = getattr(data, "tactile_points_quat_w", None)
            if any(v is None for v in (nf, sh, qw)):
                valid = False
                continue
            if not bool(_torch.isfinite(nf).all() & _torch.isfinite(sh).all() & _torch.isfinite(qw).all()):
                valid = False
                continue
            shear += float(_torch.linalg.norm(sh[0], dim=-1).sum()) * 1e3
        return {"tactile_valid": valid, "tactile_shear": shear}

    def signals(self, h, ctx) -> dict:
        pos, quat = self._pose(h, "mug")
        tcp = h.tcp_pos()
        sig = {
            "areas": h.pad_areas(),
            **self._tactile_load(h),
            "mug_pos": pos,
            "mug_quat": quat,
            # Measure slip relative to the moving gripper.
            "grasp_frame_pos": tcp.copy(),
            "mug_y": float(pos[1]),
            "mug_z": float(pos[2]),
            "tilt": _tilt(quat),
        }
        sig["tactile_seat_ratio"] = ctx["feedback"].ratio
        sig["eye_err_mm"] = self._eye_err(ctx, pos, quat) * 1e3
        sig["eye_margin_mm"] = _eye_margin(pos, quat, ctx["peg_root"], ctx["peg_tip"]) * 1e3
        sig["mug_x"] = float(pos[0])
        return sig

    @staticmethod
    def _eye_err(ctx, mug_pos, mug_quat) -> float:
        """True eye-to-peg offset along x, in metres; used for diagnostics."""
        eye = _np.asarray(mug_pos, dtype=float) + _axis(mug_quat, EYE_LOCAL)
        return float(eye[0] - ctx["peg_root"][0])

    # ---- targets -----------------------------------------------------------
    @staticmethod
    def _blade(knobs) -> _np.ndarray:
        th = _np.radians(float(knobs["grasp_pitch_deg"]))
        return _np.array([_np.cos(th), 0.0, -_np.sin(th)])

    @staticmethod
    def _wrist_err_deg(h, ctx) -> float:
        if ctx["level_quat"] is None:
            return 180.0
        d = abs(float(_np.dot(h.tcp_quat(), _np.asarray(ctx["level_quat"], dtype=float))))
        return float(_np.degrees(2.0 * _np.arccos(min(1.0, d))))

    @staticmethod
    def _tcp_for_eye(eye_xyz) -> list:
        """TCP target assuming the mug is centred in the jaws at GRASP_Z."""
        e = _np.asarray(eye_xyz, dtype=float)
        return [e[0] - EYE_LOCAL[0], e[1] - EYE_LOCAL[1], e[2] - EYE_LOCAL[2] + GRASP_Z]

    def _stage_eye(self, ctx) -> list:
        """Nominal eye target with the episode's hidden alignment offsets."""
        return [ctx["peg_root"][0] + ctx["aim_x"], ctx["stage_y"], ctx["carry_eye_z"] + ctx["aim_z"]]

    @staticmethod
    def _ramp(h, ctx, key: str, goal, speed: float) -> list:
        """Advance the command at a fixed speed, independent of the TCP's IK lag."""
        goal = _np.asarray(goal, dtype=float)
        cur = ctx.get(key)
        if cur is None:
            cur = h.tcp_pos().astype(float)
        d = goal - cur
        n = float(_np.linalg.norm(d))
        step = speed * h.control_dt
        ctx[key] = goal if n <= step else cur + d * (step / n)
        return ctx[key].tolist()

    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        ow, cw = float(knobs["open_width"]), float(knobs["close_width"])
        mug0 = ctx["mug0"]
        grasp = _np.array([mug0[0] + ctx["bias"], mug0[1], mug0[2] + GRASP_Z])
        pregrasp = grasp - self._blade(knobs) * APPROACH_BACKOFF

        if name == "home":
            return [pregrasp[0], pregrasp[1], pregrasp[2] + HOME_ABOVE_PREGRASP], 0.0, None
        if name == "above_mug":
            return pregrasp.tolist(), ow, None
        if ctx["level_quat"] is None and gel_gap(h.grip_measured()) >= LEVEL_MIN_GEL_GAP:
            ctx["level_quat"] = oblique_grasp_quat(
                h.tcp_quat(), h.jaw_axis(), h.approach_axis(), float(knobs["grasp_pitch_deg"]), 0.0
            )
        q = ctx["level_quat"]
        if name == "orient":
            return pregrasp.tolist(), ow, q
        if name == "descend":
            ctx["approach_goal"] = grasp.tolist()
            return self._ramp(h, ctx, "approach_cmd", grasp, APPROACH_SPEED), ow, q
        if name == "close":
            return grasp.tolist(), cw, q
        if name == "lift":
            return [grasp[0], grasp[1], TABLE_TOP_Z + HOVER], cw, q
        if name == "carry":
            stage = self._stage_eye(ctx)
            tcp = self._tcp_for_eye(stage)
            return [tcp[0], tcp[1], TABLE_TOP_Z + HOVER], cw, q
        if name == "align":
            return self._tcp_for_eye(self._stage_eye(ctx)), cw, q
        if name == "thread":
            eye = self._stage_eye(ctx)
            eye[1] = ctx["hang_y"]
            goal = self._tcp_for_eye(eye)
            return self._ramp(h, ctx, "thread_cmd", goal, THREAD_SPEED), cw, q
        if name == "seat":
            if ctx["seat_ref"] is None:
                ctx["seat_ref"] = float(h.tcp_pos()[2])
                ctx["feedback"].start()
            eye = self._stage_eye(ctx)
            eye[1] = ctx["hang_y"]
            eye[2] -= float(knobs["seat_drop_mm"]) * 1e-3
            return self._ramp(h, ctx, "seat_cmd", self._tcp_for_eye(eye), SEAT_SPEED), cw, q
        if name in ("settle", "release"):
            if ctx["seat_tcp"] is None:
                ctx["seat_tcp"] = h.tcp_pos().astype(float)
                ctx["seat_travel"] = float(ctx["seat_ref"] - ctx["seat_tcp"][2])
            return ctx["seat_tcp"].tolist(), (ow if name == "release" and ctx["seated"] else cw), q
        base = ctx["seat_tcp"] if ctx["seat_tcp"] is not None else h.tcp_pos().astype(float)
        # Keep holding after an unconfirmed seat, including phase timeouts.
        if not ctx["seated"]:
            return base.tolist(), cw, q
        # Withdraw along the blade to clear the hanging mug.
        tcp = base - self._blade(knobs) * RETREAT_BACKOFF
        return tcp.tolist(), ow, q

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        if name in ("lift", "carry"):
            ctx["held_area"].append(sum(sig["areas"].values()))
        if name == "park":
            # Let the released mug settle against the stop ball before scoring.
            return t_phase > 2.4
        if name in ("home", "above_mug", "lift", "carry", "retreat"):
            return h.tcp_err(target) < 0.010
        if name == "align":
            # The stop ball leaves 2.5 mm of sideways clearance.
            return h.tcp_err(target) < 0.0015
        if name == "orient":
            return (
                t_phase > 0.3 and self._wrist_err_deg(h, ctx) < ORIENT_TOL_DEG and h.tcp_err(target) < 0.015
            )
        if name == "descend":
            return h.tcp_err(ctx["approach_goal"]) < 0.008
        if name == "close":
            return t_phase > 0.4 and sum(sig["areas"].values()) > CONTACT_AREA
        if name == "thread":
            ctx["feedback"].observe_free(sig["tactile_shear"], sig["tactile_valid"])
            eye = self._stage_eye(ctx)
            eye[1] = ctx["hang_y"]
            reached = h.tcp_err(self._tcp_for_eye(eye)) < THREAD_TOL
            ctx["thread_target_reached"] = reached
            if reached:
                ctx["threaded"] = sig["eye_margin_mm"] > 0.0  # diagnostic only
            return reached
        if name == "seat":
            ctx["seated"] = ctx["feedback"].update(
                sig["tactile_shear"],
                h.control_dt,
                valid=sig["tactile_valid"],
                held=sum(sig["areas"][p] for p in _GRASP_PADS) > CONTACT_AREA,
            )
            return ctx["seated"]
        if name == "settle":
            return t_phase > 0.5
        if name == "release":
            return t_phase > 0.4
        return False

    # ---- scoring -----------------------------------------------------------
    def build_trace(self, h, ctx, hist) -> _Trace:
        from sim.envs.bimanual_tactile import TACTILE_SENSORS

        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={"mug": _np.asarray(hist["mug_pos"])},
            object_quat={"mug": _np.asarray(hist["mug_quat"])},
            extra={"grasp_frame_pos": _np.asarray(hist["grasp_frame_pos"])},
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(
            trace,
            peg_root=ctx["peg_root"],
            peg_tip=ctx["peg_tip"],
            pads=_GRASP_PADS,
            eye_x=float(knobs["eye_gap_mm"]) * 1e-3,
        )

    def episode_extra(self, ctx, hist) -> dict:
        ha = ctx["held_area"]
        final = (
            _eye_margin(hist["mug_pos"][-1], hist["mug_quat"][-1], ctx["peg_root"], ctx["peg_tip"])
            if hist.get("mug_pos")
            else float("nan")
        )
        return {
            "grasp_bias_mm": round(ctx["bias"] * 1e3, 2),
            "aim_x_mm": round(ctx["aim_x"] * 1e3, 2),
            "aim_z_mm": round(ctx["aim_z"] * 1e3, 2),
            "rail_z_delta_mm": round(ctx["rail_z_delta"] * 1e3, 2),
            "seat_travel_mm": round(ctx["seat_travel"] * 1e3, 2),
            "threaded": bool(ctx["threaded"]),
            "thread_target_reached": bool(ctx["thread_target_reached"]),
            "seated": bool(ctx["seated"]),
            "seat_shear_baseline": ctx["feedback"].baseline,
            "seat_shear_ratio": ctx["feedback"].ratio,
            "seat_shear_on_ratio": ctx["feedback"].threshold,
            "seat_shear_off_ratio": ctx["feedback"].reset_threshold,
            "seat_contact_hold_s": ctx["feedback"].elapsed,
            "eye_margin_mm": round(float(final) * 1e3, 2),
            "held_area_median": float(_np.median(ha)) if ha else 0.0,
        }


TASK = MugOnHookTask()
