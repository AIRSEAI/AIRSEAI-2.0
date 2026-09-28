"""mug_on_hook expert.

Phases::

    home -> above_mug -> orient -> descend -> close -> lift -> carry
        -> align -> thread -> seat -> settle -> release -> retreat -> park
    seat                     ends where the load transfers, not at a set depth
"""

from __future__ import annotations

import copy

import numpy as _np

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
    HANG_ALONG,
    MUG_SPAWN_Z,
    MugOnHookSceneCfg,
    sample_layout,
)
from sim.tactile import apply_sdf_collision, apply_triangle_mesh_collision
from sim.tasks.scene_utils import oblique_grasp_quat
from sim.tasks.spec import Knob, RetryReset, Task

# ---- controller parameters --------------------------------------------------
DT = 1.0 / 60.0
HOVER = 0.100  # m, transit height of the TCP above the table
# TCP height above the mug's base at the grasp, m. On the body, below the
# handle's lower arm, and high enough that the finger housing clears the
# table on a blade pitched 45 degrees.
GRASP_Z = 0.040
# Jaw commands, m per jaw: open clears the 38 mm body, close squeezes about
# 1.5 mm into each gel.
OPEN_WIDTH = 0.027
CLOSE_WIDTH = 0.019
ORIENT_TOL_DEG = 3.0
APPROACH_BACKOFF = 0.060  # m back along the blade to the pre-grasp point
RETREAT_BACKOFF = 0.090  # m back along the blade once the mug is hung
HOME_ABOVE_PREGRASP = 0.030
APPROACH_SPEED = 0.10  # m/s, the command point's speed into the grasp
LEVEL_MIN_GEL_GAP = 0.010  # m, jaw axis is meaningless while the gels overlap

# Where the eye waits before threading: this far out past the peg's end, m.
APPROACH_GAP = 0.015
# Speeds for the two phases that work against contact, m/s. Slow, so the
# contact that stops the mug registers before the arm has pushed through it.
THREAD_SPEED = 0.035
SEAT_SPEED = 0.025
# How far the mug's y must come to the hanging point to count as threaded, m.
THREAD_TOL = 0.005
# How much of the hand's descent the mug must fail to follow before the load
# counts as transferred, m. Small: this ends the descent, and the hand should
# stop where the weight goes rather than press past it.
UNLOAD_MIN = 0.002
# Vertical lift before release, metres; drop_mm adds to this offset.
SETTLE_LIFT = 0.0
# Mug tilt at reset above which the draw is thrown away, degrees.
MAX_SPAWN_TILT_DEG = 5.0

# The eye's centre in the mug's own frame, m: what the controller aims with.
EYE_LOCAL = _np.array([MUG_OUTER_R + EYE_X / 2.0, 0.0, EYE_FLOOR_Z + EYE_HEIGHT / 2.0])


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
        "seat_drop_mm": Knob(26.0, "how far the hand may descend looking for the transfer (mm)"),
        "drop_mm": Knob(
            0.0, "let go this far above where the load transferred (mm); 0 lets go against the peg"
        ),
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

    settle_seconds = 30 * DT

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
        h.hold_default(30)
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
        h.hold_default(30)

        mug_pos, mug_quat = self._pose(h, "mug")
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
        # The height the eye is carried at: the peg's, midway along the travel,
        # so the tilt costs at most half of it either end.
        mid = 0.5 * (HANG_ALONG + span)

        return {
            "side": "right",
            "rail_z_delta": float(rail[2] - TABLE_TOP_Z),
            "mug0": mug_pos,
            "bias": bias,
            "drop": float(knobs["drop_mm"]) * 1e-3,
            "aim_x": aim_x,
            "aim_z": aim_z,
            "peg_root": peg_root,
            "peg_tip": peg_tip,
            "carry_eye_z": TABLE_TOP_Z + peg_axis_z(mid),
            "hang_y": float(rail[1] + HANG_ALONG),
            "stage_y": float(peg_tip[1] + APPROACH_GAP),
            "eye_err": 0.0,
            "level_quat": None,
            "approach_cmd": None,
            "seat_ref": None,
            "seat_cmd": None,
            "seat_tcp": None,
            "unload": 0.0,
            "seat_travel": 0.0,
            "thread_ref": None,
            "thread_cmd": None,
            "thread_travel": 0.0,
            "threaded": False,
            "seated": False,
            "held_area": [],
        }

    @staticmethod
    def _pose(h, name):
        d = h.scene[name].data
        return (d.root_pos_w[0].detach().cpu().numpy().copy(), d.root_quat_w[0].detach().cpu().numpy().copy())

    def signals(self, h, ctx) -> dict:
        pos, quat = self._pose(h, "mug")
        tcp = h.tcp_pos()
        sig = {
            "areas": h.pad_areas(),
            "mug_pos": pos,
            "mug_quat": quat,
            # The grasp frame `detect_slip` measures against: without it the
            # detector falls back to the mug's first held position, and the
            # whole carry reads as slip.
            "grasp_frame_pos": tcp.copy(),
            "mug_y": float(pos[1]),
            "mug_z": float(pos[2]),
            "tilt": _tilt(quat),
        }
        # How far the hand has gone down without the mug: the expert's stand-in
        # for the gel losing the mug's weight.
        ref = ctx.get("seat_ref")
        if ref is not None:
            ctx["unload"] = float((ref[0] - tcp[2]) - (ref[1] - pos[2]))
        sig["unload_mm"] = ctx["unload"] * 1e3
        # Signed distance from the peg's axis to where the eye really is, for
        # the search direction and for the trace. The controller uses only its
        # sign, which the gel's shear carries.
        ctx["eye_err"] = self._eye_err(ctx, pos, quat)
        sig["eye_err_mm"] = ctx["eye_err"] * 1e3
        # The verdict's own probe, live: positive means the peg is inside the
        # eye right now. Trace-only, so a blocked thread can be told apart
        # from a mis-aimed one without guessing from the scalars.
        sig["eye_margin_mm"] = _eye_margin(pos, quat, ctx["peg_root"], ctx["peg_tip"]) * 1e3
        sig["mug_x"] = float(pos[0])
        return sig

    @staticmethod
    def _eye_err(ctx, mug_pos, mug_quat) -> float:
        """True eye centre minus the peg's axis, in x: the error the hidden
        grasp offset produced, metres."""
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
        """TCP that the controller believes puts the eye's centre at `eye_xyz`.

        It assumes the mug hangs centred in the jaws with its base GRASP_Z
        below the gels -- which is the assumption the hidden offset breaks.
        """
        e = _np.asarray(eye_xyz, dtype=float)
        return [e[0] - EYE_LOCAL[0], e[1] - EYE_LOCAL[1], e[2] - EYE_LOCAL[2] + GRASP_Z]

    def _stage_eye(self, ctx) -> list:
        """Where the eye goes this episode, in world metres.

        The controller aims at the peg's axis; the episode's hidden offset is
        added here and nowhere else, so the rod ends up somewhere inside the
        eye rather than at its middle.
        """
        return [ctx["peg_root"][0] + ctx["aim_x"], ctx["stage_y"], ctx["carry_eye_z"] + ctx["aim_z"]]

    @staticmethod
    def _ramp(h, ctx, key: str, goal, speed: float) -> list:
        """A command point that walks to `goal` at `speed`, m/s.

        The differential IK moves a fraction of whatever error it is handed,
        so neither the far goal nor a point capped just ahead of the TCP gives
        a speed: the first puts its largest step right at the contact, the
        second creeps. Walking the goal does.
        """
        goal = _np.asarray(goal, dtype=float)
        cur = ctx.get(key)
        if cur is None:
            cur = h.tcp_pos().astype(float)
        d = goal - cur
        n = float(_np.linalg.norm(d))
        step = speed * h.control_dt
        ctx[key] = goal if n <= step else cur + d * (step / n)
        return ctx[key].tolist() if hasattr(ctx[key], "tolist") else list(ctx[key])

    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        ow, cw = float(knobs["open_width"]), float(knobs["close_width"])
        mug0 = ctx["mug0"]
        grasp = _np.array([mug0[0] + ctx["bias"], mug0[1], mug0[2] + GRASP_Z])
        pregrasp = grasp - self._blade(knobs) * APPROACH_BACKOFF

        if name == "home":
            return [pregrasp[0], pregrasp[1], pregrasp[2] + HOME_ABOVE_PREGRASP], 0.0, None
        if name == "above_mug":
            return pregrasp.tolist(), ow, None
        # Approach on position IK, then turn the wrist in place and keep that
        # orientation for the rest of the episode.
        if ctx["level_quat"] is None and gel_gap(h.grip_measured()) >= LEVEL_MIN_GEL_GAP:
            ctx["level_quat"] = oblique_grasp_quat(
                h.tcp_quat(), h.jaw_axis(), h.approach_axis(), float(knobs["grasp_pitch_deg"]), 0.0
            )
        q = ctx["level_quat"]
        if name == "orient":
            return pregrasp.tolist(), ow, q
        if name == "descend":
            if ctx["approach_cmd"] is None:
                ctx["approach_cmd"] = h.tcp_pos().astype(float)
            d = grasp - ctx["approach_cmd"]
            n = float(_np.linalg.norm(d))
            step = APPROACH_SPEED * h.control_dt
            ctx["approach_cmd"] = ctx["approach_cmd"] + d * (step / n) if n > step else grasp
            ctx["approach_goal"] = grasp.tolist()
            return ctx["approach_cmd"].tolist(), ow, q
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
            if ctx["thread_ref"] is None:
                ctx["thread_ref"] = float(sig["mug_y"])
                ctx["thread_travel"] = abs(ctx["stage_y"] - ctx["hang_y"])
            return self._ramp(h, ctx, "thread_cmd", goal, THREAD_SPEED), cw, q
        if name == "seat":
            # Walk down until the gels say the weight has gone. `seat_drop_mm`
            # is how far the hand may look, not how far it goes.
            if ctx["seat_ref"] is None:
                ctx["seat_ref"] = (float(h.tcp_pos()[2]), float(sig["mug_z"]))
            eye = self._stage_eye(ctx)
            eye[1] = ctx["hang_y"]
            eye[2] -= float(knobs["seat_drop_mm"]) * 1e-3
            return self._ramp(h, ctx, "seat_cmd", self._tcp_for_eye(eye), SEAT_SPEED), cw, q
        if name in ("settle", "release"):
            # Both work from where the descent stopped, the only pose that
            # knows where inside the eye the rod turned out to be.
            if ctx["seat_tcp"] is None:
                ctx["seat_tcp"] = h.tcp_pos().astype(float)
                ctx["seat_travel"] = float(ctx["seat_ref"][0] - ctx["seat_tcp"][2])
            # `drop_mm` lifts further before opening, so the mug falls that far
            # onto the rod: the ablation for a controller that cannot tell
            # where the rod is and lets go short of it.
            lift = SETTLE_LIFT + float(knobs["drop_mm"]) * 1e-3
            tcp = ctx["seat_tcp"] + _np.array([0.0, 0.0, lift])
            return tcp.tolist(), (ow if name == "release" else cw), q
        # retreat and park: back out along the blade, the way the hand came
        # in. Withdrawing along the peg instead runs a finger the length of
        # the rod and takes the hung mug with it.
        base = ctx["seat_tcp"] if ctx["seat_tcp"] is not None else h.tcp_pos().astype(float)
        tcp = base + _np.array([0.0, 0.0, SETTLE_LIFT]) - self._blade(knobs) * RETREAT_BACKOFF
        return tcp.tolist(), ow, q

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        if name in ("lift", "carry"):
            ctx["held_area"].append(sum(sig["areas"].values()))
        if name == "park":
            # A released mug swings, then runs down the peg to the stop ball,
            # and the verdict reads the last sample: the episode must not end
            # before it has come to rest.
            return t_phase > 2.4
        if name in ("home", "above_mug", "lift", "carry", "retreat"):
            return h.tcp_err(target) < 0.010
        if name == "align":
            # The stop ball leaves only 2.5 mm sideways clearance.
            return h.tcp_err(target) < 0.0015
        if name == "orient":
            return (
                t_phase > 0.3 and self._wrist_err_deg(h, ctx) < ORIENT_TOL_DEG and h.tcp_err(target) < 0.015
            )
        if name == "descend":
            return h.tcp_err(ctx["approach_goal"]) < 0.008
        if name == "close":
            return t_phase > 0.4 and sum(sig["areas"].values()) > 5.0
        if name == "thread":
            # Done when the rod is through the eye and the mug has come as far
            # as it was sent. The first is the state that matters; the second
            # stops the phase ending the moment the rod first enters.
            ctx["threaded"] = sig["eye_margin_mm"] > 0.0
            gone = ctx["thread_ref"] - sig["mug_y"]
            return ctx["threaded"] and gone >= ctx["thread_travel"] - THREAD_TOL
        if name == "seat":
            ctx["seated"] = ctx["unload"] >= UNLOAD_MIN
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
            # What the episode is really about: how far the hand came down
            # before the weight went onto the peg.
            "seat_travel_mm": round(ctx["seat_travel"] * 1e3, 2),
            "threaded": bool(ctx["threaded"]),
            "drop_mm": round(float(ctx["drop"]) * 1e3, 2),
            "seated": bool(ctx["seated"]),
            "unload_mm": round(ctx["unload"] * 1e3, 2),
            "eye_margin_mm": round(float(final) * 1e3, 2),
            "held_area_median": float(_np.median(ha)) if ha else 0.0,
        }


TASK = MugOnHookTask()
