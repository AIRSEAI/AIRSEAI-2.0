"""texture_sort expert.

Phases::

    home -> above_target -> orient -> descend -> close -> classify -> lift
        -> above_bin -> lower -> open -> retreat
"""

from __future__ import annotations

import copy

import numpy as _np
import torch as _torch

from sim.criteria.texture_sort import verdict as _verdict
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.texture_sort import (
    OBJECT_SPECS,
)
from sim.envs.bimanual_tactile import TABLE_TOP_Z, gel_gap
from sim.envs.texture_sort import (
    BIN_WALL_HEIGHT,
    DEFAULT_LIGHT_SCALE,
    FIXTURES,
    OBJECT_SPAWN_Z,
    PARKED_POS,
    START_JAW_WIDTH,
    TextureSortSceneCfg,
    sample_layout,
)
from sim.tactile import apply_sdf_collision, apply_triangle_mesh_collision
from sim.tasks.scene_utils import oblique_grasp_quat, retarget_pads
from sim.tasks.spec import Task

# Controller parameters.
# Jaw command while approaching and releasing, metres per jaw: a 46 mm gel gap around the 38 mm puck.
OPEN_WIDTH = 0.026


__all__ = [
    "TextureSortTask",
    "TASK",
]

DT = 1.0 / 60.0

# --- heights ---------------------------------------------------------------
# Gel-centre height for the grasp (the TCP origin is the gel midpoint). The
# sensor housing extends 34 mm below it, so 45 mm above the table leaves the
# housing 11 mm clear while the gel still bears on the puck's side.
GRASP_TCP_Z = TABLE_TOP_Z + 0.045
# m above the grasp height for free transport. The reachable band narrows as
# it rises, and higher hovers leave pose IK settling short in xy.
HOVER_TCP_Z = GRASP_TCP_Z + 0.060
# m, gel centre above the table when releasing: the puck's base hangs 45 mm
# below it and must clear the cup rim.
BIN_DROP_Z = TABLE_TOP_Z + BIN_WALL_HEIGHT + 0.070
# m back along the blade from the grasp point to the pre-grasp point: the hand
# lines up there and then moves in along its own blade.
APPROACH_BACKOFF = 0.060
APPROACH_JITTER = 0.030  # m, xy disc radius; z spread is one third
DESCEND_DETOUR = 0.020  # m, xy disc radius
GATE_BACKOFF = 0.040  # m along the entry direction
ENTRY_PAUSE = 0.012  # m before the grasp
ENTRY_RESIDUAL = 0.006  # m lateral offset at the pause
PRELOAD = 0.008  # m, summed pad penetration
ROUGH_THRESH_MM = 0.035
GRASP_PITCH_DEG = 45.0
GRASP_YAW_DEG = 0.0
SDF_RESOLUTION = 96
# m above the pre-grasp point where `home` ends, so the wrist turns clear of the puck.
HOME_ABOVE_PREGRASP = 0.030

# --- closing ramp ----------------------------------------------------------
# Fast across the free travel, slow once a pad feels anything, so the preload
# lands where it was aimed rather than wherever the drive lag left it.
CLOSE_RATE_FAST = 0.006  # m/s per jaw
CLOSE_RATE_SLOW = 0.0025  # m/s per jaw, after first touch
FIRST_TOUCH_M = 0.00005  # m of penetration that switches to the slow rate
# Closing floor, metres per jaw: a 30 mm commanded gel gap.
GRIP_MIN_WIDTH = 0.018
AREA_CONTACT = 5.0  # mm2 summed; below this `classify` has no contact

# --- classification --------------------------------------------------------
# Ripple filter: subtract a local average and take the RMS of the remainder.
# The kernel is ~2.6 mm across, between the dimple period and the puck's own
# curvature, so a smooth puck leaves almost nothing and a dimpled one leaves
# the dimples.
RIPPLE_KERNEL = 31
RIPPLE_CONTACT_M = 5e-5  # m; below this peak the pad is not in contact
RIPPLE_PATCH_TAXELS = 200  # minimum loaded taxels for a usable sample

# --- transport -------------------------------------------------------------
# Commanded TCP speed limits, m/s. Differential IK turns the target into a
# joint delta for the next tick, so an unbounded command makes the largest
# displacement the first one. The command point advances from the previous
# command, so the position term keeps pulling while pose IK turns the wrist.
FREE_SPEED = 0.30
APPROACH_SPEED = 0.10  # along the blade into the grasp
CARRY_SPEED = 0.18
CARRY_PHASES = ("close", "classify", "lift", "above_bin", "lower")

# Integral correction on the transport target. Pose IK settles short in xy
# over the cups -- a steady-state offset, not a rate limit -- and a puck
# released off a cup's centre lands on the rim. Over the cups the commanded
# xy therefore carries the accumulated miss. The phase's completion test
# still reads the TRUE error, never the biased one.
XY_BIAS_GAIN = 0.05  # per step at 60 Hz
XY_BIAS_MAX = 0.030  # m
BIAS_PHASES = ("above_bin", "lower")

# --- phase completion ------------------------------------------------------
ORIENT_TOL_DEG = 3.0  # wrist within this of its grasp orientation
# Measured gel gap, m, before the level orientation is computed. The jaw axis
# is the difference of the two gel centroids, which has no reliable sign while
# the shut gels overlap.
LEVEL_MIN_GEL_GAP = 0.010
SETTLE_STEPS = 40  # steps held at HOME, jaws shut, before t = 0


class TextureSortTask(Task):
    name = "texture_sort"
    # Names neither class: the instruction must not give away what the policy
    # is supposed to feel for.
    instruction = "put the block in the matching cup"
    report_name = "texture_sort_scripted.json"
    knobs = {}
    phases = [
        ("home", 2.0),
        ("above_target", 2.5),
        ("orient", 3.0),
        ("descend", 7.0),
        ("close", 4.0),
        ("classify", 1.2),
        ("lift", 3.0),
        ("above_bin", 5.5),
        ("lower", 4.0),
        ("open", 0.8),
        ("retreat", 1.5),
    ]
    # The policy takes control at t = 0: no prologue, the whole episode is
    # recorded, matching a rollout from the robot's own HOME.
    handover_phase = None
    horizon_s = 30.0
    scene_cam = ((1.25, -1.00, 0.80), (0.15, -0.08, 0.30))

    def _timeout(self, name: str) -> float:
        return dict(self.phases)[name]

    # ---- scene -------------------------------------------------------------
    def make_scene_cfg(self, knobs: dict):
        cfg = TextureSortSceneCfg(num_envs=1, env_spacing=3.0)
        cfg.set_light_scale(DEFAULT_LIGHT_SCALE)
        return cfg

    def apply_collision(self, h) -> None:
        for spec in OBJECT_SPECS.values():
            apply_sdf_collision(f"/World/envs/env_0/{spec.name}", sdf_resolution=SDF_RESOLUTION)
        for static in FIXTURES:
            # Triangle meshes, not convex hulls: a convex hull of the cup
            # fills it in and the puck lands on top.
            apply_triangle_mesh_collision(f"/World/envs/env_0/{static}")

    # ---- episode -----------------------------------------------------------
    def reset(self, h, rng, ep, knobs) -> dict:
        h.select_arm("right")
        # The hidden draw. The expert never reads `target_key` again; the bin
        # it flies to comes from the tactile reading in `classify`.
        target_key = "smooth" if rng.random() < 0.5 else "rough"
        spec = OBJECT_SPECS[target_key]
        layout = sample_layout(rng)

        for n in FIXTURES:
            h.place(h.scene[n], layout[n], TABLE_TOP_Z)
        for key, s in OBJECT_SPECS.items():
            if key == target_key:
                h.place(h.scene[s.name], layout["object"], OBJECT_SPAWN_Z)
            else:
                h.place(h.scene[s.name], PARKED_POS[:2], PARKED_POS[2])
        retarget_pads(h.sensors, spec.name)
        h.reset_robot()
        h.reset_ik()
        # Hold HOME exactly, jaws shut, while the puck settles: the first
        # recorded frame is the robot's HOME.
        h.hold_default(SETTLE_STEPS)

        # Aim at where the puck IS, read after the settle: only the class is
        # hidden, not the pose.
        seat = h.scene[spec.name].data.root_pos_w[0, :2].detach().cpu().numpy()
        return self._new_ctx(target_key, layout, seat)

    def prepare_expert(self, ctx: dict, seed: int, episode: int) -> dict:
        """Draw approach offsets independently of scene randomisation."""
        # The final component separates the expert stream from the scene stream.
        entropy = [seed, episode, 1]
        rng = _np.random.default_rng(entropy)

        def disc(radius):
            angle = rng.uniform(0.0, 2.0 * _np.pi)
            return radius * _np.sqrt(rng.uniform(0.0, 1.0)) * _np.array([_np.cos(angle), _np.sin(angle)])

        offset = _np.zeros(3)
        offset[:2] = disc(APPROACH_JITTER)
        offset[2] = rng.uniform(-APPROACH_JITTER / 3.0, APPROACH_JITTER / 3.0)
        ctx["approach_offset"] = offset
        ctx["descend_detour"] = disc(DESCEND_DETOUR)
        ctx["gate_stage"] = 0
        return {
            "rng_entropy": entropy,
            "bit_generator": type(rng.bit_generator).__name__,
            "approach_offset_mm": (offset * 1e3).tolist(),
            "descend_detour_mm": (ctx["descend_detour"] * 1e3).tolist(),
        }

    def _new_ctx(self, target_key, layout, seat) -> dict:
        """The episode's working state, however the episode was started."""
        spec = OBJECT_SPECS[target_key]
        return {
            "approach_offset": _np.zeros(3),
            "descend_detour": _np.zeros(2),
            "gate_stage": 0,
            "side": "right",
            "target_key": target_key,
            "correct_bin": spec.bin_index,
            "obj_prim": spec.name,
            "layout": layout,
            "seat": _np.asarray(seat, dtype=float),
            "bins": [layout["bin_0"], layout["bin_1"]],
            "level_quat": None,
            "grip_hold": None,
            "grip": START_JAW_WIDTH["right"],
            "xy_bias": _np.zeros(2),
            "goal": None,
            "cmd_pos": None,
            "release_xy": None,
            "ripples": [],
            "ripple_mm": 0.0,
            "chosen_bin": 0,
            "blade_tilt_deg": float("nan"),
            "drop_offset_mm": float("nan"),
        }

    # ---- frozen initial states ---------------------------------------------

    restore_contact_clearance_m = 1.0
    settle_seconds = SETTLE_STEPS * DT

    def sample_init_state(self, robot, rng, state_id, knobs):
        target_key = "smooth" if rng.random() < 0.5 else "rough"
        return self.layout_init_state(robot, sample_layout(rng), target_key, state_id)

    def layout_init_state(self, robot, layout, target_key, state_id):
        """Build a start using the authored smooth or rough asset."""
        if target_key not in OBJECT_SPECS:
            raise ValueError(f"Unknown target class: {target_key}")
        objects = {n: [*layout[n], TABLE_TOP_Z, 1.0, 0.0, 0.0, 0.0] for n in FIXTURES}
        for key, spec in OBJECT_SPECS.items():
            position = (*layout["object"], OBJECT_SPAWN_Z) if key == target_key else PARKED_POS
            objects[spec.name] = [*position, 1.0, 0.0, 0.0, 0.0]
        return dict(
            id=state_id,
            robot=copy.deepcopy(robot),
            objects=objects,
            task_vars=dict(target_key=target_key, layout=copy.deepcopy(layout)),
        )

    def begin_restored_episode(self, h, state, knobs):
        values = state["task_vars"]
        spec = OBJECT_SPECS[values["target_key"]]
        h.select_arm("right")
        seat = h.scene[spec.name].data.root_pos_w[0, :2].cpu().numpy().copy()
        layout = {name: h.scene[name].data.root_pos_w[0, :2].cpu().numpy().copy() for name in FIXTURES}
        layout["object"] = seat.copy()
        return self._new_ctx(values["target_key"], layout, seat)

    def validate_init_state(self, state):
        values = state["task_vars"]
        if values.get("target_key") not in OBJECT_SPECS:
            raise ValueError("texture_sort requires target_key=smooth or rough")
        layout = values.get("layout", {})
        if set(layout) != {"object", *FIXTURES}:
            raise ValueError("texture_sort requires object and cup layout coordinates")
        for name, xy in layout.items():
            xy = _np.asarray(xy, dtype=float)
            prim = OBJECT_SPECS[values["target_key"]].name if name == "object" else name
            if xy.shape != (2,) or not _np.allclose(xy, state["objects"][prim][:2], atol=0.01):
                raise ValueError(f"Layout and object pose disagree for {name}")

    def init_state_objects(self) -> tuple[str, ...]:
        # Both pucks: which one is on the table and which is parked is the
        # episode's hidden draw, and it has to come back the same way.
        return FIXTURES + tuple(s.name for s in OBJECT_SPECS.values())

    def snapshot_init_state(self, h, ctx: dict) -> dict:
        return {
            "target_key": ctx["target_key"],
            "layout": {k: [float(v[0]), float(v[1])] for k, v in ctx["layout"].items()},
        }

    def restore_init_state(self, h, state: dict, knobs: dict) -> dict:
        v = state["task_vars"]
        spec = OBJECT_SPECS[v["target_key"]]
        h.select_arm("right")
        # The pads sense one named prim, so they have to be pointed at the
        # puck this state uses before anything touches it.
        retarget_pads(h.sensors, spec.name)
        # The snapshot already follows settling. Advancing it again moves
        # the inactive body parked below the ground plane. Refresh transforms
        # for sensing without advancing the restored starting state.
        h.sim.forward()
        seat = h.scene[spec.name].data.root_pos_w[0, :2].detach().cpu().numpy()
        layout = {k: tuple(val) for k, val in v["layout"].items()}
        return self._new_ctx(v["target_key"], layout, seat)

    @staticmethod
    def _blade() -> _np.ndarray:
        """Unit blade direction at the fixed grasp orientation."""
        th = _np.radians(GRASP_PITCH_DEG)
        psi = _np.radians(GRASP_YAW_DEG)
        return _np.array([_np.cos(th) * _np.cos(psi), _np.cos(th) * _np.sin(psi), -_np.sin(th)])

    @staticmethod
    def _wrist_err_deg(h, ctx) -> float:
        """Angle between the TCP orientation and the locked grasp orientation."""
        if ctx["level_quat"] is None:
            return 180.0
        d = abs(float(_np.dot(h.tcp_quat(), _np.asarray(ctx["level_quat"], dtype=float))))
        return float(_np.degrees(2.0 * _np.arccos(min(1.0, d))))

    # ---- signals -----------------------------------------------------------
    def _ripple_mm(self, h, ctx) -> float:
        """RMS of the penetration map's high-frequency content, millimetres.

        The classifier's only input. A geometric quantity (SDF
        interpenetration), not a force. Averaged over whichever pads have a
        real contact patch, and zero when neither does, so a `classify` that
        never touched reads 0.0 and sorts as smooth rather than as noise.
        """
        vals = []
        for p in h.ch["pads"]:
            d = h.sensors[p].data
            if d.penetration_depth is None:
                continue
            pen = d.penetration_depth[0].view(1, 1, h.rows, h.cols)
            peak = float(pen.max())
            if peak < RIPPLE_CONTACT_M:
                continue
            blur = _torch.nn.functional.avg_pool2d(
                pen, kernel_size=RIPPLE_KERNEL, stride=1, padding=RIPPLE_KERNEL // 2, count_include_pad=False
            )
            resid = pen - blur
            # Only over the loaded patch: including the unloaded rim would
            # divide the ripple by however much of the pad happened to be free.
            mask = pen > 0.25 * peak
            if int(mask.sum()) < RIPPLE_PATCH_TAXELS:
                continue
            vals.append(float(resid[mask].pow(2).mean().sqrt()) * 1e3)
        return float(_np.mean(vals)) if vals else 0.0

    completion_on_success = True

    def signals(self, h, ctx) -> dict:
        pen = h.pad_penetration()
        return {
            "areas": h.pad_areas(),
            "gripper_gap_m": gel_gap(h.grip_measured()),
            "pen": pen,
            "pen_total": float(sum(pen.values())),
            "tilt": h.blade_tilt_deg(),
            "obj_pos": h.scene[ctx["obj_prim"]].data.root_pos_w[0].detach().cpu().numpy().copy(),
        }

    # ---- phases ------------------------------------------------------------
    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        seat, bins = ctx["seat"], ctx["bins"]
        grip = ctx["grip"]
        grasp = _np.array([seat[0], seat[1], GRASP_TCP_Z])
        pregrasp = grasp - self._blade() * APPROACH_BACKOFF + ctx["approach_offset"]
        if name == "home":
            # Jaws still shut, as HOME leaves them; `above_target` opens them.
            grip = START_JAW_WIDTH["right"]
            tgt = [pregrasp[0], pregrasp[1], pregrasp[2] + HOME_ABOVE_PREGRASP]
        elif name in ("above_target", "orient"):
            grip = OPEN_WIDTH
            tgt = pregrasp.tolist()
            # Position IK only until the jaws have opened the gels apart: the
            # jaw axis has no reliable sign while they overlap.
            if ctx["level_quat"] is None and gel_gap(h.grip_measured()) >= LEVEL_MIN_GEL_GAP:
                ctx["level_quat"] = oblique_grasp_quat(
                    h.tcp_quat(), h.jaw_axis(), h.approach_axis(), GRASP_PITCH_DEG, GRASP_YAW_DEG
                )
        elif name == "descend":
            grip = OPEN_WIDTH
            tgt = grasp.tolist()
            if ctx["gate_stage"] < 3:
                blade = self._blade()
                gate = grasp - blade * GATE_BACKOFF
                detour = ctx["descend_detour"]
                if ctx["gate_stage"] == 0:
                    waypoint = gate + _np.array([*detour, 0.0])
                elif ctx["gate_stage"] == 1:
                    waypoint = gate
                else:
                    waypoint = grasp - blade * ENTRY_PAUSE
                    length = float(_np.linalg.norm(detour))
                    if length > 1e-9:
                        waypoint += _np.array([*detour, 0.0]) * (ENTRY_RESIDUAL / length)
                at_command = ctx["cmd_pos"] is not None and _np.linalg.norm(waypoint - ctx["cmd_pos"]) < 0.002
                # Wait for the measured hand at the gate and the short stop.
                if at_command and (ctx["gate_stage"] == 0 or h.tcp_err(waypoint) < 0.005):
                    ctx["gate_stage"] += 1
                if ctx["gate_stage"] < 3:
                    tgt = waypoint.tolist()
        elif name == "close":
            # Latch on the MEASURED jaw position once the preload is reached;
            # the drive lags a moving command and would keep closing into it.
            preload = PRELOAD
            if ctx["grip_hold"] is not None:
                grip = ctx["grip_hold"]
            elif sig["pen_total"] < preload and grip > GRIP_MIN_WIDTH:
                rate = CLOSE_RATE_SLOW if sig["pen_total"] > FIRST_TOUCH_M else CLOSE_RATE_FAST
                grip = max(GRIP_MIN_WIDTH, grip - rate * h.control_dt)
            else:
                grip = ctx["grip_hold"] = h.grip_measured()
            tgt = [seat[0], seat[1], GRASP_TCP_Z]
        elif name == "classify":
            tgt = [seat[0], seat[1], GRASP_TCP_Z]
        elif name == "lift":
            tgt = [seat[0], seat[1], HOVER_TCP_Z]
        elif name == "above_bin":
            # The tactile branch. Everything downstream of here differs
            # between the two classes; nothing upstream does.
            b = bins[ctx["chosen_bin"]]
            tgt = [b[0], b[1], HOVER_TCP_Z]
        elif name == "lower":
            b = bins[ctx["chosen_bin"]]
            tgt = [b[0], b[1], BIN_DROP_Z]
        elif name == "open":
            # Hold the arm where it is while the jaws open, and drop the bias:
            # a biased command beyond the arm's reach makes it lunge the
            # instant the puck is released.
            grip = OPEN_WIDTH
            if ctx["release_xy"] is None:
                ctx["release_xy"] = h.tcp_pos()[:2].copy()
                ctx["xy_bias"][:] = 0.0
            tgt = [ctx["release_xy"][0], ctx["release_xy"][1], BIN_DROP_Z]
        else:  # retreat
            grip = OPEN_WIDTH
            tgt = [ctx["release_xy"][0], ctx["release_xy"][1], HOVER_TCP_Z]

        ctx["grip"] = grip
        ctx["goal"] = list(tgt)
        if name in BIAS_PHASES:
            ctx["xy_bias"][:] = _np.clip(
                ctx["xy_bias"]
                + XY_BIAS_GAIN * (h.control_dt / DT) * (_np.asarray(tgt[:2]) - h.tcp_pos()[:2]),
                -XY_BIAS_MAX,
                XY_BIAS_MAX,
            )
            tgt = [tgt[0] + ctx["xy_bias"][0], tgt[1] + ctx["xy_bias"][1], tgt[2]]
        speed = CARRY_SPEED if name in CARRY_PHASES else APPROACH_SPEED if name == "descend" else FREE_SPEED
        return self._rate_limited(h, ctx, tgt, speed), grip, ctx["level_quat"]

    def _rate_limited(self, h, ctx, goal, speed: float) -> list:
        """Advance the commanded point toward `goal` by at most `speed` * DT."""
        if ctx["cmd_pos"] is None:
            ctx["cmd_pos"] = h.tcp_pos().astype(float)
        delta = _np.asarray(goal, dtype=float) - ctx["cmd_pos"]
        dist = float(_np.linalg.norm(delta))
        step = speed * h.control_dt
        ctx["cmd_pos"] = (
            ctx["cmd_pos"] + delta * (step / dist) if dist > step else _np.asarray(goal, dtype=float)
        )
        return ctx["cmd_pos"].tolist()

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        # Completion reads the phase's true goal, never the biased,
        # rate-limited point that was commanded.
        target = ctx["goal"]
        tcp = h.tcp_pos()
        if name in ("home", "above_target"):
            return h.tcp_err(target) < 0.015
        if name == "orient":
            return t_phase > 0.3 and self._wrist_err_deg(h, ctx) < ORIENT_TOL_DEG
        if name == "descend":
            return ctx["gate_stage"] >= 3 and h.tcp_err(target) < 0.005
        if name == "close":
            ctx["blade_tilt_deg"] = round(sig["tilt"], 1)
            preload = PRELOAD
            both = all(v >= 0.3 * preload for v in sig["pen"].values())
            return (sig["pen_total"] >= preload and both) or ctx["grip"] <= GRIP_MIN_WIDTH
        if name == "classify":
            if sum(sig["areas"].values()) > AREA_CONTACT:
                ctx["ripples"].append(self._ripple_mm(h, ctx))
            done = t_phase >= 0.9 * self._timeout("classify")
            if done:
                # Median, not mean: one frame with the pad sliding across the
                # surface reads a ripple that has nothing to do with dimples.
                r = float(_np.median(ctx["ripples"])) if ctx["ripples"] else 0.0
                ctx["ripple_mm"] = round(r, 4)
                ctx["chosen_bin"] = 1 if r > ROUGH_THRESH_MM else 0
            return done
        if name == "lift":
            # Height is what this phase is for: under pose IK the arm settles
            # short in xy at hover height, so a spherical tolerance never fires.
            return abs(tcp[2] - target[2]) < 0.010
        if name == "above_bin":
            # On the TRUE error; the integral bias is what gets it there.
            return h.tcp_err(target) < 0.010
        if name == "lower":
            off = float(_np.linalg.norm(tcp[:2] - _np.asarray(target[:2])))
            ctx["drop_offset_mm"] = round(off * 1e3, 1)
            return abs(tcp[2] - target[2]) < 0.005 and off < 0.012
        if name == "open":
            return t_phase > 0.8 * self._timeout("open")
        if name == "retreat":
            return h.tcp_err(target) < 0.025
        return False

    # ---- scoring -----------------------------------------------------------
    def build_trace(self, h, ctx, hist) -> _Trace:
        pos = _np.asarray(hist["obj_pos"])
        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in h.ch["pads"]},
            object_pos={"target": pos},
        )

    def verdict(self, trace, ctx, knobs):
        result = _verdict(
            trace,
            bin_xy=_np.asarray([ctx["layout"][b] for b in FIXTURES]),
            correct_bin=ctx["correct_bin"],
            pads=tuple(trace.pad_area),
        )
        ctx["landed_bin"] = int(result.metrics.get("chosen_bin", -1))
        return result

    def episode_extra(self, ctx, hist) -> dict:
        return {
            "target_class": ctx["target_key"],
            "correct_bin": ctx["correct_bin"],
            "intended_bin": ctx["chosen_bin"],
            "chosen_bin": ctx.get("landed_bin", -1),
            "ripple_mm": ctx["ripple_mm"],
            "n_ripple_samples": len(ctx["ripples"]),
            "blade_tilt_deg": ctx["blade_tilt_deg"],
            "drop_offset_mm": ctx["drop_offset_mm"],
            "layout": {n: [round(float(v), 4) for v in ctx["layout"][n]] for n in ctx["layout"]},
        }

    def report_extra(self, knobs) -> dict:
        return {
            "light_scale": DEFAULT_LIGHT_SCALE,
            "rough_thresh_mm": ROUGH_THRESH_MM,
            "preload_mm": PRELOAD * 1e3,
            "grip_floor_mm": GRIP_MIN_WIDTH * 1e3,
            "approach_jitter_mm": APPROACH_JITTER * 1e3,
            "descend_detour_mm": DESCEND_DETOUR * 1e3,
            "entry_pause_mm": ENTRY_PAUSE * 1e3,
            "entry_residual_mm": ENTRY_RESIDUAL * 1e3,
            "pair_episodes": False,
            "preload_jitter": 0.0,
            "sdf_resolution": SDF_RESOLUTION,
            "grasp_pitch_deg": GRASP_PITCH_DEG,
            "grasp_yaw_deg": GRASP_YAW_DEG,
        }


TASK = TextureSortTask()
