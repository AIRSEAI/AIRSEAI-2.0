"""stack_cube expert.

Phases::

    home -> above_a -> orient -> descend -> close -> lift -> above_b -> lower
        -> open -> retreat
"""

from __future__ import annotations

import copy

import numpy as _np

from sim.criteria.stack_cube import verdict as _verdict
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.stack_cube import (
    CUBE_SIZE,
)
from sim.envs.bimanual_tactile import RIGHT_ARM_SENSORS as _GRASP_PADS
from sim.envs.bimanual_tactile import TABLE_TOP_Z, gel_gap
from sim.envs.stack_cube import (
    CUBE_A_XY,
    CUBE_B_XY,
    SPAWN_JITTER,
    StackCubeSceneCfg,
)
from sim.tactile import apply_sdf_collision
from sim.tasks.scene_utils import oblique_grasp_quat
from sim.tasks.spec import Knob, Task

# Controller parameters.
HOVER = 0.10  # m, transit height above the table top
# Grasp height of the TCP (gel centre) above the table, m. Pinned by the
# sensor housing below the gel, which must clear the table.
GRASP_TCP_ABOVE_TABLE = 0.034
# Jaw commands, m per jaw: open clears the cube, close squeezes about
# 1.5 mm into the gel.
OPEN_WIDTH = 0.027
CLOSE_WIDTH = 0.02225
ORIENT_TOL_DEG = 3.0  # wrist within this of its grasp orientation, degrees
# m back along the blade from the grasp point to the pre-grasp point.
APPROACH_BACKOFF = 0.060
HOME_ABOVE_PREGRASP = 0.030  # m above the pre-grasp point where `home` ends
APPROACH_SPEED = 0.10  # m/s, the command point's speed into the grasp
DT = 1.0 / 60.0
# Measured gel gap, m, before the grasp orientation is computed: the jaw axis
# has no reliable sign while the gels overlap.
LEVEL_MIN_GEL_GAP = 0.010
# Physics steps the cubes are given to come to rest after they are placed.
SETTLE_STEPS = 30


class StackCubeTask(Task):
    name = "stack_cube"
    instruction = "stack the red cube on the blue cube"
    report_name = "stack_cube_scripted.json"
    knobs = {
        "open_width": Knob(OPEN_WIDTH, "jaw command while approaching (m)"),
        "close_width": Knob(CLOSE_WIDTH, "jaw command while holding (m)"),
        "grasp_pitch_deg": Knob(45.0, "blade pitch below horizontal at the grasp, deg"),
        "grasp_yaw_deg": Knob(0.0, "blade heading in the world xy plane, deg; 0 is +x"),
    }
    # Advancement is by condition where one exists (position reached, contact
    # made), by timeout otherwise -- a pure-timeout machine hides IK failures.
    phases = [
        ("home", 3.0),
        ("above_a", 2.0),
        ("orient", 2.0),
        ("descend", 2.0),
        ("close", 0.8),
        ("lift", 1.2),
        ("above_b", 2.0),
        ("lower", 1.5),
        ("open", 0.6),
        ("retreat", 1.0),
    ]
    handover_phase = None  # the policy drives from t = 0, the robot's HOME
    horizon_s = 25.0
    scene_cam = (
        (0.85, -0.45, 0.45 + 0.10),
        (0.5 * (CUBE_A_XY[0] + CUBE_B_XY[0]), 0.5 * (CUBE_A_XY[1] + CUBE_B_XY[1]), TABLE_TOP_Z + 0.05),
    )

    def make_scene_cfg(self, knobs):
        return StackCubeSceneCfg(num_envs=1, env_spacing=3.0)

    def apply_collision(self, h) -> None:
        for prim in ("cube_a", "cube_b"):
            n = apply_sdf_collision(f"/World/envs/env_0/{prim}")
            print(f"STEP sdf on {prim}: {n and len(n)} mesh(es)", flush=True)

    def reset(self, h, rng, ep, knobs) -> dict:
        h.select_arm("right")
        a_xy = _np.array(CUBE_A_XY) + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
        b_xy = _np.array(CUBE_B_XY) + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
        z = TABLE_TOP_Z + CUBE_SIZE / 2 + 0.002
        h.place(h.scene["cube_a"], a_xy, z)
        h.place(h.scene["cube_b"], b_xy, z)
        h.reset_robot()
        h.hold_default(SETTLE_STEPS)
        return self._new_ctx(h, "right")

    def _new_ctx(self, h, side: str) -> dict:
        """The episode's working state, however the episode was started. The
        targets are the settled, MEASURED cube positions, not the commanded
        ones."""
        a0 = h.scene["cube_a"].data.root_pos_w[0].detach().cpu().numpy().copy()
        b0 = h.scene["cube_b"].data.root_pos_w[0].detach().cpu().numpy().copy()
        return {"side": side, "a0": a0, "b0": b0, "held_area": [], "level_quat": None, "approach_cmd": None}

    # ---- frozen initial states ---------------------------------------------

    restore_contact_clearance_m = 1.0
    settle_seconds = SETTLE_STEPS * DT

    def sample_init_state(self, robot, rng, state_id, knobs):
        z = TABLE_TOP_Z + CUBE_SIZE / 2 + 0.002
        objects = {}
        for name, xy in (("cube_a", CUBE_A_XY), ("cube_b", CUBE_B_XY)):
            position = _np.asarray(xy) + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
            objects[name] = [*position.tolist(), z, 1.0, 0.0, 0.0, 0.0]
        return dict(id=state_id, robot=copy.deepcopy(robot), objects=objects, task_vars={"side": "right"})

    def begin_restored_episode(self, h, state, knobs):
        side = state["task_vars"]["side"]
        if side != "right":
            raise ValueError("stack_cube requires the right arm")
        h.select_arm(side)
        return self._new_ctx(h, side)

    def validate_init_state(self, state):
        if state["task_vars"].get("side") != "right":
            raise ValueError("stack_cube requires side=right")

    def init_state_objects(self) -> tuple[str, ...]:
        return ("cube_a", "cube_b")

    def snapshot_init_state(self, h, ctx: dict) -> dict:
        # Where the cubes ended up is measured off the scene; the arm is the
        # only other thing the episode chose.
        return {"side": ctx["side"]}

    def restore_init_state(self, h, state: dict, knobs: dict) -> dict:
        side = state["task_vars"]["side"]
        h.select_arm(side)
        h.hold_default(SETTLE_STEPS)
        return self._new_ctx(h, side)

    def signals(self, h, ctx) -> dict:
        return {
            "areas": h.pad_areas(),
            "tilt": h.blade_tilt_deg(),
            "a_pos": h.scene["cube_a"].data.root_pos_w[0].detach().cpu().numpy().copy(),
            "b_pos": h.scene["cube_b"].data.root_pos_w[0].detach().cpu().numpy().copy(),
        }

    @staticmethod
    def _blade(knobs) -> _np.ndarray:
        """Unit blade direction at the grasp, from the pitch and heading knobs."""
        th = _np.radians(float(knobs["grasp_pitch_deg"]))
        psi = _np.radians(float(knobs["grasp_yaw_deg"]))
        return _np.array([_np.cos(th) * _np.cos(psi), _np.cos(th) * _np.sin(psi), -_np.sin(th)])

    @staticmethod
    def _wrist_err_deg(h, ctx) -> float:
        """Angle between the TCP orientation and the locked grasp orientation."""
        if ctx["level_quat"] is None:
            return 180.0
        d = abs(float(_np.dot(h.tcp_quat(), _np.asarray(ctx["level_quat"], dtype=float))))
        return float(_np.degrees(2.0 * _np.arccos(min(1.0, d))))

    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        a0, b0 = ctx["a0"], ctx["b0"]
        ow, cw = float(knobs["open_width"]), float(knobs["close_width"])
        stack_z = TABLE_TOP_Z + CUBE_SIZE + GRASP_TCP_ABOVE_TABLE + 0.004
        grasp = _np.array([a0[0], a0[1], TABLE_TOP_Z + GRASP_TCP_ABOVE_TABLE])
        pregrasp = grasp - self._blade(knobs) * APPROACH_BACKOFF
        if name == "home":
            # Jaws still shut, as HOME leaves them; `above_a` opens them.
            return [pregrasp[0], pregrasp[1], pregrasp[2] + HOME_ABOVE_PREGRASP], 0.0, None
        if name == "above_a":
            return pregrasp.tolist(), ow, None
        # Approach on position IK, then turn the wrist in place and keep that
        # orientation locked for the rest of the episode: commanding position
        # and orientation together from afar stalls short of the target. The
        # jaw axis is only meaningful once the gels have opened apart.
        if ctx["level_quat"] is None and gel_gap(h.grip_measured()) >= LEVEL_MIN_GEL_GAP:
            ctx["level_quat"] = oblique_grasp_quat(
                h.tcp_quat(),
                h.jaw_axis(),
                h.approach_axis(),
                float(knobs["grasp_pitch_deg"]),
                float(knobs["grasp_yaw_deg"]),
            )
        q = ctx["level_quat"]
        if name == "orient":
            return pregrasp.tolist(), ow, q
        if name == "descend":
            # In along the blade: the command point slides from where the TCP
            # is to the grasp point at APPROACH_SPEED.
            if ctx["approach_cmd"] is None:
                ctx["approach_cmd"] = h.tcp_pos().astype(float)
            d = grasp - ctx["approach_cmd"]
            n = float(_np.linalg.norm(d))
            step = APPROACH_SPEED * h.control_dt
            ctx["approach_cmd"] = ctx["approach_cmd"] + d * (step / n) if n > step else grasp
            ctx["approach_goal"] = grasp.tolist()
            return ctx["approach_cmd"].tolist(), ow, q
        if name == "close":
            return [a0[0], a0[1], TABLE_TOP_Z + GRASP_TCP_ABOVE_TABLE], cw, q
        if name == "lift":
            return [a0[0], a0[1], TABLE_TOP_Z + HOVER], cw, q
        if name == "above_b":
            return [b0[0], b0[1], TABLE_TOP_Z + HOVER], cw, q
        if name == "lower":
            # cube A's bottom must land on cube B's top
            return [b0[0], b0[1], stack_z], cw, q
        if name == "open":
            return [b0[0], b0[1], stack_z], ow, q
        return [b0[0], b0[1], TABLE_TOP_Z + HOVER + 0.05], ow, q  # retreat

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        if name in ("lift", "above_b"):
            ctx["held_area"].append(sum(sig["areas"].values()))
        if name in ("home", "above_a", "lift", "above_b", "retreat"):
            return h.tcp_err(target) < 0.015
        if name == "orient":
            return (
                t_phase > 0.3 and self._wrist_err_deg(h, ctx) < ORIENT_TOL_DEG and h.tcp_err(target) < 0.015
            )
        if name == "descend":
            # The true grasp point, not the sliding command point.
            return h.tcp_err(ctx["approach_goal"]) < 0.008
        if name == "close":
            return t_phase > 0.4 and sum(sig["areas"].values()) > 5.0
        if name in ("open", "lower"):
            return t_phase > dict(self.phases)[name] * 0.8
        return False

    def build_trace(self, h, ctx, hist) -> _Trace:
        from sim.envs.bimanual_tactile import TACTILE_SENSORS

        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={"cube_a": _np.asarray(hist["a_pos"]), "cube_b": _np.asarray(hist["b_pos"])},
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(trace, cube_size=CUBE_SIZE, pads=_GRASP_PADS)

    def episode_extra(self, ctx, hist) -> dict:
        ha = ctx["held_area"]
        return {
            "a0": _np.asarray(ctx["a0"]).tolist(),
            "b0": _np.asarray(ctx["b0"]).tolist(),
            "held_area_median": float(_np.median(ha)) if ha else 0.0,
            "final_a": hist["a_pos"][-1].tolist() if hist["a_pos"] else None,
            "final_b": hist["b_pos"][-1].tolist() if hist["b_pos"] else None,
        }


TASK = StackCubeTask()
