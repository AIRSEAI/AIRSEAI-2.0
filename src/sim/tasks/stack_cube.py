"""stack_cube expert.

Phases::

    home -> above_a -> orient -> descend -> close -> lift -> above_b -> lower
        -> open -> retreat
"""

from __future__ import annotations

import numpy as _np

from sim.criteria.tasks import stack_cube as _verdict
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.stack_cube import (
    CUBE_SIZE,
)
from sim.envs.bimanual_tactile import RIGHT_ARM_SENSORS as _GRASP_PADS
from sim.envs.bimanual_tactile import TABLE_TOP_Z
from sim.envs.stack_cube import (
    CUBE_A_XY,
    CUBE_B_XY,
    SPAWN_JITTER,
    StackCubeSceneCfg,
)
from sim.tactile import apply_sdf_collision
from sim.tasks.scene_utils import top_down_grasp_quat
from sim.tasks.spec import Knob, Task

# Controller parameters.
HOVER = 0.10                  # m, transit height above the table top
# Grasp height of the TCP (gel centre) above the table, m. Pinned by the
# sensor housing below the gel, which must clear the table.
GRASP_TCP_ABOVE_TABLE = 0.034
# Jaw commands, m per jaw: open clears the cube, close squeezes about
# 1.5 mm into the gel.
OPEN_WIDTH = 0.027
CLOSE_WIDTH = 0.02225
ORIENT_TILT_DEG = 3.0         # blade tilt that counts as levelled, degrees


class StackCubeTask(Task):
    name = "stack_cube"
    instruction = "stack the red cube on the blue cube"
    report_name = "stack_cube_scripted.json"
    knobs = {
        "open_width": Knob(OPEN_WIDTH, "jaw command while approaching (m)"),
        "close_width": Knob(CLOSE_WIDTH, "jaw command while holding (m)"),
    }
    # Advancement is by condition where one exists (position reached, contact
    # made), by timeout otherwise -- a pure-timeout machine hides IK failures.
    phases = [
        ("home", 1.2), ("above_a", 2.0), ("orient", 2.0), ("descend", 1.5),
        ("close", 0.8), ("lift", 1.2), ("above_b", 2.0), ("lower", 1.5),
        ("open", 0.6), ("retreat", 1.0),
    ]
    handover_phase = None          # the policy drives from t = 0
    horizon_s = 25.0
    scene_cam = ((0.85, -0.45, 0.45 + 0.10),
                 (0.5 * (CUBE_A_XY[0] + CUBE_B_XY[0]),
                  0.5 * (CUBE_A_XY[1] + CUBE_B_XY[1]), TABLE_TOP_Z + 0.05))

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
        h.hold_default(30)
        # Targets are the settled, MEASURED positions, not the commanded ones.
        a0 = h.scene["cube_a"].data.root_pos_w[0].detach().cpu().numpy().copy()
        b0 = h.scene["cube_b"].data.root_pos_w[0].detach().cpu().numpy().copy()
        return {"side": "right", "a0": a0, "b0": b0, "held_area": [],
                "level_quat": None}

    def signals(self, h, ctx) -> dict:
        return {"areas": h.pad_areas(),
                "tilt": h.blade_tilt_deg(),
                "a_pos": h.scene["cube_a"].data.root_pos_w[0].detach().cpu().numpy().copy(),
                "b_pos": h.scene["cube_b"].data.root_pos_w[0].detach().cpu().numpy().copy()}

    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        a0, b0 = ctx["a0"], ctx["b0"]
        ow, cw = float(knobs["open_width"]), float(knobs["close_width"])
        stack_z = TABLE_TOP_Z + CUBE_SIZE + GRASP_TCP_ABOVE_TABLE + 0.004
        if name == "home":
            return [a0[0], a0[1], TABLE_TOP_Z + HOVER + 0.05], ow, None
        if name == "above_a":
            return [a0[0], a0[1], TABLE_TOP_Z + HOVER], ow, None
        # Approach on position IK, then level the wrist in place and keep that
        # orientation locked for the rest of the episode: commanding position
        # and orientation together from afar stalls short of the target.
        if ctx["level_quat"] is None:
            ctx["level_quat"] = top_down_grasp_quat(
                h.tcp_quat(), h.jaw_axis(), h.approach_axis())
        q = ctx["level_quat"]
        if name == "orient":
            return [a0[0], a0[1], TABLE_TOP_Z + HOVER], ow, q
        if name == "descend":
            return [a0[0], a0[1], TABLE_TOP_Z + GRASP_TCP_ABOVE_TABLE], ow, q
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
        return [b0[0], b0[1], TABLE_TOP_Z + HOVER + 0.05], ow, q   # retreat

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        if name in ("lift", "above_b"):
            ctx["held_area"].append(sum(sig["areas"].values()))
        if name in ("home", "above_a", "lift", "above_b", "retreat"):
            return h.tcp_err(target) < 0.015
        if name == "orient":
            return (t_phase > 0.3 and sig["tilt"] < ORIENT_TILT_DEG
                    and h.tcp_err(target) < 0.015)
        if name == "descend":
            return h.tcp_err(target) < 0.008
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
            object_pos={"cube_a": _np.asarray(hist["a_pos"]),
                        "cube_b": _np.asarray(hist["b_pos"])},
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(trace, cube_size=CUBE_SIZE, pads=_GRASP_PADS)

    def episode_extra(self, ctx, hist) -> dict:
        ha = ctx["held_area"]
        return {"a0": _np.asarray(ctx["a0"]).tolist(), "b0": _np.asarray(ctx["b0"]).tolist(),
                "held_area_median": float(_np.median(ha)) if ha else 0.0,
                "final_a": hist["a_pos"][-1].tolist() if hist["a_pos"] else None,
                "final_b": hist["b_pos"][-1].tolist() if hist["b_pos"] else None}


TASK = StackCubeTask()
