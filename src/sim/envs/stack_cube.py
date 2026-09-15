"""stack_cube scene: a table and two cubes.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.stack_cube             + table + two cubes           (this file)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.dims.stack_cube import (
    CUBE_SIZE,
)
from sim.envs.bimanual_tactile import (
    ARM_LIFTED_JOINT_POS,
    TABLE_TOP_Z,
    BimanualTactileSceneCfg,
    point_pads_at,
    start_joint_pos,
)

__all__ = [
    "CUBE_A_XY",
    "CUBE_B_XY",
    "PAD_TARGET",
    "SPAWN_JITTER",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "StackCubeSceneCfg",
]


# ----- layout ----------------------------------------------------------------
# Nominal cube positions, metres. x = 0.26 is the table-top line every task
# puts its manipulanda on: the arm's top-grasp reach at this table height.
CUBE_A_XY = (0.26, -0.15)      # the cube that is picked

CUBE_B_XY = (0.26, -0.02)      # the cube it is stacked onto

SPAWN_JITTER = 0.02            # metres, uniform +/- on x and y, per episode


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw.
START_JAW_WIDTH = {"right": 0.020, "left": 0.010}
# Arm pose at the start of every episode.
START_JOINT_POS = ARM_LIFTED_JOINT_POS
# The prim every tactile pad senses. TacSL follows one prim per pad.
PAD_TARGET = "{ENV_REGEX_NS}/cube_a"


# ----- spawn helpers ---------------------------------------------------------
def _spawn_cube(prim: str, xy: tuple[float, float], colour: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim,
        # Meshes, not primitives: TacSL needs an SDF collision mesh on
        # anything the pads sense.
        spawn=sim_utils.MeshCuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.1, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], TABLE_TOP_Z + CUBE_SIZE / 2 + 0.002)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class StackCubeSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + table + two cubes.

    Cube poses here are nominal; the expert randomises them per episode.
    Set `robot.spawn.usd_path` to the grafted asset before constructing, as
    with the parent class.
    """

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.75, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.66, 0.55)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.32, 0.0, TABLE_TOP_Z - 0.02)),
    )

    # In front of the right gripper. cube_a is picked and stacked onto cube_b.
    cube_a = _spawn_cube("{ENV_REGEX_NS}/cube_a", CUBE_A_XY, (0.85, 0.25, 0.20))
    cube_b = _spawn_cube("{ENV_REGEX_NS}/cube_b", CUBE_B_XY, (0.20, 0.35, 0.85))

    def __post_init__(self):
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)
