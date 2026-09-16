"""handover scene: one bar, long enough for both grippers to hold at once.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.handover               + the bar                     (this file)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from sim.envs.bimanual_tactile import (
    REAL_HOME_JOINT_POS,
    BimanualTactileSceneCfg,
    point_pads_at,
    start_joint_pos,
)

__all__ = [
    "BAR_LENGTH",
    "BAR_MASS",
    "BAR_THICKNESS",
    "GIVE_XY",
    "HANDOVER_Z",
    "PAD_TARGET",
    "SPAWN_JITTER",
    "SPAWN_Z_JITTER",
    "STANDOFF_Z",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "HandoverSceneCfg",
]


# ----- objects ---------------------------------------------------------------
# The bar: a square section, metres. 40 mm sits mid-stroke for the jaws and
# is wider than the 24 mm gel window, so the contact patch is set by the pad.
BAR_THICKNESS = 0.040

BAR_LENGTH = 0.280

# Bar mass, kg.
BAR_MASS = 0.02


# ----- layout ----------------------------------------------------------------
# Where the giver (left) holds the bar out, metres; the receiver's grasp
# point follows from the bar's length. Inside both arms' reach at HANDOVER_Z.
GIVE_XY = (0.26, 0.08)

HANDOVER_Z = 0.15              # bar height at the exchange; no table under it

STANDOFF_Z = 0.28              # where the receiver waits above the exchange

SPAWN_JITTER = 0.015           # metres, uniform +/- on the giver's xy, per episode

SPAWN_Z_JITTER = 0.010         # metres, uniform +/- on HANDOVER_Z, per episode


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw: shut, as the
# robot's HOME leaves them.
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
# Arm pose at the start of every episode: the robot's own HOME.
START_JOINT_POS = REAL_HOME_JOINT_POS
# The prim every tactile pad senses.
PAD_TARGET = "{ENV_REGEX_NS}/manipuland"


# ----- spawn helpers ---------------------------------------------------------
def _spawn_bar() -> RigidObjectCfg:
    """The bar, a mesh so TacSL can sense it, parked below the floor until
    the expert places it."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/manipuland",
        spawn=sim_utils.MeshCuboidCfg(
            # Long axis is +y, which is also the axis the two grippers are
            # separated along. The controller reports the bar's tilt off
            # horizontal about this axis.
            size=(BAR_THICKNESS, BAR_LENGTH, BAR_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=BAR_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.1, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.80, 0.55, 0.20)),
        ),
        # Parked below the floor; the expert places it once the receiving
        # arm is out of the way.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class HandoverSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + a bar, and nothing else.

    Set `robot.spawn.usd_path` before constructing, as with the parent class.
    """

    # Replaces the minimal unit's placeholder; every pad already senses
    # `manipuland`.
    manipuland = _spawn_bar()

    def __post_init__(self):
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)
