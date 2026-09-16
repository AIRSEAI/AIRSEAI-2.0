"""handover_cup_transfer scene: a bar standing in one cup, to be moved to the other.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.handover_cup_transfer  + table + two cups + the bar  (this file)
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.handover_cup_transfer import (
    CUP_COLOR,
    CUP_FLOOR,
    CUP_HEIGHT,
)
from sim.envs.bimanual_tactile import (
    REAL_HOME_JOINT_POS,
    TABLE_TOP_Z,
    BimanualTactileSceneCfg,
    point_pads_at,
    start_joint_pos,
)

__all__ = [
    "ASSETS_DIR",
    "ASSET_NAMES",
    "BAR_FOOT_Z",
    "BAR_LENGTH",
    "BAR_MASS",
    "BAR_SECTION",
    "CUP_RIM_Z",
    "CUP_XY",
    "DEST",
    "PAD_TARGET",
    "SOURCE",
    "SPAWN_JITTER",
    "SPAWN_YAW_JITTER",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "HandoverCupTransferSceneCfg",
    "ensure_assets",
]


# ----- assets ----------------------------------------------------------------
# Generated USDs for this task. If they are missing, or stale after an
# edit to `sim.dims.handover_cup_transfer`, regenerate them with
#     python benchmark/run/make_handover_cup_transfer_assets.py
# (the installer does this for every task).
ASSETS_DIR = task_asset_dir("handover_cup_transfer")

# One cup per colour.
ASSET_NAMES = tuple(f"cup_{n}" for n in CUP_COLOR)


def _usd(name: str) -> Path:
    """Path to one of this task's generated USDs."""
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    """Raise, naming the command that fixes it, if any USD is missing."""
    missing = [n for n in ASSET_NAMES if not _usd(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"handover_cup_transfer geometry missing from {ASSETS_DIR}: {', '.join(missing)}\n"
            f"    python benchmark/run/make_handover_cup_transfer_assets.py --headless")


# ----- objects ---------------------------------------------------------------
# Cross-section (x, y) at yaw 0, metres. The jaws close along x, so the 2 mm
# side is pinched and the 5 mm side is the face the gel bears against.
BAR_SECTION = (0.002, 0.005)

# Length, metres: long enough for both gels to sit fully on the bar 35 mm
# apart with the lower grasp still inside the palm's channel.
BAR_LENGTH = 0.140

BAR_MASS = 0.008                     # kg

SOURCE, DEST = "amber", "teal"


# ----- layout ----------------------------------------------------------------
# Cup centres, metres. The bar starts in the SOURCE cup.
CUP_XY = {"amber": (0.26, 0.15), "teal": (0.26, -0.15)}

BAR_FOOT_Z = TABLE_TOP_Z + CUP_FLOOR

CUP_RIM_Z = TABLE_TOP_Z + CUP_HEIGHT

SPAWN_JITTER = 0.0005                # metres, uniform +/- on the bar's xy, per episode

SPAWN_YAW_JITTER = 20.0              # degrees, uniform +/- on the bar's yaw, per episode


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw: shut, as the
# robot's HOME leaves them.
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
# Arm pose at the start of every episode: the robot's own HOME.
START_JOINT_POS = REAL_HOME_JOINT_POS
# The prim every tactile pad senses.
PAD_TARGET = "{ENV_REGEX_NS}/bar"


# ----- spawn helpers ---------------------------------------------------------
def _spawn_bar(prim: str, xy: tuple[float, float], z: float) -> RigidObjectCfg:
    w, t = BAR_SECTION
    return RigidObjectCfg(
        prim_path=prim,
        # Mesh, not primitive: TacSL needs an SDF collision mesh on anything a
        # sensor tracks, and the bar is the only sensed object here.
        spawn=sim_utils.MeshCuboidCfg(
            # Long axis local +z, so `tilt_deg` reads how far off upright
            # directly.
            size=(w, t, BAR_LENGTH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=BAR_MASS),
            # Contact offset well under the bar's 2 mm thickness, so the bore
            # walls do not see each other through it.
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.1, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.90, 0.88, 0.30), metallic=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], z)),
    )


def _spawn_cup(prim: str, name: str) -> RigidObjectCfg:
    """A cup, kinematic so `reset` can re-pose it each episode."""
    usd = _usd(f"cup_{name}")
    xy = CUP_XY[name]
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], TABLE_TOP_Z)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class HandoverCupTransferSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + table + two cups + the bar.

    Set `robot.spawn.usd_path` before constructing, as with the parent class.
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

    cup_src = _spawn_cup("{ENV_REGEX_NS}/cup_src", SOURCE)
    cup_dst = _spawn_cup("{ENV_REGEX_NS}/cup_dst", DEST)

    bar = _spawn_bar("{ENV_REGEX_NS}/bar", CUP_XY[SOURCE],
                     BAR_FOOT_Z + BAR_LENGTH / 2)

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=3000.0, radius=0.05, color=(1.0, 0.98, 0.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.51, -0.02, 0.70)),
    )

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)
