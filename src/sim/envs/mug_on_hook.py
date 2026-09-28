"""mug_on_hook scene: a mug on the table and a wall rail with one peg.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.mug_on_hook            + table + rail + mug      (this file)
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.mug_on_hook import HANDLE_SECTION, STOP_BALL_R, peg_span
from sim.envs.bimanual_tactile import (
    REAL_HOME_JOINT_POS,
    TABLE_TOP_Z,
    BimanualTactileSceneCfg,
    point_pads_at,
    start_joint_pos,
)
from sim.envs.preview import PreviewSpec

__all__ = [
    "ASSETS_DIR",
    "ASSET_NAMES",
    "HANG_ALONG",
    "MUG_JITTER",
    "MUG_MASS",
    "MUG_SPAWN_Z",
    "MUG_XY",
    "MUG_YAW_JITTER_DEG",
    "PAD_TARGET",
    "PREVIEW",
    "RAIL_JITTER",
    "RAIL_Z_JITTER",
    "RAIL_XY",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "MugOnHookSceneCfg",
    "ensure_assets",
    "make_preview_cfg",
    "sample_layout",
]


# ----- assets ----------------------------------------------------------------
# Generated USDs for this task. If they are missing, or stale after an edit to
# `sim.dims.mug_on_hook`, regenerate them with
#     python benchmark/run/make_mug_on_hook_assets.py
# (the installer does this for every task).
ASSETS_DIR = task_asset_dir("mug_on_hook")

ASSET_NAMES = ("mug", "hook")


def _usd(name: str) -> Path:
    """Path to one of this task's generated USDs."""
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    """Raise, naming the command that fixes it, if any USD is missing."""
    missing = [n for n in ASSET_NAMES if not _usd(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"mug_on_hook geometry missing from {ASSETS_DIR}: {', '.join(missing)}\n"
            f"    python benchmark/run/make_mug_on_hook_assets.py --headless"
        )


# ----- objects ---------------------------------------------------------------
# The mug's mass, kg. Two flat gels meet a 38 mm cylinder along a line, so the
# hold is far weaker than the contact area suggests; this is light enough to
# carry without creeping and heavy enough that its shear, and the moment that
# shear goes, are well clear of the noise.
MUG_MASS = 0.045


# ----- layout ----------------------------------------------------------------
# The rail's board face, metres: far enough out on the right arm's side that
# the hanging point stays inside the arm's reachable y band and the near
# finger keeps clear of the board.
RAIL_XY = (0.265, -0.240)

# How far along the peg, from the board's face, the eye is threaded to, metres.
# The mug does not stay there: the rod falls, so it runs down to the stop ball
# and comes to rest against it. This is only where the hand lets go of it.
HANG_ALONG = 0.053


def rest_along() -> float:
    """How far out the mug comes to rest, metres: against the stop ball."""
    return peg_span() - STOP_BALL_R - HANDLE_SECTION / 2.0


# Mug start centre, metres: inside the right arm's top-grasp region.
MUG_XY = (0.245, -0.090)

# Metres, uniform +/- on x and y, per episode.
RAIL_JITTER = 0.005
# The fixture's height is unknown to the scripted controller. Its vertical
# contact point is found while seating the mug.
RAIL_Z_JITTER = 0.006
MUG_JITTER = 0.012
# The handle's heading varies a little about +x, degrees.
MUG_YAW_JITTER_DEG = 6.0

# The mug's base at spawn: standing on the table.
MUG_SPAWN_Z = TABLE_TOP_Z


def sample_layout(
    rng,
    rail_jitter: float = RAIL_JITTER,
    mug_jitter: float = MUG_JITTER,
    yaw_jitter_deg: float = MUG_YAW_JITTER_DEG,
    rail_z_jitter: float = RAIL_Z_JITTER,
) -> dict:
    """Draw the rail's xyz and the mug's pose on the table."""
    out: dict = {}
    dx, dy = rng.uniform(-rail_jitter, rail_jitter, 2) if rail_jitter else (0.0, 0.0)
    out["rail"] = (float(RAIL_XY[0] + dx), float(RAIL_XY[1] + dy))
    out["rail_z"] = float(TABLE_TOP_Z + rng.uniform(-rail_z_jitter, rail_z_jitter))
    dx, dy = rng.uniform(-mug_jitter, mug_jitter, 2) if mug_jitter else (0.0, 0.0)
    out["mug"] = (float(MUG_XY[0] + dx), float(MUG_XY[1] + dy))
    out["mug_yaw"] = float(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
    return out


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw: shut, as the
# robot's HOME leaves them.
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
# Arm pose at the start of every episode: the robot's own HOME.
START_JOINT_POS = REAL_HOME_JOINT_POS
# The prim every tactile pad senses. One arm does this task; all four pads
# watch the mug, so no pad switches target mid-episode.
PAD_TARGET = "{ENV_REGEX_NS}/mug"


# ----- spawn helpers ---------------------------------------------------------
def _spawn_mug(prim: str) -> RigidObjectCfg:
    """The mug, standing on the table."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd("mug")),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Depenetration well under the default metre per second: a
                # body this light leaves a deep contact at that speed.
                disable_gravity=False,
                max_depenetration_velocity=0.2,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=MUG_MASS),
            # Contact generated a millimetre and a half out, so the solver
            # answers the gels, the table and the peg before any is deep in;
            # well under the stop ball's 2.5 mm sideways clearance.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*MUG_XY, MUG_SPAWN_Z)),
    )


def _spawn_fixture(prim: str, mesh: str, xy, z: float) -> RigidObjectCfg:
    """The rail: kinematic, so `reset` can re-pose it."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(mesh)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*xy, z)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class MugOnHookSceneCfg(BimanualTactileSceneCfg):
    """Robot, tactile pads, table, mug and rail."""

    manipuland = None

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.75, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.52, 0.42, 0.32)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.0, TABLE_TOP_Z - 0.02)),
    )

    hook = _spawn_fixture("{ENV_REGEX_NS}/hook", "hook", RAIL_XY, TABLE_TOP_Z)
    mug = _spawn_mug("{ENV_REGEX_NS}/mug")

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(intensity=3000.0, radius=0.05, color=(1.0, 0.98, 0.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.51, -0.02, 0.70)),
    )

    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.SphereLightCfg(intensity=250.0, radius=0.12, color=(0.92, 0.95, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, -0.55, 0.55)),
    )

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)


PREVIEW = PreviewSpec(
    sdf_prims=("mug",),
    tri_prims=("hook",),
    work_target=(0.265, -0.170, TABLE_TOP_Z + 0.07),
    sdf_resolution=128,
)


def make_preview_cfg() -> MugOnHookSceneCfg:
    return MugOnHookSceneCfg(num_envs=1, env_spacing=3.0)
