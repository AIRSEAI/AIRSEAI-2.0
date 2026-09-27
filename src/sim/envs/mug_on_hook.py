"""mug_on_hook scene: a mug standing on a coaster, and a wall rail with one peg.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.mug_on_hook            + table + rail + coaster + mug      (this file)
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.mug_on_hook import (
    COASTER_FLOOR,
    COASTER_INNER_R,
    HANDLE_SECTION,
    MUG_OUTER_R,
    STOP_BALL_R,
    hang_offset_x,
    hang_offset_z,
    peg_axis_z,
    peg_span,
)
from sim.envs.bimanual_tactile import (
    REAL_HOME_JOINT_POS,
    TABLE_TOP_Z,
    BimanualTactileSceneCfg,
    park_default_manipuland,
    point_pads_at,
    start_joint_pos,
)

__all__ = [
    "ASSETS_DIR",
    "ASSET_NAMES",
    "COASTER_JITTER",
    "COASTER_XY",
    "HANG_ALONG",
    "MUG_JITTER",
    "MUG_MASS",
    "MUG_SPAWN_Z",
    "MUG_YAW_JITTER_DEG",
    "PAD_TARGET",
    "PARKED_POS",
    "PREVIEW_CLEARANCE",
    "RAIL_JITTER",
    "RAIL_XY",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "MugOnHookSceneCfg",
    "ensure_assets",
    "hang_pose",
    "sample_layout",
]


# ----- assets ----------------------------------------------------------------
# Generated USDs for this task. If they are missing, or stale after an edit to
# `sim.dims.mug_on_hook`, regenerate them with
#     python benchmark/run/make_mug_on_hook_assets.py
# (the installer does this for every task).
ASSETS_DIR = task_asset_dir("mug_on_hook")

ASSET_NAMES = ("mug", "hook", "coaster")


def _usd(name: str) -> Path:
    """Path to one of this task's generated USDs."""
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    """Raise, naming the command that fixes it, if any USD is missing."""
    missing = [n for n in ASSET_NAMES if not _usd(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"mug_on_hook geometry missing from {ASSETS_DIR}: {', '.join(missing)}\n"
            f"    python benchmark/run/make_mug_on_hook_assets.py --headless")


# ----- objects ---------------------------------------------------------------
# The mug's mass, kg. Two flat gels meet a 38 mm cylinder along a line, so the
# hold is far weaker than the contact area suggests; this is light enough to
# carry without creeping and heavy enough that its shear, and the moment that
# shear goes, are well clear of the noise.
MUG_MASS = 0.045

# Where the placeholder sphere waits: below the ground plane.
PARKED_POS = (0.0, 0.0, -5.0)


# ----- layout ----------------------------------------------------------------
# The rail's board face, metres: far enough out on the right arm's side that
# the hanging point stays inside the arm's reachable y band and the near
# finger keeps clear of the board.
RAIL_XY = (0.265, -0.240)

# How far along the peg, from the board's face, the eye is threaded to, metres.
# The mug does not stay there: the rod falls, so it runs down to the stop ball
# and comes to rest against it. This is only where the hand lets go of it.
HANG_ALONG = 0.045


def rest_along() -> float:
    """How far out the mug comes to rest, metres: against the stop ball."""
    return peg_span() - STOP_BALL_R - HANDLE_SECTION / 2.0

# The coaster's centre, metres: well inside the right arm's top-grasp region
# and clear of the rail.
COASTER_XY = (0.245, -0.090)

# Metres, uniform +/- on x and y, per episode. The mug's own jitter is bounded
# by the coaster's dish, which clears it by 7 mm.
RAIL_JITTER = 0.005
COASTER_JITTER = 0.008
MUG_JITTER = 0.004
# The handle's heading varies a little about +x, degrees.
MUG_YAW_JITTER_DEG = 6.0

# The mug's base at spawn: standing on the coaster's floor.
MUG_SPAWN_Z = TABLE_TOP_Z + COASTER_FLOOR

# How far above the seated pose `set_mug_hung` starts the mug, metres.
PREVIEW_CLEARANCE = 0.001


def hang_pose(rail_xy=RAIL_XY, along: float | None = None) -> tuple[float, float, float]:
    """Where a mug hung `along` metres out on the peg has its base, metres.

    The seated pose: the eye's roof on the peg's top, the body on the near
    side of it. Defaults to where the mug comes to rest, against the ball.
    """
    along = rest_along() if along is None else along
    return (rail_xy[0] + hang_offset_x(),
            rail_xy[1] + along,
            TABLE_TOP_Z + peg_axis_z(along) + hang_offset_z())


def sample_layout(rng, rail_jitter: float = RAIL_JITTER,
                  coaster_jitter: float = COASTER_JITTER,
                  mug_jitter: float = MUG_JITTER,
                  yaw_jitter_deg: float = MUG_YAW_JITTER_DEG) -> dict:
    """Draw xy for the rail and the coaster, and the mug's pose in the dish.

    The mug's offset is capped to what the dish allows, so it always starts
    upright on the floor rather than riding up the inner wall.
    """
    out: dict = {}
    for key, (x, y), j in (("rail", RAIL_XY, rail_jitter),
                           ("coaster", COASTER_XY, coaster_jitter)):
        dx, dy = rng.uniform(-j, j, 2) if j else (0.0, 0.0)
        out[key] = (float(x + dx), float(y + dy))
    room = COASTER_INNER_R - MUG_OUTER_R
    dx, dy = rng.uniform(-mug_jitter, mug_jitter, 2) if mug_jitter else (0.0, 0.0)
    r = math.hypot(dx, dy)
    if r > room:
        dx, dy = dx * room / r, dy * room / r
    cx, cy = out["coaster"]
    out["mug"] = (float(cx + dx), float(cy + dy))
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
    """The mug, standing on the coaster."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd("mug")),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Depenetration well under the default metre per second: a
                # body this light leaves a deep contact at that speed.
                disable_gravity=False, max_depenetration_velocity=0.2,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=MUG_MASS),
            # Contact generated a millimetre and a half out, so the solver
            # answers the gels, the coaster and the peg before any is deep in;
            # well under the 3.5 mm the eye clears the peg by.
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0015, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(*COASTER_XY, MUG_SPAWN_Z)),
    )


def _spawn_fixture(prim: str, mesh: str, xy, z: float) -> RigidObjectCfg:
    """The rail or the coaster: kinematic, so `reset` can re-pose it."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(mesh)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*xy, z)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class MugOnHookSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + table + wall rail + coaster + mug.

    Set `robot.spawn.usd_path` to the grafted asset before constructing, as
    with the parent class. `set_mug_hung` spawns the mug already on the peg,
    which is how a preview shows the pose the task aims at.
    """

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.75, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.52, 0.42, 0.32)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.32, 0.0, TABLE_TOP_Z - 0.02)),
    )

    hook = _spawn_fixture("{ENV_REGEX_NS}/hook", "hook", RAIL_XY, TABLE_TOP_Z)
    coaster = _spawn_fixture("{ENV_REGEX_NS}/coaster", "coaster", COASTER_XY,
                             TABLE_TOP_Z)
    mug = _spawn_mug("{ENV_REGEX_NS}/mug")

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=3000.0, radius=0.05, color=(1.0, 0.98, 0.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.51, -0.02, 0.70)),
    )

    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=250.0, radius=0.12, color=(0.92, 0.95, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, -0.55, 0.55)),
    )

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        park_default_manipuland(self, PARKED_POS)
        self.manipuland.spawn.rigid_props.kinematic_enabled = True
        point_pads_at(self, PAD_TARGET)

    def set_mug_hung(self, along: float = HANG_ALONG) -> tuple[float, float, float]:
        """Spawn the mug on the peg instead of on the coaster, for a preview."""
        x, y, z = hang_pose(RAIL_XY, along)
        # A millimetre of air, so the settle resolves downward onto the peg
        # rather than out of an initial overlap.
        pos = (x, y, z + PREVIEW_CLEARANCE)
        self.mug.init_state.pos = pos
        return pos
