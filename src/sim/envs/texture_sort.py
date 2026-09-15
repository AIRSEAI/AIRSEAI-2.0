"""texture_sort scene: two pucks that differ only in surface, sorted by touch.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.texture_sort           + table + two pucks + two cups  (this file)
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.texture_sort import (
    CUP_HEIGHT,
    CUP_INNER_R,
    OBJECT_SPECS,
    ObjectSpec,
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
    "BIN_RADIUS",
    "BIN_WALL_HEIGHT",
    "BIN_XY",
    "DEFAULT_LIGHT_SCALE",
    "DOME_INTENSITY",
    "FIXTURES",
    "KEY_LIGHT_INTENSITY",
    "LAYOUT_MIN_SEP",
    "LAYOUT_X",
    "LAYOUT_Y",
    "LIGHT_SCALE_SWEEP",
    "OBJECT_MASS",
    "OBJECT_SPAWN_Z",
    "OBJECT_XY",
    "PAD_TARGET",
    "PARKED_POS",
    "PLACED",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "TextureSortSceneCfg",
    "ensure_assets",
    "sample_layout",
]


# ----- assets ----------------------------------------------------------------
# Generated USDs for this task. If they are missing, or stale after an
# edit to `sim.dims.texture_sort`, regenerate them with
#     python benchmark/run/make_texture_sort_assets.py
# (the installer does this for every task).
ASSETS_DIR = task_asset_dir("texture_sort")

ASSET_NAMES = ("puck_smooth", "puck_rough", "sort_cup")

def _usd(name: str) -> Path:
    """Path to one of this task's generated USDs."""
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    """Raise, naming the command that fixes it, if any USD is missing."""
    missing = [n for n in ASSET_NAMES if not _usd(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"texture_sort geometry missing from {ASSETS_DIR}: {', '.join(missing)}\n"
            f"    python benchmark/run/make_texture_sort_assets.py --headless")

# ----- objects ---------------------------------------------------------------
# Puck mass, kg. Low enough for the jaw preload the gel tolerates, high
# enough that the closing jaws do not tip it over.
OBJECT_MASS = 0.030

# The authored mesh has its origin at the base (z = 0 .. height), so a puck
# standing on the table has its root at the table top.
OBJECT_SPAWN_Z = TABLE_TOP_Z

# Where the class that is not drawn this episode waits, well under the floor.
PARKED_POS = (0.0, 0.0, -5.0)


# ----- layout ----------------------------------------------------------------
# Nominal positions, on the right arm's reachable line at x = 0.26.
# `sample_layout` redraws all three every episode; these are the fallback.
OBJECT_XY = (0.26, -0.155)

# Bin index meaning: 0 = smooth (blue cup), 1 = textured (orange cup).
BIN_XY = ((0.26, -0.075), (0.26, -0.245))

# The criterion counts a puck within BIN_RADIUS of a cup centre as in it.
BIN_RADIUS = CUP_INNER_R

BIN_WALL_HEIGHT = CUP_HEIGHT

# Per-episode layout: bounds on the centres of the puck and both cups, inside
# the right arm's reachable band, metres. The minimum separation keeps a cup
# wall outside the open jaws; cup order is drawn too.
LAYOUT_X = (0.21, 0.30)

LAYOUT_Y = (-0.255, -0.065)

LAYOUT_MIN_SEP = 0.100

FIXTURES = ("bin_0", "bin_1")

PLACED = ("object",) + FIXTURES

def sample_layout(rng, tries: int = 2000) -> dict[str, tuple[float, float]]:
    """Draw xy for the puck and both cups, rejecting draws closer than
    LAYOUT_MIN_SEP. Falls back to the nominal layout if `tries` is exhausted.
    """
    import numpy as np
    for _ in range(tries):
        pts = np.column_stack([rng.uniform(*LAYOUT_X, 3), rng.uniform(*LAYOUT_Y, 3)])
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        if d[np.triu_indices(3, 1)].min() >= LAYOUT_MIN_SEP:
            return {n: (float(p[0]), float(p[1])) for n, p in zip(PLACED, pts, strict=True)}
    return {"object": OBJECT_XY, "bin_0": BIN_XY[0], "bin_1": BIN_XY[1]}


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw.
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
# Arm pose at the start of every episode: the robot's own HOME, jaws shut,
# so a demonstration's first frames match a rollout's.
START_JOINT_POS = REAL_HOME_JOINT_POS
# The prim every tactile pad senses at scene build; the expert re-points the
# pads at the drawn puck each episode.
PAD_TARGET = "{ENV_REGEX_NS}/puck_smooth"


# ----- lighting --------------------------------------------------------------
# Light level. `set_light_scale` multiplies both emitters; the sweep is the
# task's visual-dose axis.
DOME_INTENSITY = 3000.0

KEY_LIGHT_INTENSITY = 3000.0

DEFAULT_LIGHT_SCALE = 0.15

LIGHT_SCALE_SWEEP = (1.0, 0.4, 0.15, 0.05)


# ----- spawn helpers ---------------------------------------------------------
def _spawn_puck(prim: str, spec: ObjectSpec, pos) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(spec.name)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=spec.colour, roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _spawn_cup(prim: str, name: str, pos, colour) -> RigidObjectCfg:
    """A cup, kinematic so `reset` can re-pose it each episode."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(name)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class TextureSortSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + table + two pucks + two cups.

    Set `robot.spawn.usd_path` to the grafted asset before constructing, as
    with the parent class. Which puck is the target, and where it and the cups
    stand, is drawn by the expert at reset.
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

    # bin_0 blue (smooth), bin_1 orange (textured). The cups are told apart
    # by colour; the pucks are not, which is the point of the task.
    bin_0 = _spawn_cup("{ENV_REGEX_NS}/bin_0", "sort_cup",
                    (BIN_XY[0][0], BIN_XY[0][1], TABLE_TOP_Z),
                    (0.16, 0.35, 0.70))
    bin_1 = _spawn_cup("{ENV_REGEX_NS}/bin_1", "sort_cup",
                    (BIN_XY[1][0], BIN_XY[1][1], TABLE_TOP_Z),
                    (0.85, 0.45, 0.12))

    # Both classes are spawned and the reset parks whichever is not drawn at
    # PARKED_POS. Both must exist from the start: TacSL resolves its sensed
    # prim at initialisation, before the draw.
    puck_smooth = _spawn_puck("{ENV_REGEX_NS}/puck_smooth", OBJECT_SPECS["smooth"],
                          (OBJECT_XY[0], OBJECT_XY[1], OBJECT_SPAWN_Z))
    puck_rough = _spawn_puck("{ENV_REGEX_NS}/puck_rough", OBJECT_SPECS["rough"],
                         PARKED_POS)

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=KEY_LIGHT_INTENSITY, radius=0.05,
            color=(1.0, 0.98, 0.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.51, -0.16, 0.70)),
    )

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        park_default_manipuland(self, PARKED_POS)
        point_pads_at(self, PAD_TARGET)
        self.set_light_scale(DEFAULT_LIGHT_SCALE)

    def set_light_scale(self, scale: float = DEFAULT_LIGHT_SCALE) -> float:
        """Multiply both emitters by `scale`. Returns the scale applied."""
        if scale <= 0.0:
            raise ValueError(f"light_scale must be positive, got {scale}")
        self.dome_light.spawn.intensity = DOME_INTENSITY * scale
        self.key_light.spawn.intensity = KEY_LIGHT_INTENSITY * scale
        return scale
