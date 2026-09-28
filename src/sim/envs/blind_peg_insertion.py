"""blind_peg_insertion scene: a peg and a board whose hole clearance is a knob.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.blind_peg_insertion    + table + board + peg         (this file)
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.blind_peg_insertion import (
    BOARD_THICKNESS,
    BOARD_X,
    BOARD_Y,
    CLEARANCE_SWEEP,
    DEFAULT_CLEARANCE,
    PEG_DIAMETER,
    PEG_LENGTH,
    hole_diameter,
)
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
    "BOARD_TOP_Z",
    "DEFAULT_LIGHT_SCALE",
    "DOME_INTENSITY",
    "HOLE_JITTER",
    "HOLE_XY",
    "KEY_LIGHT_INTENSITY",
    "LIGHT_SCALE_SWEEP",
    "PAD_TARGET",
    "PEG_SPAWN_XY",
    "SPAWN_JITTER",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "BlindPegInsertionSceneCfg",
    "ensure_assets",
]


# ----- assets ----------------------------------------------------------------
# Generated USDs for this task. If they are missing, or stale after an
# edit to `sim.dims.blind_peg_insertion`, regenerate them with
#     python benchmark/run/make_blind_peg_insertion_assets.py
# (the installer does this for every task).
ASSETS_DIR = task_asset_dir("blind_peg_insertion")

# One board per clearance, named by hole diameter.
ASSET_NAMES = tuple(f"board_hole_{hole_diameter(c) * 1e3:.2f}mm" for c in CLEARANCE_SWEEP)


def _usd(name: str) -> Path:
    """Path to one of this task's generated USDs."""
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    """Raise, naming the command that fixes it, if any USD is missing."""
    missing = [n for n in ASSET_NAMES if not _usd(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"blind_peg_insertion geometry missing from {ASSETS_DIR}: {', '.join(missing)}\n"
            f"    python benchmark/run/make_blind_peg_insertion_assets.py --headless"
        )


# ----- layout ----------------------------------------------------------------
BOARD_TOP_Z = TABLE_TOP_Z + BOARD_THICKNESS

# Nominal episode geometry. The peg spawns on one side of the midline and
# that arm does the task.
HOLE_XY = (0.26, 0.0)

PEG_SPAWN_XY = {"right": (0.26, -0.18), "left": (0.26, 0.18)}

SPAWN_JITTER = 0.01  # metres, uniform +/- on x and y

# How far the board (and so the hole's true pose) is redrawn each episode,
# metres. Tighter than the peg's jitter: the hole sits near the edge of both
# arms' reachable set.
HOLE_JITTER = 0.008


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw.
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
# Arm pose at the start of every episode: the robot's own HOME, jaws shut;
# the expert's `home` phase opens them.
START_JOINT_POS = REAL_HOME_JOINT_POS
# The prim every tactile pad senses. The board is felt through the peg.
PAD_TARGET = "{ENV_REGEX_NS}/peg"


# ----- lighting --------------------------------------------------------------
# Light level. `set_light_scale` multiplies both emitters together, so no
# specular highlight survives on the board rim; the sweep is the
# dim-light axis.
DOME_INTENSITY = 3000.0

KEY_LIGHT_INTENSITY = 3000.0

DEFAULT_LIGHT_SCALE = 1.0

LIGHT_SCALE_SWEEP = (1.0, 0.4, 0.15, 0.05)


# ----- spawn helpers ---------------------------------------------------------
def _spawn_peg(prim: str, xy: tuple[float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim,
        # Mesh, not a primitive: TacSL needs an SDF collision mesh on
        # anything a sensor tracks, and the peg is the only sensed object.
        spawn=sim_utils.MeshCylinderCfg(
            radius=PEG_DIAMETER / 2,
            height=PEG_LENGTH,
            # CCD is applied after spawn by `sim.tactile.sdf.apply_ccd`;
            # RigidBodyPropertiesCfg has no field for it here.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
            # Speculative-contact margin, metres, so a falling peg does not
            # tunnel. Detection range only; it does not change the grasp.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.010, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.1, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.72, 0.25), metallic=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], TABLE_TOP_Z + PEG_LENGTH / 2 + 0.002)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class BlindPegInsertionSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + table + hole board + peg.

    Set `robot.spawn.usd_path` to the grafted asset before constructing, as
    with the parent class. The hole clearance is applied after construction,
    since one scene is built per swept value::

        cfg = BlindPegInsertionSceneCfg(num_envs=1, env_spacing=2.0)
        cfg.set_clearance(0.002)
    """

    manipuland = None

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.75, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.66, 0.55)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.0, TABLE_TOP_Z - 0.02)),
    )

    # Kinematic rather than a static collider, so `reset` can re-place it
    # each episode while it keeps infinite mass and ignores gravity. A fixed
    # board would pin the hole's true xy and let a policy skip the search.
    board = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/board",
        spawn=sim_utils.CuboidCfg(  # placeholder, replaced by set_clearance
            size=(BOARD_X, BOARD_Y, BOARD_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.38, 0.42), metallic=0.6),
        ),
        # Nominal pose; `reset` draws the episode's within HOLE_JITTER of it.
        # Which arm inserts is decided by which side the peg spawns on.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(HOLE_XY[0], HOLE_XY[1], TABLE_TOP_Z + BOARD_THICKNESS / 2)
        ),
    )

    # Nominal pose; `reset` re-poses it, and may put it on the left side.
    peg = _spawn_peg("{ENV_REGEX_NS}/peg", PEG_SPAWN_XY["right"])

    # A sphere, not a distant light: a distant light has no falloff, so
    # dimming it leaves the board rim the brightest thing in frame.
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(intensity=KEY_LIGHT_INTENSITY, radius=0.05, color=(1.0, 0.98, 0.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.51, -0.02, 0.70)),
    )

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)

    def set_clearance(self, clearance: float = DEFAULT_CLEARANCE) -> float:
        """Point the board at the USD variant for this radial clearance.

        Returns the resulting hole diameter, metres.
        """
        if clearance not in CLEARANCE_SWEEP:
            raise ValueError(f"clearance {clearance} has no authored board; available: {CLEARANCE_SWEEP}")
        d_mm = round(hole_diameter(clearance) * 1000, 2)
        usd = _usd(f"board_hole_{d_mm:.2f}mm")
        # `UsdFileCfg` has no `physics_material` field; the board uses the
        # default material.
        self.board.spawn = sim_utils.UsdFileCfg(
            usd_path=str(usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        return hole_diameter(clearance)

    def set_light_scale(self, scale: float = DEFAULT_LIGHT_SCALE) -> float:
        """Multiply both emitters by `scale`, which is the dim-light
        knob. Any positive value is accepted, not only the sweep points.
        Returns the scale applied, for the caller to record.
        """
        if scale <= 0.0:
            raise ValueError(f"light_scale must be positive, got {scale}")
        self.dome_light.spawn.intensity = DOME_INTENSITY * scale
        self.key_light.spawn.intensity = KEY_LIGHT_INTENSITY * scale
        return scale


# -- preview -----------------------------------------------------------------

# Pulled back to show the arms relative to the objects.
PREVIEW = PreviewSpec(sdf_prims=("peg",), tri_prims=("board",), work_target=(0.26, 0.0, TABLE_TOP_Z + 0.06))


def make_preview_cfg() -> BlindPegInsertionSceneCfg:
    """Build the scene for preview."""
    cfg = BlindPegInsertionSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.set_clearance()  # anchor board, 1 mm clearance
    cfg.set_light_scale(1.0)
    return cfg
