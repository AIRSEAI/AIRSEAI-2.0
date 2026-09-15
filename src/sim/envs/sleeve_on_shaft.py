"""sleeve_on_shaft scene: a split sleeve to be slid onto a fixed shaft.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.sleeve_on_shaft        + table + shaft + two sleeve materials   (this file)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.sleeve_on_shaft import (
    BASE_HEIGHT,
    CLEARANCE_SWEEP,
    DEFAULT_CLEARANCE,
    SHAFT_HEIGHT,
    SHELL_MESH_NAMES,
    SLEEVE_COLOR,
    SLEEVE_HEIGHT,
    SLEEVE_OUTER_R,
    SLEEVE_WALL,
    SPLIT_GAP,
    shaft_mesh_name,
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
    "MATERIALS",
    "PAD_TARGET",
    "PARKED_POS",
    "PICK_XY",
    "SHAFT_TOP_Z",
    "SHAFT_XY",
    "SLEEVE_SPAWN_Z",
    "SPAWN_JITTER",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "SleeveMaterial",
    "SleeveOnShaftSceneCfg",
    "ensure_assets",
    "material_prims",
    "ring_stiffness",
]


# ----- assets ----------------------------------------------------------------
# Generated USDs for this task. If they are missing, or stale after an
# edit to `sim.dims.sleeve_on_shaft`, regenerate them with
#     python benchmark/run/make_sleeve_on_shaft_assets.py
# (the installer does this for every task).
ASSETS_DIR = task_asset_dir("sleeve_on_shaft")

ASSET_NAMES = SHELL_MESH_NAMES + tuple(shaft_mesh_name(c) for c in CLEARANCE_SWEEP)


def _usd(name: str) -> Path:
    """Path to one of this task's generated USDs."""
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    """Raise, naming the command that fixes it, if any USD is missing."""
    missing = [n for n in ASSET_NAMES if not _usd(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"sleeve_on_shaft geometry missing from {ASSETS_DIR}: {', '.join(missing)}\n"
            f"    python benchmark/run/make_sleeve_on_shaft_assets.py --headless")


# ----- objects ---------------------------------------------------------------
def ring_stiffness(youngs_modulus: float) -> float:
    """Diametral stiffness of the sleeve under two opposed line loads, N/m.

    Thin ring, two diametrically opposed loads F: the diameter along the load
    shortens by delta = 0.149 F R^3 / (E I), with R the mean radius and
    I = L t^3 / 12 for wall t and length L. The shell joint's spring takes
    k = F / delta.
    """
    r_mean = SLEEVE_OUTER_R - SLEEVE_WALL / 2
    inertia = SLEEVE_HEIGHT * SLEEVE_WALL ** 3 / 12.0
    return youngs_modulus * inertia / (0.149 * r_mean ** 3)


@dataclass(frozen=True)
class SleeveMaterial:
    """One hidden sleeve material. Nothing here is visible."""

    name: str
    youngs_modulus: float   # Pa
    half_mass: float        # kg per shell, so the sleeve weighs twice this
    damping: float          # N s/m on the shell joint's linear drive

    @property
    def stiffness(self) -> float:
        """Shell joint spring, N/m, from the ring formula."""
        return ring_stiffness(self.youngs_modulus)


# The two hidden materials. The soft sleeve shuts its bore onto the shaft
# under a squeeze the stiff one needs to be carried, and the stiff one is too
# heavy for the grip that keeps the soft one open.
MATERIALS = (
    SleeveMaterial("soft", youngs_modulus=26.2e6, half_mass=0.010, damping=10.0),
    SleeveMaterial("stiff", youngs_modulus=438e6, half_mass=0.100, damping=60.0),
)

# Where the material that was not drawn waits: below the ground plane.
PARKED_POS = (0.0, 0.0, -5.0)


def material_prims(name: str) -> tuple[str, str]:
    """Scene attribute names of a material's two shells: (`<name>_a`, `<name>_b`)."""
    return (f"{name}_a", f"{name}_b")


# ----- layout ----------------------------------------------------------------
# Where the sleeve stands, and where the shaft stands, metres.
PICK_XY = (0.26, -0.15)

SHAFT_XY = (0.26, -0.02)

SPAWN_JITTER = 0.010           # metres, uniform +/- on x and y of both, per episode

# Sleeve centre height when standing on the table.
SLEEVE_SPAWN_Z = TABLE_TOP_Z + SLEEVE_HEIGHT / 2 + 0.002

# The shaft mesh has its base bottom at its origin; it stands on the table.
SHAFT_TOP_Z = TABLE_TOP_Z + BASE_HEIGHT + SHAFT_HEIGHT

_HALF_GAP = SPLIT_GAP / 2


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw: shut, as the
# robot's HOME leaves them.
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
# Arm pose at the start of every episode: the robot's own HOME, so a
# demonstration's first frames match a rollout's.
START_JOINT_POS = REAL_HOME_JOINT_POS
# The prim each tactile pad senses at scene build: each right pad faces the
# shell on its side (`_a` is +y, `_b` is -y). The expert re-points them at
# the drawn material each episode.
PAD_TARGET = {
    "right_left_finger_tactile": "{ENV_REGEX_NS}/soft_a",
    "right_right_finger_tactile": "{ENV_REGEX_NS}/soft_b",
    "left_left_finger_tactile": "{ENV_REGEX_NS}/soft_a",
    "left_right_finger_tactile": "{ENV_REGEX_NS}/soft_a",
}


# ----- spawn helpers ---------------------------------------------------------
def _spawn_shell(prim: str, mesh: str, material: SleeveMaterial, pos) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(mesh)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=material.half_mass),
            # Contact offset well under SPLIT_GAP and the shaft clearance, so
            # neither the other shell nor a centred shaft generates contacts.
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0004, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=SLEEVE_COLOR, roughness=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _spawn_shaft(clearance: float) -> RigidObjectCfg:
    """The shaft on its base, kinematic so `reset` can re-pose it each episode."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/shaft",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(shaft_mesh_name(clearance))),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0004, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(SHAFT_XY[0], SHAFT_XY[1], TABLE_TOP_Z)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class SleeveOnShaftSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + table + shaft + both sleeve materials, four shells in all.

    Set `robot.spawn.usd_path` to the grafted asset before constructing, as
    with the parent class. The expert draws a material at reset, moves its two
    shells onto the table, parks the other, re-points the pads, and authors
    the shell joint with that material's stiffness.
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

    shaft = _spawn_shaft(DEFAULT_CLEARANCE)

    # `_a` is the +y shell and `_b` the -y one, spawned SPLIT_GAP apart.
    soft_a = _spawn_shell("{ENV_REGEX_NS}/soft_a", SHELL_MESH_NAMES[0], MATERIALS[0],
                          (PICK_XY[0], PICK_XY[1] + _HALF_GAP, SLEEVE_SPAWN_Z))
    soft_b = _spawn_shell("{ENV_REGEX_NS}/soft_b", SHELL_MESH_NAMES[1], MATERIALS[0],
                          (PICK_XY[0], PICK_XY[1] - _HALF_GAP, SLEEVE_SPAWN_Z))
    stiff_a = _spawn_shell("{ENV_REGEX_NS}/stiff_a", SHELL_MESH_NAMES[0], MATERIALS[1], PARKED_POS)
    stiff_b = _spawn_shell("{ENV_REGEX_NS}/stiff_b", SHELL_MESH_NAMES[1], MATERIALS[1], PARKED_POS)

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        park_default_manipuland(self, PARKED_POS)
        point_pads_at(self, PAD_TARGET)

    def set_material_mass(self, name: str, half_mass: float) -> float:
        """Set both shells of material `name` to `half_mass`, kg each."""
        for prim in material_prims(name):
            getattr(self, prim).spawn.mass_props.mass = float(half_mass)
        return float(half_mass)

    def set_clearance(self, clearance: float) -> float:
        """Load the shaft mesh for a radial clearance in CLEARANCE_SWEEP, metres."""
        if clearance not in CLEARANCE_SWEEP:
            raise ValueError(f"clearance {clearance} not generated; one of {CLEARANCE_SWEEP}")
        self.shaft.spawn.usd_path = str(_usd(shaft_mesh_name(clearance)))
        return clearance
