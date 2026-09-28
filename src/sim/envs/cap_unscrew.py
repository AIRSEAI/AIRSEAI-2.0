"""cap_unscrew scene: a capped bottle standing free to spin at the centreline,
and a tray beside it for the cap.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.cap_unscrew            + table + bottle + cap + tray       (this file)
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.cap_unscrew import (
    BOTTLE_H,
    CAP_BORE_DEPTH,
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
    "BOTTLE_JITTER",
    "BOTTLE_LIFT",
    "BOTTLE_MASS",
    "BOTTLE_SPAWN_Z",
    "BOTTLE_SPIN_DAMPING",
    "BOTTLE_XY",
    "CAP_LIFT",
    "CAP_MASS",
    "CAP_SPAWN_Z",
    "PAD_TARGET",
    "SPINDLE_Z",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "THREAD_ANGLE_SWEEP",
    "THREAD_DAMPING",
    "TRAY_JITTER",
    "TRAY_XY",
    "CapUnscrewSceneCfg",
    "ensure_assets",
    "sample_layout",
]


# ----- assets ----------------------------------------------------------------
# Generated USDs for this task. If they are missing, or stale after an
# edit to `sim.dims.cap_unscrew`, regenerate them with
#     python benchmark/run/make_cap_unscrew_assets.py
# (the installer does this for every task).
ASSETS_DIR = task_asset_dir("cap_unscrew")

ASSET_NAMES = ("bottle", "cap", "tray")


def _usd(name: str) -> Path:
    """Path to one of this task's generated USDs."""
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    """Raise, naming the command that fixes it, if any USD is missing."""
    missing = [n for n in ASSET_NAMES if not _usd(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"cap_unscrew geometry missing from {ASSETS_DIR}: {', '.join(missing)}\n"
            f"    python benchmark/run/make_cap_unscrew_assets.py --headless"
        )


# ----- objects ---------------------------------------------------------------
# Masses, kg: the bottle heavy enough to be carried round by a cap turned
# against it, the cap heavy enough not to be moved by a contact impulse.
BOTTLE_MASS = 0.150
CAP_MASS = 0.050

# Height the spin joint holds the bottle at above the table, metres: clear of
# it, so no contact friction resists the spin.
BOTTLE_LIFT = 0.0005

# Damping of the bottle's spin, N m s/rad: above solver noise, well under the
# thread's, so an unheld bottle turns with its cap.
BOTTLE_SPIN_DAMPING = 0.02

# The thread: a joint freeing only the cap's turn about the bottle's axis,
# damped (N m s/rad) and released once the turn reaches the thread angle.
# The angle, degrees, is the task's knob and is not visible.
THREAD_DAMPING = 0.1  # N m s/rad
THREAD_ANGLE_SWEEP = (90.0, 120.0, 150.0)

# Height the thread holds the cap above resting on the neck, metres: over the
# contact offset, so the pair carries no force until the cap is freed.
CAP_LIFT = 0.001


# ----- layout ----------------------------------------------------------------
# Bottle axis, metres: on the centreline, the one place both arms reach at
# the height of the bottle's body and cap.
BOTTLE_XY = (0.22, 0.0)

# Tray centre, metres: on the right arm's side, inside its reach band.
TRAY_XY = (0.24, -0.17)

# Metres, uniform +/- on x and y, per episode.
BOTTLE_JITTER = 0.005
TRAY_JITTER = 0.010


def sample_layout(
    rng, bottle_jitter: float = BOTTLE_JITTER, tray_jitter: float = TRAY_JITTER
) -> dict[str, tuple[float, float]]:
    """Draw xy for the bottle and the tray, each jittered about its stand."""
    out = {}
    for key, (x, y), j in (("bottle", BOTTLE_XY, bottle_jitter), ("tray", TRAY_XY, tray_jitter)):
        dx, dy = rng.uniform(-j, j, 2) if j else (0.0, 0.0)
        out[key] = (float(x + dx), float(y + dy))
    return out


# Spawn heights: the bottle's base, the cap's skirt, and the spindle, which
# sits inside the table under the bottle's axis, out of sight.
BOTTLE_SPAWN_Z = TABLE_TOP_Z + BOTTLE_LIFT
CAP_SPAWN_Z = BOTTLE_SPAWN_Z + BOTTLE_H - CAP_BORE_DEPTH + CAP_LIFT
SPINDLE_Z = TABLE_TOP_Z - 0.010


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw: shut, as the
# robot's HOME leaves them.
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
# Arm pose at the start of every episode: the robot's own HOME.
START_JOINT_POS = REAL_HOME_JOINT_POS
# The prim each tactile pad senses: the right hand turns the cap, the left
# holds the bottle.
PAD_TARGET = {
    "right_left_finger_tactile": "{ENV_REGEX_NS}/cap",
    "right_right_finger_tactile": "{ENV_REGEX_NS}/cap",
    "left_left_finger_tactile": "{ENV_REGEX_NS}/bottle",
    "left_right_finger_tactile": "{ENV_REGEX_NS}/bottle",
}


# ----- spawn helpers ---------------------------------------------------------
def _spawn_part(prim: str, mesh: str, mass: float, z: float) -> RigidObjectCfg:
    """The bottle or the cap, standing on the bottle's axis at height `z`."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(mesh)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Depenetration well under the default metre per second,
                # which throws a part this light.
                disable_gravity=False,
                max_depenetration_velocity=0.2,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            # Contact generated before the gels or the tray are deep in,
            # and under the bore's clearance on the neck.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*BOTTLE_XY, z)),
    )


def _spawn_spindle(prim: str) -> RigidObjectCfg:
    """The bottle's anchor: a small kinematic body with no collider, so
    `reset` can move the bottle's spin axis with the bottle."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.004,
            height=0.004,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*BOTTLE_XY, SPINDLE_Z)),
    )


def _spawn_tray(prim: str) -> RigidObjectCfg:
    """The tray, kinematic so `reset` can re-pose it each episode."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd("tray")),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*TRAY_XY, TABLE_TOP_Z)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class CapUnscrewSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + table + bottle + cap + tray, and the bottle's hidden anchor.

    The spin and thread joints are authored by the task before the scene
    starts, not here; without them the bottle simply stands on the table with
    the cap on. Set `robot.spawn.usd_path` to the grafted asset before
    constructing, as with the parent class.
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

    spindle = _spawn_spindle("{ENV_REGEX_NS}/spindle")
    tray = _spawn_tray("{ENV_REGEX_NS}/tray")

    bottle = _spawn_part("{ENV_REGEX_NS}/bottle", "bottle", BOTTLE_MASS, BOTTLE_SPAWN_Z)
    cap = _spawn_part("{ENV_REGEX_NS}/cap", "cap", CAP_MASS, CAP_SPAWN_Z)

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(intensity=3000.0, radius=0.05, color=(1.0, 0.98, 0.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.51, -0.02, 0.70)),
    )

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)


PREVIEW = PreviewSpec(
    sdf_prims=("bottle", "cap"),
    tri_prims=("tray",),
    work_target=(0.22, 0.0, TABLE_TOP_Z + 0.08),
    sdf_resolution=128,
)


def make_preview_cfg() -> CapUnscrewSceneCfg:
    """Build the task scene for preview."""
    return CapUnscrewSceneCfg(num_envs=1, env_spacing=3.0)
