"""handover scene: a spring-loaded split sleeve standing compressed in a cup
on one side, an empty cup with a key standing in it on the other.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.handover               + table + two cups + the sleeve      (this file)
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.handover import (
    CUP_FLOOR,
    CUP_HEIGHT,
    DEFAULT_TARGET_BORE,
    SHELL_MESH_NAMES,
    SLEEVE_COLOR,
    SLEEVE_LENGTH,
    SPAWN_GAP,
    TARGET_BORE_SWEEP,
    target_mesh_name,
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
    "CUP_FLOOR_Z",
    "CUP_JITTER",
    "CUP_RIM_Z",
    "CUP_XY",
    "PAD_TARGET",
    "SHELL_MASS",
    "SHELL_PRIMS",
    "SHELL_QUAT",
    "SLEEVE_DAMPING",
    "SLEEVE_SPAWN_Z",
    "SLEEVE_STIFFNESS",
    "SOURCE",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "TARGET",
    "HandoverSceneCfg",
    "ensure_assets",
    "sample_layout",
]


# ----- assets ----------------------------------------------------------------
# Generated USDs for this task. If they are missing, or stale after an
# edit to `sim.dims.handover`, regenerate them with
#     python benchmark/run/make_handover_assets.py
# (the installer does this for every task).
ASSETS_DIR = task_asset_dir("handover")

ASSET_NAMES = SHELL_MESH_NAMES + ("cup_source",) + tuple(target_mesh_name(d) for d in TARGET_BORE_SWEEP)


def _usd(name: str) -> Path:
    """Path to one of this task's generated USDs."""
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    """Raise, naming the command that fixes it, if any USD is missing."""
    missing = [n for n in ASSET_NAMES if not _usd(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"handover geometry missing from {ASSETS_DIR}: {', '.join(missing)}\n"
            f"    python benchmark/run/make_handover_assets.py --headless"
        )


# ----- objects ---------------------------------------------------------------
# Scene attribute names of the two shells. The shells are authored split
# across x and stand turned a quarter turn about z, so the split runs across
# y and `_a` is the +y half.
SHELL_PRIMS = ("sleeve_a", "sleeve_b")

# The shells' standing orientation, (w, x, y, z): a quarter turn about z.
SHELL_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)

# Mass of each shell, kg; the sleeve weighs twice this.
SHELL_MASS = 0.010

# The spring between the shells, N/m and N s/m: a few newtons at the source
# gap, inside what a held grasp resists.
SLEEVE_STIFFNESS = 600.0
SLEEVE_DAMPING = 10.0


# The sleeve's centre when it stands on a cup's floor.
SLEEVE_SPAWN_Z = TABLE_TOP_Z + CUP_FLOOR + SLEEVE_LENGTH / 2 + 0.001


# ----- layout ----------------------------------------------------------------
# The two cups' roles: the sleeve starts in the source and ends in the target.
SOURCE, TARGET = "source", "target"

# Cup centres, metres: the source on the left arm's side, the target on the
# right's, each near its own arm's outer reach.
CUP_XY = {SOURCE: (0.18, 0.16), TARGET: (0.18, -0.16)}

# Metres, uniform +/- on each cup's xy, per episode.
CUP_JITTER = 0.003


def sample_layout(rng, jitter: float = CUP_JITTER) -> dict[str, tuple[float, float]]:
    """Draw xy for both cups, each jittered independently about its stand."""
    out = {}
    for key in (SOURCE, TARGET):
        x, y = CUP_XY[key]
        dx, dy = rng.uniform(-jitter, jitter, 2) if jitter else (0.0, 0.0)
        out[key] = (float(x + dx), float(y + dy))
    return out


# Rim height of each cup.
CUP_RIM_Z = {role: TABLE_TOP_Z + h for role, h in CUP_HEIGHT.items()}

# Height of a cup's floor top, where a seated sleeve's lower rim rests.
CUP_FLOOR_Z = TABLE_TOP_Z + CUP_FLOOR


# ----- start state -----------------------------------------------------------
# Jaw opening at the start of every episode, metres per jaw: shut, as the
# robot's HOME leaves them.
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
# Arm pose at the start of every episode: the robot's own HOME.
START_JOINT_POS = REAL_HOME_JOINT_POS
# The prim each tactile pad senses at scene build: the right pads one shell
# each, the left pads both the same shell, since the tactile views fault when
# both arms' pads are split across the two shells at build. The expert
# re-points each arm's pads once its wrist is placed.
PAD_TARGET = {
    "right_left_finger_tactile": "{ENV_REGEX_NS}/sleeve_a",
    "right_right_finger_tactile": "{ENV_REGEX_NS}/sleeve_b",
    "left_left_finger_tactile": "{ENV_REGEX_NS}/sleeve_a",
    "left_right_finger_tactile": "{ENV_REGEX_NS}/sleeve_a",
}


# ----- spawn helpers ---------------------------------------------------------
def _spawn_shell(prim: str, mesh: str, y_off: float) -> RigidObjectCfg:
    """One half-shell, standing in the source cup `y_off` from its axis."""
    xy = CUP_XY[SOURCE]
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(mesh)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SHELL_MASS),
            # Contact offset well under the gap and the bore clearance, so
            # neither the other shell nor the cup generates contacts before
            # touching.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=SLEEVE_COLOR, roughness=0.4, metallic=0.3
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1] + y_off, SLEEVE_SPAWN_Z), rot=SHELL_QUAT),
    )


def _spawn_cup(prim: str, key: str, mesh: str) -> RigidObjectCfg:
    """A cup, kinematic so `reset` can re-pose it each episode."""
    xy = CUP_XY[key]
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd(mesh)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], TABLE_TOP_Z)),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class HandoverSceneCfg(BimanualTactileSceneCfg):
    """Minimal unit + table + two cups + the sleeve's two shells.

    Set `robot.spawn.usd_path` to the grafted asset before constructing, as
    with the parent class.
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

    cup_source = _spawn_cup("{ENV_REGEX_NS}/cup_source", SOURCE, "cup_source")
    cup_target = _spawn_cup("{ENV_REGEX_NS}/cup_target", TARGET, target_mesh_name(DEFAULT_TARGET_BORE))

    # `_a` is the +y shell and `_b` the -y one, spawned SPAWN_GAP apart.
    sleeve_a = _spawn_shell("{ENV_REGEX_NS}/sleeve_a", SHELL_MESH_NAMES[0], SPAWN_GAP / 2)
    sleeve_b = _spawn_shell("{ENV_REGEX_NS}/sleeve_b", SHELL_MESH_NAMES[1], -SPAWN_GAP / 2)

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(intensity=3000.0, radius=0.05, color=(1.0, 0.98, 0.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.51, -0.02, 0.70)),
    )

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)

    def set_target_bore(self, bore: float) -> float:
        """Load the target cup mesh for a bore in TARGET_BORE_SWEEP, metres."""
        bore = round(float(bore), 6)
        if bore not in TARGET_BORE_SWEEP:
            raise ValueError(f"target bore {bore} not generated; one of {TARGET_BORE_SWEEP}")
        self.cup_target.spawn.usd_path = str(_usd(target_mesh_name(bore)))
        return bore


# -- preview -----------------------------------------------------------------

# Framed to show both cups and the sleeve standing in the source one.
PREVIEW = PreviewSpec(
    sdf_prims=("sleeve_a", "sleeve_b"),
    tri_prims=("cup_source", "cup_target"),
    work_target=(0.22, 0.0, TABLE_TOP_Z + 0.06),
    sdf_resolution=128,
)


def make_preview_cfg() -> HandoverSceneCfg:
    """Build the scene for preview."""
    return HandoverSceneCfg(num_envs=1, env_spacing=3.0)
