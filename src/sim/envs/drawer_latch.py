"""The standard workstation with one spring-return tool drawer."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.drawer_latch import HANDLE_CENTER_Z, YAW_CENTER_DEG
from sim.envs.bimanual_tactile import REAL_HOME_JOINT_POS, TABLE_TOP_Z, point_pads_at, start_joint_pos
from sim.envs.preview import PreviewSpec
from sim.envs.stack_cube import StackCubeSceneCfg

__all__ = [
    "ASSETS_DIR",
    "CASE_XY",
    "DRAWER_FACE_TURN_DEG",
    "DRAWER_QUAT",
    "DRAWER_YAW_DEG",
    "DRAWER_Z",
    "PAD_TARGET",
    "PREVIEW",
    "SLIDE_AXIS_XY",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "DrawerLatchSceneCfg",
    "ensure_assets",
    "make_preview_cfg",
]

ASSETS_DIR = task_asset_dir("drawer_latch")
CASE_XY = (0.46, 0.0)
DRAWER_FACE_TURN_DEG = YAW_CENTER_DEG
DRAWER_YAW_DEG = DRAWER_FACE_TURN_DEG
DRAWER_QUAT = (math.cos(math.radians(DRAWER_YAW_DEG) / 2), 0.0, 0.0,
               math.sin(math.radians(DRAWER_YAW_DEG) / 2))
SLIDE_AXIS_XY = (math.cos(math.radians(DRAWER_YAW_DEG)), math.sin(math.radians(DRAWER_YAW_DEG)))
DRAWER_Z = TABLE_TOP_Z
START_JOINT_POS = REAL_HOME_JOINT_POS
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
PAD_TARGET = "{ENV_REGEX_NS}/drawer"


def ensure_assets() -> None:
    missing = [name for name in ("case", "drawer") if not (ASSETS_DIR / f"{name}.usd").exists()]
    if missing:
        raise FileNotFoundError(
            f"drawer_latch geometry missing: {', '.join(missing)}; "
            "run python benchmark/run/make_drawer_latch_assets.py --headless"
        )


def _case(prim: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "case.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CASE_XY, DRAWER_Z), rot=DRAWER_QUAT),
    )


def _drawer(prim: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "drawer.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=0.2,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0015),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CASE_XY, DRAWER_Z), rot=DRAWER_QUAT),
    )


@configclass
class DrawerLatchSceneCfg(StackCubeSceneCfg):
    """Keep the existing table, ground and lighting; replace cubes with a drawer."""

    cube_a = None
    cube_b = None
    case = _case("{ENV_REGEX_NS}/case")
    drawer = _drawer("{ENV_REGEX_NS}/drawer")

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        point_pads_at(self, PAD_TARGET)


PREVIEW = PreviewSpec(
    sdf_prims=("drawer",),
    work_target=(CASE_XY[0], CASE_XY[1], TABLE_TOP_Z + HANDLE_CENTER_Z),
    eye_dir=(-0.35, -0.86, 0.36),
)


def make_preview_cfg() -> DrawerLatchSceneCfg:
    return DrawerLatchSceneCfg(num_envs=1, env_spacing=3.0)
