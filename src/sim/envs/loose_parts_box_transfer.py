"""Open parts box, loose contents and a shelf on the shared tabletop scene."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.loose_parts_box_transfer import (
    BALL_COLOR,
    BALL_MASS,
    BALL_RADIUS,
    BOX_COLOR,
    BOX_LENGTH,
    BOX_MASS,
    BOX_WALL,
    LARGE_BALL_COLOR,
    LARGE_BALL_RADIUS,
    MAT_COLOR,
    MAT_HEIGHT,
    MAT_LENGTH,
    MAT_WIDTH,
    NUT_COLOR,
    NUT_MASS,
    SHELF_HEIGHT,
)
from sim.envs.bimanual_tactile import (
    REAL_HOME_JOINT_POS,
    TABLE_TOP_Z,
    TACTILE_SENSORS,
    BimanualTactileSceneCfg,
    point_pads_at,
    start_joint_pos,
)
from sim.envs.preview import PreviewSpec

ASSETS_DIR = task_asset_dir("loose_parts_box_transfer")
BOX_XY = (0.24, -0.03)
GOAL_XY = (0.36, 0.20)
INNER_HALF_LENGTH = BOX_LENGTH / 2 - BOX_WALL
NUT_OFFSETS = tuple(
    (fraction * INNER_HALF_LENGTH, y)
    for y in (-0.041, 0.041)
    for fraction in (-7 / 9, -7 / 15, -7 / 45, 7 / 45, 7 / 15, 7 / 9)
)
BALL_SPECS = tuple(
    (fraction * INNER_HALF_LENGTH, y, LARGE_BALL_RADIUS, 0.0)
    for y in (-0.018, 0.018)
    for fraction in (-5 / 9, 5 / 9)
) + tuple(
    (fraction * INNER_HALF_LENGTH, y, BALL_RADIUS, 0.0)
    for fraction, y in ((-77 / 90, 0.0), (77 / 90, 0.0), (0.0, -0.019), (0.0, 0.019))
)
NUT_NAMES = tuple(f"nut_{i}" for i in range(len(NUT_OFFSETS)))
BALL_NAMES = tuple(f"ball_{i}" for i in range(len(BALL_SPECS)))
LOOSE_NAMES = NUT_NAMES + BALL_NAMES
START_JOINT_POS = REAL_HOME_JOINT_POS
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
PAD_TARGET = {
    name: "{ENV_REGEX_NS}/parts_box/" + ("near_wall" if name.startswith("right") else "far_wall")
    for name in TACTILE_SENSORS
}


def _usd(name: str) -> Path:
    return ASSETS_DIR / f"{name}.usd"


def ensure_assets() -> None:
    names = ("parts_box", "hex_nut", "folded_shelf")
    missing = [name for name in names if not _usd(name).exists()]
    if missing:
        raise FileNotFoundError(f"loose_parts_box_transfer assets missing: {', '.join(missing)}")


def _spawn_box() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/parts_box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd("parts_box")),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOX_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*BOX_XY, TABLE_TOP_Z)),
    )


def _spawn_nut(index: int) -> RigidObjectCfg:
    x, y = NUT_OFFSETS[index]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/nut_{index}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_usd("hex_nut")),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=NUT_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=NUT_COLOR, metallic=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(BOX_XY[0] + x, BOX_XY[1] + y, TABLE_TOP_Z + BOX_WALL)),
    )


def _spawn_ball(index: int) -> RigidObjectCfg:
    x, y, radius, z_offset = BALL_SPECS[index]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/ball_{index}",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=LARGE_BALL_COLOR if radius > BALL_RADIUS else BALL_COLOR,
                metallic=0.0 if radius > BALL_RADIUS else 0.35,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BOX_XY[0] + x, BOX_XY[1] + y, TABLE_TOP_Z + BOX_WALL + radius + z_offset)
        ),
    )


@configclass
class LoosePartsBoxTransferSceneCfg(BimanualTactileSceneCfg):
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
    shelf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/shelf",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_usd("folded_shelf"))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*GOAL_XY, TABLE_TOP_Z)),
    )
    goal_mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/goal_mat",
        spawn=sim_utils.CuboidCfg(
            size=(MAT_LENGTH, MAT_WIDTH, MAT_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=MAT_COLOR),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*GOAL_XY, TABLE_TOP_Z + SHELF_HEIGHT + MAT_HEIGHT / 2)),
    )
    parts_box = _spawn_box()
    nut_0 = _spawn_nut(0)
    nut_1 = _spawn_nut(1)
    nut_2 = _spawn_nut(2)
    nut_3 = _spawn_nut(3)
    nut_4 = _spawn_nut(4)
    nut_5 = _spawn_nut(5)
    nut_6 = _spawn_nut(6)
    nut_7 = _spawn_nut(7)
    nut_8 = _spawn_nut(8)
    nut_9 = _spawn_nut(9)
    nut_10 = _spawn_nut(10)
    nut_11 = _spawn_nut(11)
    ball_0 = _spawn_ball(0)
    ball_1 = _spawn_ball(1)
    ball_2 = _spawn_ball(2)
    ball_3 = _spawn_ball(3)
    ball_4 = _spawn_ball(4)
    ball_5 = _spawn_ball(5)
    ball_6 = _spawn_ball(6)
    ball_7 = _spawn_ball(7)

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)


PREVIEW = PreviewSpec(
    sdf_prims=("parts_box",),
    work_target=(0.34, 0.0, TABLE_TOP_Z + 0.06),
)


def make_preview_cfg() -> LoosePartsBoxTransferSceneCfg:
    return LoosePartsBoxTransferSceneCfg(num_envs=1, env_spacing=3.0)
