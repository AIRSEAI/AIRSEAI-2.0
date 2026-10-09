"""loose_parts_box_transfer scene: an open parts box and a destination mat.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else
    envs.loose_parts_box_transfer + table + box + loose parts + mat
"""

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
    BOX_MASS,
    BOX_WALL,
    MAT_COLOR,
    MAT_HEIGHT,
    MAT_LENGTH,
    MAT_WIDTH,
    NUT_COLOR,
    NUT_HEIGHT,
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
NUT_OFFSETS = (
    (-0.070, -0.041), (-0.042, -0.041), (-0.014, -0.041),
    (0.014, -0.041), (0.042, -0.041), (0.070, -0.041),
    (-0.070, 0.041), (-0.042, 0.041), (-0.014, 0.041),
    (0.014, 0.041), (0.042, 0.041), (0.070, 0.041),
)
BALL_OFFSETS = tuple(
    (x, y)
    for y in (-0.044, -0.022, 0.0, 0.022, 0.044)
    for x in (-0.077, -0.055, -0.033, -0.011, 0.011, 0.033, 0.055, 0.077)
    if not (abs(y) > 0.03 and abs(x) < 0.02)
)
BALL_Z_OFFSETS = tuple(NUT_HEIGHT if abs(y) > 0.03 else 0.0 for _, y in BALL_OFFSETS)
NUT_NAMES = tuple(f"nut_{i}" for i in range(len(NUT_OFFSETS)))
BALL_NAMES = tuple(f"ball_{i}" for i in range(len(BALL_OFFSETS)))
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
    names = ("parts_box", "hex_nut", "goal_mat", "folded_shelf")
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
    x, y = BALL_OFFSETS[index]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/ball_{index}",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BALL_COLOR, metallic=0.35),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BOX_XY[0] + x, BOX_XY[1] + y, TABLE_TOP_Z + BOX_WALL + BALL_RADIUS + BALL_Z_OFFSETS[index])
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
    ball_8 = _spawn_ball(8)
    ball_9 = _spawn_ball(9)
    ball_10 = _spawn_ball(10)
    ball_11 = _spawn_ball(11)
    ball_12 = _spawn_ball(12)
    ball_13 = _spawn_ball(13)
    ball_14 = _spawn_ball(14)
    ball_15 = _spawn_ball(15)
    ball_16 = _spawn_ball(16)
    ball_17 = _spawn_ball(17)
    ball_18 = _spawn_ball(18)
    ball_19 = _spawn_ball(19)
    ball_20 = _spawn_ball(20)
    ball_21 = _spawn_ball(21)
    ball_22 = _spawn_ball(22)
    ball_23 = _spawn_ball(23)
    ball_24 = _spawn_ball(24)
    ball_25 = _spawn_ball(25)
    ball_26 = _spawn_ball(26)
    ball_27 = _spawn_ball(27)
    ball_28 = _spawn_ball(28)
    ball_29 = _spawn_ball(29)
    ball_30 = _spawn_ball(30)
    ball_31 = _spawn_ball(31)
    ball_32 = _spawn_ball(32)
    ball_33 = _spawn_ball(33)
    ball_34 = _spawn_ball(34)
    ball_35 = _spawn_ball(35)

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)


PREVIEW = PreviewSpec(
    sdf_prims=("parts_box",),
    work_target=(0.34, 0.0, TABLE_TOP_Z + 0.06),
)


def make_preview_cfg() -> LoosePartsBoxTransferSceneCfg:
    from sim.tasks.loose_parts_box_transfer import TASK

    return TASK.make_scene_cfg({name: knob.default for name, knob in TASK.knobs.items()})
