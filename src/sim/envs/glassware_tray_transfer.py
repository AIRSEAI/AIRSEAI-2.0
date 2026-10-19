# SPDX-License-Identifier: Apache-2.0
"""A wooden tray carrying three glass tumblers to a placemat."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.glassware_tray_transfer import CUP_OFFSETS, TRAY_THICKNESS
from sim.envs.bimanual_tactile import (
    REAL_HOME_JOINT_POS,
    TABLE_TOP_Z,
    BimanualTactileSceneCfg,
    point_pads_at,
    start_joint_pos,
)
from sim.envs.preview import PreviewSpec

ASSETS_DIR = task_asset_dir("glassware_tray_transfer")
SURFACE_Z = TABLE_TOP_Z + 0.05
START_XY = (0.195, 0.0)
GOAL_XY = (0.435, 0.0)
SUPPORT_SIZE = (0.26, 0.30, 0.003)
SUPPORT_TOP_Z = SURFACE_Z + SUPPORT_SIZE[2]
TRAY_MASS = 0.16
CUP_MASS = 0.12
SPAWN_CLEARANCE = 0.001
START_JOINT_POS = REAL_HOME_JOINT_POS
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
PAD_TARGET = "{ENV_REGEX_NS}/tray"


def _support(name, xy, colour):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=SUPPORT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*xy, SURFACE_Z + SUPPORT_SIZE[2] / 2)),
    )


@configclass
class GlasswareTrayTransferSceneCfg(BimanualTactileSceneCfg):
    """Robot, wooden tray, three glass tumblers and a destination placemat."""

    manipuland = None
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.62, 0.80, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.32, 0.25)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.34, 0.0, SURFACE_Z - 0.02)),
    )
    goal_support = _support("goal_support", GOAL_XY, (0.16, 0.25, 0.20))
    tray = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tray",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "tray.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.2,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=TRAY_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*START_XY, SURFACE_Z + SPAWN_CLEARANCE)),
    )
    cup_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cup_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "glass_cup.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.2,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUP_MASS),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                START_XY[0] + CUP_OFFSETS[0][0],
                START_XY[1] + CUP_OFFSETS[0][1],
                SURFACE_Z + TRAY_THICKNESS + 2 * SPAWN_CLEARANCE,
            )
        ),
    )
    cup_2 = cup_1.replace(
        prim_path="{ENV_REGEX_NS}/cup_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                START_XY[0] + CUP_OFFSETS[1][0],
                START_XY[1] + CUP_OFFSETS[1][1],
                SURFACE_Z + TRAY_THICKNESS + 2 * SPAWN_CLEARANCE,
            )
        ),
    )
    cup_3 = cup_1.replace(
        prim_path="{ENV_REGEX_NS}/cup_3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                START_XY[0] + CUP_OFFSETS[2][0],
                START_XY[1] + CUP_OFFSETS[2][1],
                SURFACE_Z + TRAY_THICKNESS + 2 * SPAWN_CLEARANCE,
            )
        ),
    )

    def __post_init__(self):
        if any(not (ASSETS_DIR / name).is_file() for name in ("tray.usd", "glass_cup.usd")):
            raise FileNotFoundError(
                "Run python benchmark/run/make_glassware_tray_transfer_assets.py --headless"
            )
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        point_pads_at(self, PAD_TARGET)


PREVIEW = PreviewSpec(sdf_prims=("tray",), work_target=(0.34, 0.0, SUPPORT_TOP_Z))


def make_preview_cfg() -> GlasswareTrayTransferSceneCfg:
    return GlasswareTrayTransferSceneCfg(num_envs=1, env_spacing=3.0)
