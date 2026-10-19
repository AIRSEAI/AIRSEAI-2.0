# SPDX-License-Identifier: Apache-2.0
"""Robot layer: a bimanual OpenArm with four tactile pads, and nothing else.

Layering::

    envs.bimanual_tactile       robot + tactile pads, nothing else   (this file)
    envs.<task>                 + table + the task's objects
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from sim.envs.preview import PreviewSpec

__all__ = [
    "ARM_LIFTED_JOINT_POS",
    "FINGER_JOINTS",
    "GEL_GAP_OFFSET",
    "GRASP_POINT",
    "REAL_HOME_JOINT_POS",
    "RIGHT_ARM_SENSORS",
    "SENSOR_FEATURE_NAME",
    "TABLE_TOP_Z",
    "TACTILE_SENSORS",
    "TCP_BLADE_L",
    "TCP_JAW_L",
    "BimanualTactileSceneCfg",
    "gel_gap",
    "point_pads_at",
    "start_joint_pos",
]


# ----- world -----------------------------------------------------------------
# Work-surface height, metres, shared by every table-top task.
TABLE_TOP_Z = 0.20

# Where the placeholder sphere spawns: world coordinates, robot at the origin.
GRASP_POINT = (0.0003, -0.1533, 0.0969)


# ----- gripper ---------------------------------------------------------------
# The gripper's frame in TCP-local coordinates.
TCP_BLADE_L = (0.0, 0.0, 1.0)  # along the blade, palm on this side

TCP_JAW_L = (0.0, 1.0, 0.0)  # the pad normal: the jaws close along it

# Gap between the two gel faces at gripper command g is `2g - GEL_GAP_OFFSET`,
# metres. Use it to size objects and to floor a closing ramp; read the
# penetration field, not this, for how hard a grasp is squeezing.
GEL_GAP_OFFSET = 0.006

FINGER_JOINTS = {"right": "openarm_right_finger_joint.*", "left": "openarm_left_finger_joint.*"}


def gel_gap(g: float) -> float:
    """Gap between the two gel faces at gripper command `g`, metres."""
    return 2.0 * g - GEL_GAP_OFFSET


# ----- sensors ---------------------------------------------------------------
# Scene attribute names of the four tactile pads.
TACTILE_SENSORS = (
    "right_left_finger_tactile",
    "right_right_finger_tactile",
    "left_left_finger_tactile",
    "left_right_finger_tactile",
)

# The two pads on the right gripper, the arm the real robot instruments.
RIGHT_ARM_SENSORS = TACTILE_SENSORS[:2]

# Scene attribute -> dataset feature name.
SENSOR_FEATURE_NAME = {
    "right_left_finger_tactile": "right_hand_middle_finger",
    "right_right_finger_tactile": "right_hand_index_finger",
    "left_left_finger_tactile": "left_hand_middle_finger",
    "left_right_finger_tactile": "left_hand_index_finger",
}


# ----- poses -----------------------------------------------------------------
# The real robot's HOME pose, radians. Keep it equal to the robot's HOME_POSITION.
REAL_HOME_JOINT_POS = {
    "openarm_right_joint1": math.radians(-30.0),
    "openarm_right_joint2": 0.0,
    "openarm_right_joint3": 0.0,
    "openarm_right_joint4": math.radians(100.0),
    "openarm_right_joint5": 0.0,
    "openarm_right_joint6": 0.0,
    "openarm_right_joint7": math.radians(30.0),
    "openarm_left_joint1": math.radians(30.0),
    "openarm_left_joint2": 0.0,
    "openarm_left_joint3": 0.0,
    "openarm_left_joint4": math.radians(100.0),
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": math.radians(-30.0),
}

# Both hands raised clear of a tabletop, radians.
ARM_LIFTED_JOINT_POS = {
    "openarm_right_joint1": 0.0,
    "openarm_right_joint2": 0.35,
    "openarm_right_joint3": 0.0,
    "openarm_right_joint4": 1.2,
    "openarm_right_joint5": 0.0,
    "openarm_right_joint6": 0.0,
    "openarm_right_joint7": 0.0,
    "openarm_left_joint1": 0.0,
    "openarm_left_joint2": -0.35,
    "openarm_left_joint3": 0.0,
    "openarm_left_joint4": 1.2,
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": 0.0,
}


# ----- spawn helpers ---------------------------------------------------------
def _tactile_pad(finger: str) -> VisuoTactileSensorCfg:
    """One GelSight pad on finger link `openarm_<finger>_finger_tactile`.

    The camera renders at GelSight's calibrated size; recorders upscale it to
    the dataset's 288 x 384. Every pad senses the placeholder sphere until a
    task scene re-points it with `point_pads_at`.
    """
    link = f"{{ENV_REGEX_NS}}/Robot/openarm_{finger}_finger_tactile"
    return VisuoTactileSensorCfg(
        prim_path=f"{link}/elastomer/tactile_sensor",
        history_length=0,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(288, 384),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/manipuland",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path=f"{link}/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


# ----- scene -----------------------------------------------------------------
@configclass
class BimanualTactileSceneCfg(InteractiveSceneCfg):
    """Bimanual OpenArm, four tactile pads (two per gripper), one placeholder sphere.

    `robot.spawn.usd_path` is empty on purpose: set it to the tactile robot
    USD (`sim.assets.provision.ensure_robot_asset()`) before constructing.
    """

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        # Compliant contact lets the gels interpenetrate; that penetration is
        # the tactile signal. Gel paths are relative to the robot prim.
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path="",
            physics_material_prim_path=[
                "openarm_right_left_finger_tactile/elastomer",
                "openarm_right_right_finger_tactile/elastomer",
                "openarm_left_left_finger_tactile/elastomer",
                "openarm_left_right_finger_tactile/elastomer",
            ],
            compliant_contact_stiffness=2000.0,
            compliant_contact_damping=20.0,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.0005),
            # Gravity off on the links: the arm is driven by task-space IK.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={"openarm_right_finger_joint.*": 0.014, "openarm_left_finger_joint.*": 0.010},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]", "openarm_right_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # Only finger_joint1 is driven; finger_joint2 mirrors it through
            # a mimic constraint.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint1", "openarm_right_finger_joint1"],
                stiffness=2000.0,
                damping=100.0,
                effort_limit_sim=60.0,
            ),
        },
    )

    # The minimal scene tactile target. Task scenes replace it with their own objects.
    manipuland = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/manipuland",
        spawn=sim_utils.MeshSphereCfg(
            radius=0.008,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2), metallic=0.3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=GRASP_POINT),
    )

    right_left_finger_tactile = _tactile_pad("right_left")
    right_right_finger_tactile = _tactile_pad("right_right")
    left_left_finger_tactile = _tactile_pad("left_left")
    left_right_finger_tactile = _tactile_pad("left_right")


# ----- helpers for task scenes -----------------------------------------------
def start_joint_pos(base: dict, jaw_width: dict, **overrides) -> dict:
    """Initial joint targets: `base` (one of the poses above), each gripper
    opened to `jaw_width[side]` metres, plus any per-joint `overrides`."""
    return dict(base) | {FINGER_JOINTS[s]: w for s, w in jaw_width.items()} | overrides


def point_pads_at(cfg, target) -> None:
    """Set every tactile pad's sensed prim. `target` is one prim expression
    for all four pads, or a {pad name: prim expression} dict."""
    for pad in TACTILE_SENSORS:
        prim = target[pad] if isinstance(target, dict) else target
        getattr(cfg, pad).contact_object_prim_path_expr = prim


# -- preview -----------------------------------------------------------------

PREVIEW = PreviewSpec(sdf_prims=("manipuland",), work_target=(0.0, -0.15, 0.10))


def make_preview_cfg() -> BimanualTactileSceneCfg:
    """Build the scene for preview."""
    cfg = BimanualTactileSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.robot.init_state.joint_pos = start_joint_pos(REAL_HOME_JOINT_POS, {"right": 0.0, "left": 0.0})
    return cfg
