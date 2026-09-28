"""One environment construction path, including dataset and review cameras."""

import math

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass

from sim.assets.provision import ensure_robot_asset
from sim.envs.bimanual_tactile import TACTILE_SENSORS
from sim.harness import look_at
from sim.physx_settings import small_scene_physx_cfg

from . import mdp
from .tactile import CompatibleTactileSensor


@configclass
class ActionsCfg:
    joints = lab_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=mdp.JOINT_NAMES,
        preserve_order=True,
        scale={"openarm_(right|left)_joint[1-7]": math.pi / 180, "openarm_.*_finger_joint1": 0.001},
        use_default_offset=False,
    )


@configclass
class StateCfg(ObservationGroupCfg):
    joints = ObservationTermCfg(func=mdp.joint_state)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    state = StateCfg()


@configclass
class EventsCfg:
    collisions = EventTermCfg(func=mdp.configure_collisions, mode="prestartup")
    reset = EventTermCfg(func=mdp.reset_scene, mode="reset")


@configclass
class VectorEnvCfg(ManagerBasedEnvCfg):
    restore_task_state = None


def build_cfg(
    scene_module,
    *,
    num_envs=1,
    device="cuda:0",
    robot_usd="",
    joint_deg="",
    scene_cfg=None,
    seed=0,
    review_size=(960, 720),
):
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    scene = scene_module.make_preview_cfg() if scene_cfg is None else scene_cfg
    scene.num_envs = num_envs
    # TacSL requires per-environment cooked SDF shapes. Keep USD physics
    # independent while sharing a single PhysX tensor simulation/Kit process.
    scene.replicate_physics = False
    scene.robot.spawn.usd_path = ensure_robot_asset(robot_usd)
    if joint_deg:
        for item in joint_deg.split(","):
            name, value = item.split("=")
            scene.robot.init_state.joint_pos[name.strip()] = math.radians(float(value))
    spec = scene_module.PREVIEW
    scene.review_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/review_camera",
        width=review_size[0],
        height=review_size[1],
        offset=TiledCameraCfg.OffsetCfg(
            pos=spec.eye, rot=look_at(spec.eye, spec.look_at), convention="opengl"
        ),
        data_types=["rgb"],
        spawn=PinholeCameraCfg(),
    )
    for name, link in [
        ("base", "openarm_body_link/head_camera/camera"),
        ("left_wrist", "openarm_left_hand/wrist_camera/camera"),
        ("right_wrist", "openarm_right_hand/wrist_camera/camera"),
    ]:
        setattr(
            scene,
            f"{name}_camera",
            TiledCameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
                width=640,
                height=480,
                data_types=["rgb"],
                spawn=None,
            ),
        )
    for name in TACTILE_SENSORS:
        getattr(scene, name).class_type = CompatibleTactileSensor
    events = EventsCfg()
    events.collisions.params = dict(
        sdf_prims=spec.sdf_prims, tri_prims=spec.tri_prims, sdf_resolution=spec.sdf_resolution
    )
    # Stateless antialiasing avoids temporal reconstruction retaining another
    # episode's silhouettes after an individual environment teleports.
    cfg = VectorEnvCfg(
        scene=scene,
        actions=ActionsCfg(),
        observations=ObservationsCfg(),
        events=events,
        decimation=3,
        seed=seed,
        num_rerenders_on_reset=2,
        sim=SimulationCfg(
            dt=1 / 60,
            render_interval=3,
            device=device,
            physx=small_scene_physx_cfg(),
            render=RenderCfg(antialiasing_mode="FXAA", enable_dlssg=False),
        ),
    )
    cfg.viewer.eye = spec.eye
    cfg.viewer.lookat = spec.look_at
    return cfg
