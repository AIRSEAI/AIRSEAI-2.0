#!/usr/bin/env python3
"""Launch a tactile environment: interactive viewer, or an offscreen video.

Each entry in `ENVS` pairs a scene config with the prims that need SDF or
triangle-mesh collision and a camera framing. The robot asset builds itself
on first use; `--robot_usd` overrides it.

    # the minimal unit (bimanual OpenArm + 4 tactile fingers + cameras)
    python benchmark/run/launch_env.py --env minimal --gui
    # a task environment (adds that task's furniture and objects)
    python benchmark/run/launch_env.py --env stack_cube --gui
    # offscreen, as an mp4 or a single settled PNG
    python benchmark/run/launch_env.py --env stack_cube --video ~/demo_stack_cube.mp4 --steps 300
    python benchmark/run/launch_env.py --env stack_cube --snapshot x.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch a tactile environment.")
parser.add_argument("--env", default="stack_cube",
                    help="registry key, see ENVS")
parser.add_argument("--robot_usd", type=str, default="",
                    help="override the asset; default builds/uses the "
                         "one under assets/")
parser.add_argument("--gui", action="store_true",
                    help="open the Isaac viewer (default is offscreen)")
parser.add_argument("--video", type=str, default="",
                    help="record the scene to this mp4 instead of watching it")
parser.add_argument("--steps", type=int, default=0,
                    help="step count before exit; 0 = run until closed "
                         "(--video defaults to 300)")
parser.add_argument("--fps", type=int, default=30, help="video frame rate")
parser.add_argument("--snapshot", type=str, default="",
                    help="write one settled PNG of the scene to this path and exit")
parser.add_argument("--settle_steps", type=int, default=90,
                    help="physics steps before the snapshot")
parser.add_argument("--joint_deg", type=str, default="",
                    help="initial joint override, 'name=deg,...' (e.g. openarm_right_joint7=90)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --gui decides the app mode; cameras are always on because the tactile
# sensors are camera-based and the scene refuses to build without them.
args_cli.headless = not args_cli.gui
args_cli.enable_cameras = True
if args_cli.video and not args_cli.steps:
    args_cli.steps = 300
if args_cli.snapshot:
    args_cli.steps = args_cli.settle_steps

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import math

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import TiledCamera, TiledCameraCfg

from sim.assets.provision import ensure_robot_asset
from sim.envs.bimanual_tactile import TABLE_TOP_Z
from sim.physx_settings import small_scene_physx_cfg
from sim.tactile import (
    apply_sdf_collision,
    apply_triangle_mesh_collision,
)


class EnvSpec:
    """One launchable environment: cfg factory + what to fix up after spawn."""

    def __init__(self, make_cfg, sdf_prims=(), tri_prims=(),
                 look_at=(0.30, -0.08, 0.15), eye=(0.85, -0.45, 0.45),
                 sdf_resolution=256):
        self.make_cfg = make_cfg          # () -> InteractiveSceneCfg
        self.sdf_prims = sdf_prims        # prims needing SDF collision
        self.tri_prims = tri_prims        # prims that must stay triangle meshes
        self.look_at = look_at            # video camera aim
        self.eye = eye
        # PhysX cooks one SDF per sensed mesh at `sim.reset()`. 256 is the
        # default and is fine for a handful of primitives; dense meshes cook
        # slowly at that resolution, so envs with them lower it.
        self.sdf_resolution = sdf_resolution


def _minimal_cfg():
    from sim.envs.bimanual_tactile import BimanualTactileSceneCfg
    return BimanualTactileSceneCfg(num_envs=1, env_spacing=3.0)


def _peg_cfg(light_scale: float = 1.0):
    from sim.envs.blind_peg_insertion import BlindPegInsertionSceneCfg
    cfg = BlindPegInsertionSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.set_clearance()          # anchor board, 1 mm clearance
    cfg.set_light_scale(light_scale)
    return cfg


def _stack_cfg():
    from sim.envs.stack_cube import StackCubeSceneCfg
    return StackCubeSceneCfg(num_envs=1, env_spacing=3.0)


def _handover_cfg():
    from sim.envs.handover import HandoverSceneCfg
    return HandoverSceneCfg(num_envs=1, env_spacing=3.0)


def _cap_unscrew_cfg():
    from sim.envs.cap_unscrew import CapUnscrewSceneCfg
    return CapUnscrewSceneCfg(num_envs=1, env_spacing=3.0)


def _sleeve_cfg():
    from sim.envs.sleeve_on_shaft import SleeveOnShaftSceneCfg
    return SleeveOnShaftSceneCfg(num_envs=1, env_spacing=3.0)


def _sort_cfg():
    from sim.envs.texture_sort import TextureSortSceneCfg
    cfg = TextureSortSceneCfg(num_envs=1, env_spacing=3.0)
    # Full light so the preview shows the geometry; the task itself runs at
    # the `--light_scale` given to the runner.
    cfg.set_light_scale(1.0)
    return cfg


ENVS = {
    "minimal": EnvSpec(_minimal_cfg, sdf_prims=("manipuland",),
                       look_at=(0.0, -0.15, 0.10), eye=(0.45, -0.45, 0.35)),
    "stack_cube": EnvSpec(_stack_cfg, sdf_prims=("cube_a", "cube_b")),
    # Pulled back to show the arms relative to the objects.
    "blind_peg_insertion": EnvSpec(_peg_cfg, sdf_prims=("peg",), tri_prims=("board",),
                   look_at=(0.20, -0.08, 0.18), eye=(1.05, -0.80, 0.62)),
    # Framed to show both cups and the sleeve standing in the source one.
    "handover": EnvSpec(_handover_cfg, sdf_prims=("sleeve_a", "sleeve_b"),
                        tri_prims=("cup_source", "cup_target"),
                        look_at=(0.22, 0.0, TABLE_TOP_Z + 0.05), eye=(0.95, -0.50, 0.60),
                        sdf_resolution=128),
    # Framed to take in both arms at HOME, the bottle on the centreline and
    # the tray on the right.
    "cap_unscrew": EnvSpec(_cap_unscrew_cfg, sdf_prims=("bottle", "cap"),
                           tri_prims=("tray",),
                           look_at=(0.20, -0.06, TABLE_TOP_Z + 0.08),
                           eye=(1.00, -0.70, 0.62), sdf_resolution=128),
    "sleeve_on_shaft": EnvSpec(
        _sleeve_cfg,
        sdf_prims=("soft_a", "soft_b", "stiff_a", "stiff_b"),
        tri_prims=("shaft",),
        look_at=(0.24, -0.08, 0.24), eye=(0.95, -0.72, 0.52),
        sdf_resolution=128),
    # Higher eye (about 44 degrees elevation): at the 22 degrees the other envs
    # use, the near rim of each 80 mm bin hides its 27 mm pit.
    "texture_sort": EnvSpec(
        _sort_cfg, sdf_prims=("puck_smooth", "puck_rough"),
        tri_prims=("bin_0", "bin_1"),
        # Pulled back to take in the whole robot: the task starts from HOME
        # and the snapshot has to show that posture, not only the table.
        look_at=(0.15, -0.08, 0.30), eye=(1.25, -1.00, 0.80),
        sdf_resolution=96),
}


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = map(np.asarray, (eye, target, up))
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    w = max(math.sqrt(max(1.0 + R.trace(), 0.0)) / 2.0, 1e-8)
    return (w, (R[2, 1] - R[1, 2]) / (4 * w),
            (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))


def main() -> int:
    spec = ENVS[args_cli.env]
    usd = ensure_robot_asset(args_cli.robot_usd)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1 / 60, device=args_cli.device, physx=small_scene_physx_cfg()))
    cfg = spec.make_cfg()
    if args_cli.joint_deg:
        for item in args_cli.joint_deg.split(","):
            name, deg = item.split("=")
            cfg.robot.init_state.joint_pos[name.strip()] = math.radians(float(deg))
            print(f"STEP joint override {name.strip()} = {deg} deg", flush=True)
    cfg.robot.spawn.usd_path = usd
    scene = InteractiveScene(cfg)
    for prim in spec.sdf_prims:
        n = apply_sdf_collision(f"/World/envs/env_0/{prim}",
                                sdf_resolution=spec.sdf_resolution)
        print(f"[launch_env] sdf on {prim}: {len(n)} mesh(es)", flush=True)
    for prim in spec.tri_prims:
        n = apply_triangle_mesh_collision(f"/World/envs/env_0/{prim}")
        print(f"[launch_env] triangle mesh on {prim}: {len(n)} mesh(es)",
              flush=True)

    cam = None
    if args_cli.video or args_cli.snapshot:
        cam = TiledCamera(TiledCameraCfg(
            prim_path="/World/scene_cam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=spec.eye, rot=_look_at_quat(spec.eye, spec.look_at),
                convention="opengl"),
            width=960, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg()))

    sim.reset()
    robot = scene["robot"]
    # Place the arms at their start pose, as the task runner's reset does:
    # otherwise they begin folded and swing up through the scene under their
    # drives.
    robot.write_joint_state_to_sim(robot.data.default_joint_pos,
                                   robot.data.default_joint_vel)
    print(f"[launch_env] {args_cli.env}: {len(robot.joint_names)} joints, "
          f"{len(scene.sensors)} sensors", flush=True)

    writer = None
    if args_cli.video:
        from lerobot_tactile_ext.codec import Mp4Stream
        path = Path(os.path.expanduser(args_cli.video))
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = Mp4Stream(path, args_cli.fps)

    every = max(1, round(60 / args_cli.fps))     # sim runs at 60 Hz
    i = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=True)
        scene.update(1 / 60)
        if cam is not None:
            cam.update(1 / 60)
            if writer is not None and i % every == 0:
                writer.add(cam.data.output["rgb"][0, ..., :3]
                           .detach().cpu().numpy())
        i += 1
        if args_cli.steps and i >= args_cli.steps:
            break

    if writer is not None:
        stats = writer.close()
        print(f"[launch_env] wrote {args_cli.video} ({stats['frames']} frames)",
              flush=True)
    if args_cli.snapshot:
        from PIL import Image
        path = Path(os.path.expanduser(args_cli.snapshot))
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        Image.fromarray(rgb.astype("uint8")).save(path)
        print(f"[launch_env] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    import sys
    sys.stdout.flush()
    os._exit(rc)       # skip Kit's own shutdown, which can hang
