#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch a tactile environment: interactive viewer, or an offscreen video.

Preview a task or the minimal scene using a shared camera at 20 Hz.

    # the minimal unit (bimanual OpenArm + 4 tactile fingers + cameras)
    python benchmark/run/launch_env.py --env minimal --gui
    # a task environment (adds that task's furniture and objects)
    python benchmark/run/launch_env.py --env <task> --gui
    # offscreen, as an mp4 or a single settled PNG
    python benchmark/run/launch_env.py --env <task> --video ~/<task>_clip.mp4 --steps 300
    python benchmark/run/launch_env.py --env <task> --snapshot x.png
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher
from PIL import Image

from lerobot_tactile_ext.codec import Mp4Stream
from sim.manager_based import make_env
from sim.manager_based.common.mdp import hold_action
from sim.manager_based.common.metrics import GPUMemoryMonitor
from sim.manager_based.common.provenance import write_manifest
from sim.runtime import install_error_handler
from sim.tactile import apply_sdf_collision, apply_triangle_mesh_collision
from sim.tasks.registry import TASKS, load_scene
from sim.video import compose_grid_frame

install_error_handler()

MINIMAL = "minimal"  # the bare robot, not a task
ENV_IDS = [MINIMAL] + sorted(TASKS)
VIDEO_FPS = 20
PHYSICS_HZ = 60
CAMERA_PATH = "/World/scene_cam"
CAMERA_RESOLUTION = (960, 720)

parser = argparse.ArgumentParser(description="Launch a tactile environment.")
parser.add_argument(
    "--env", required=True, choices=ENV_IDS, help="a registry task id, or `minimal` for the bare robot"
)
parser.add_argument("--num_envs", type=int, default=1, help="native environments in one Kit process")
parser.add_argument("--view_env", type=int, default=-1, help="GUI focus slot; -1 shows the whole scene")
parser.add_argument("--report", default="", help="write preview metadata JSON")
parser.add_argument(
    "--robot_usd", type=str, default="", help="override the asset; default builds/uses the one under assets/"
)
parser.add_argument("--gui", action="store_true", help="open the Isaac viewer (default is offscreen)")
parser.add_argument(
    "--video", type=str, default="", help="record the scene to this mp4 instead of watching it"
)
parser.add_argument(
    "--steps", type=int, help="step count before exit; 0 = run until closed (--video defaults to 300)"
)
parser.add_argument(
    "--fps", type=int, choices=(VIDEO_FPS,), default=VIDEO_FPS, help="video frame rate (fixed at 20 Hz)"
)
parser.add_argument(
    "--snapshot", type=str, default="", help="write one settled PNG of the scene to this path and exit"
)
parser.add_argument("--settle_steps", type=int, default=90, help="physics steps before the snapshot")
parser.add_argument(
    "--joint_deg",
    type=str,
    default="",
    help="initial joint override, 'name=deg,...' (e.g. openarm_right_joint7=90)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
manager_preview = args_cli.env in TASKS and TASKS[args_cli.env].manager_cfg is not None
if args_cli.num_envs != 1 and not manager_preview:
    parser.error("this environment does not support native multi-env preview")
if args_cli.steps is not None and args_cli.steps < 0:
    parser.error("--steps must be non-negative")
if args_cli.settle_steps < 1:
    parser.error("--settle_steps must be positive")
if not -1 <= args_cli.view_env < args_cli.num_envs:
    parser.error("--view_env must be -1 or an existing environment index")

# --gui decides the app mode; cameras are always on because the tactile
# sensors are camera-based and the scene refuses to build without them.
args_cli.headless = not args_cli.gui
args_cli.enable_cameras = True
if args_cli.video and args_cli.steps is None:
    args_cli.steps = 300
if args_cli.snapshot and args_cli.steps is None:
    args_cli.steps = args_cli.settle_steps

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


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
    return (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))


def legacy_main() -> int:
    # Simulation modules require a running Kit application.
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import TiledCamera, TiledCameraCfg

    from sim.assets.provision import ensure_robot_asset
    from sim.physx_settings import small_scene_physx_cfg

    scene_module = (
        importlib.import_module("sim.envs.bimanual_tactile")
        if args_cli.env == MINIMAL
        else load_scene(args_cli.env)
    )
    spec = scene_module.PREVIEW
    usd = ensure_robot_asset(args_cli.robot_usd)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / PHYSICS_HZ, device=args_cli.device, physx=small_scene_physx_cfg())
    )
    cfg = scene_module.make_preview_cfg()
    if args_cli.joint_deg:
        for item in args_cli.joint_deg.split(","):
            name, deg = item.split("=")
            cfg.robot.init_state.joint_pos[name.strip()] = math.radians(float(deg))
            print(f"STEP joint override {name.strip()} = {deg} deg", flush=True)
    cfg.robot.spawn.usd_path = usd
    scene = InteractiveScene(cfg)
    for prim in spec.sdf_prims:
        n = apply_sdf_collision(f"/World/envs/env_0/{prim}", sdf_resolution=spec.sdf_resolution)
        print(f"[launch_env] sdf on {prim}: {len(n)} mesh(es)", flush=True)
    for prim in spec.tri_prims:
        n = apply_triangle_mesh_collision(f"/World/envs/env_0/{prim}")
        print(f"[launch_env] triangle mesh on {prim}: {len(n)} mesh(es)", flush=True)

    cam = None
    if args_cli.gui or args_cli.video or args_cli.snapshot:
        cam = TiledCamera(
            TiledCameraCfg(
                prim_path=CAMERA_PATH,
                offset=TiledCameraCfg.OffsetCfg(
                    pos=spec.eye, rot=_look_at_quat(spec.eye, spec.look_at), convention="opengl"
                ),
                width=CAMERA_RESOLUTION[0],
                height=CAMERA_RESOLUTION[1],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(),
            )
        )

    sim.reset()
    if args_cli.gui:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active GUI viewport is available")
        viewport.camera_path = Sdf.Path(CAMERA_PATH)
        viewport.resolution = CAMERA_RESOLUTION
    robot = scene["robot"]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.reset()
    robot.write_data_to_sim()
    print("[launch_env] camera " + json.dumps({"eye": spec.eye, "target": spec.look_at}), flush=True)
    print(
        f"[launch_env] {args_cli.env}: {len(robot.joint_names)} joints, {len(scene.sensors)} sensors",
        flush=True,
    )

    writer = None
    if args_cli.video:
        path = Path(os.path.expanduser(args_cli.video))
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = Mp4Stream(path, VIDEO_FPS)

    every = PHYSICS_HZ // VIDEO_FPS
    i = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=True)
        scene.update(1 / PHYSICS_HZ)
        if cam is not None:
            cam.update(1 / PHYSICS_HZ)
            if writer is not None and i % every == 0:
                writer.add(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy())
        i += 1
        if args_cli.steps and i >= args_cli.steps:
            break

    arm_ids = [
        i
        for i, name in enumerate(robot.joint_names)
        if name.startswith(("openarm_left_joint", "openarm_right_joint"))
    ]
    actual = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
    target = robot.data.default_joint_pos[0, arm_ids].detach().cpu().numpy()
    print(
        "[launch_env] posture "
        + json.dumps(
            {
                "joint_names": [robot.joint_names[i] for i in arm_ids],
                "target_deg": np.rad2deg(target).tolist(),
                "actual_deg": np.rad2deg(actual).tolist(),
                "max_error_deg": float(np.max(np.abs(np.rad2deg(actual - target)))),
                "joint_override": args_cli.joint_deg,
            }
        ),
        flush=True,
    )
    if writer is not None:
        stats = writer.close()
        print(f"[launch_env] wrote {args_cli.video} ({stats['frames']} frames)", flush=True)
    if args_cli.snapshot:
        path = Path(os.path.expanduser(args_cli.snapshot))
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        Image.fromarray(rgb.astype("uint8")).save(path)
        print(f"[launch_env] wrote {path}", flush=True)
    return 0


def main() -> int:
    if not manager_preview:
        return legacy_main()
    with ExitStack() as cleanup:
        return manager_main(cleanup)


def manager_main(cleanup) -> int:
    # Scene imports require a running Kit application.
    from sim.envs.bimanual_tactile import TACTILE_SENSORS

    memory = GPUMemoryMonitor(args_cli.device)
    cleanup.callback(memory.finish)
    env = make_env(
        args_cli.env,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        robot_usd=args_cli.robot_usd,
        joint_deg=args_cli.joint_deg,
    )
    cleanup.callback(env.close)
    env.reset()
    action = hold_action(env)
    if args_cli.gui:
        origins = env.scene.env_origins.cpu().numpy()
        spec = load_scene(args_cli.env).PREVIEW
        if args_cli.view_env >= 0:
            origin = origins[args_cli.view_env]
            eye, target = np.array(spec.eye) + origin, np.array(spec.look_at) + origin
        else:
            center = origins.mean(axis=0)
            extent = np.linalg.norm(origins - center, axis=1).max()
            target = center + np.array(spec.look_at)
            eye = target + (np.array(spec.eye) - np.array(spec.look_at)) * (1 + extent)
        env.sim.set_camera_view(eye.tolist(), target.tolist())
    writer = None
    if args_cli.video:
        path = Path(args_cli.video).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = Mp4Stream(path, VIDEO_FPS)
        cleanup.callback(writer.close)
    frames, physics_steps = [], 0
    start = time.monotonic()
    with torch.inference_mode():
        while simulation_app.is_running():
            env.step(action)
            for name in TACTILE_SENSORS:
                _ = env.scene[name].data
            physics_steps += env.cfg.decimation
            if writer is not None:
                rgb = env.scene["review_camera"].data.output["rgb"].detach().cpu().numpy()
                writer.add(
                    compose_grid_frame(
                        list(rgb),
                        [
                            (f"env {slot}", f"sim {physics_steps / PHYSICS_HZ:.3f}s")
                            for slot in range(env.num_envs)
                        ],
                    )
                )
                frames.append(
                    dict(frame=len(frames), sim_tick=physics_steps, env_ids=list(range(env.num_envs)))
                )
            if args_cli.steps and physics_steps >= args_cli.steps:
                break
    if writer is not None:
        Path(args_cli.video).expanduser().with_suffix(".frames.json").write_text(json.dumps(frames, indent=2))
    if args_cli.snapshot:
        path = Path(args_cli.snapshot).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = env.scene["review_camera"].data.output["rgb"].detach().cpu().numpy()
        Image.fromarray(
            compose_grid_frame(
                list(rgb),
                [(f"env {slot}", f"sim {physics_steps / PHYSICS_HZ:.3f}s") for slot in range(env.num_envs)],
            )
        ).save(path)
    report = dict(
        task=args_cli.env,
        num_envs=env.num_envs,
        backend="ManagerBasedEnv",
        pid=os.getpid(),
        antialiasing_mode=env.cfg.sim.render.antialiasing_mode,
        physics_steps=physics_steps,
        simulation_seconds=physics_steps / PHYSICS_HZ,
        wall_seconds=time.monotonic() - start,
        video_frames=len(frames),
        video_fps=VIDEO_FPS,
        env_origins=env.scene.env_origins.cpu().tolist(),
        collisions=env.collision_manifest,
        sensors={name: type(sensor).__name__ for name, sensor in env.scene.sensors.items()},
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(env.device),
        gpu_memory=memory.finish(),
    )
    if args_cli.report:
        path = Path(args_cli.report).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        write_manifest(path.with_suffix(".manifest.json"), **report)
    print("[launch_env] " + json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
