# SPDX-License-Identifier: Apache-2.0
"""Named degree/mm actions, observations and per-environment collision setup."""

import math

import torch

from sim.tactile import apply_sdf_collision, apply_triangle_mesh_collision

JOINT_NAMES = [
    name
    for side in ("right", "left")
    for name in [*[f"openarm_{side}_joint{i}" for i in range(1, 8)], f"openarm_{side}_finger_joint1"]
]
SCALES = [math.pi / 180] * 7 + [0.001]
SCALES *= 2


def joint_state(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)
    return robot.data.joint_pos[:, ids] / torch.tensor(SCALES, device=env.device)


def hold_action(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)
    return robot.data.joint_pos_target[:, ids] / torch.tensor(SCALES, device=env.device)


def configure_collisions(env, env_ids, sdf_prims=(), tri_prims=(), sdf_resolution=256):
    env.collision_manifest = []
    for slot in range(env.num_envs):
        for names, func, kwargs in (
            (sdf_prims, apply_sdf_collision, {"sdf_resolution": sdf_resolution}),
            (tri_prims, apply_triangle_mesh_collision, {}),
        ):
            for name in names:
                path = f"/World/envs/env_{slot}/{name}"
                meshes = func(path, **kwargs)
                if not meshes:
                    raise RuntimeError(f"No collision meshes configured: {path}")
                env.collision_manifest.append({"env_id": slot, "object": name, "meshes": meshes})


def reset_scene(env, env_ids):
    from isaaclab.envs.mdp import reset_scene_to_default

    from sim.envs.bimanual_tactile import TACTILE_SENSORS

    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    reset_scene_to_default(env, env_ids)
    robot = env.scene["robot"]
    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)
    for name in TACTILE_SENSORS:
        sensor = env.scene[name]
        if hasattr(sensor, "select_target"):
            sensor.select_target(env_ids, sensor.default_target)
    if hasattr(env, "frozen_state_ids"):
        for slot in env_ids.tolist():
            env.frozen_state_ids[slot] = None
            env.frozen_task_vars[slot] = None
    if hasattr(env, "pending_restores"):
        for slot in env_ids.tolist():
            env.pending_restores.pop(slot, None)
