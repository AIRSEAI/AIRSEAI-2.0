# SPDX-License-Identifier: Apache-2.0
"""Native batched cap_unscrew scene and per-environment contacts."""

from sim.envs import cap_unscrew as scene
from sim.manager_based.common.env_cfg import build_cfg
from sim.tasks.cap_unscrew import TASK, THREAD_JOINT_NAME
from sim.tasks.scene_utils import set_joint_enabled


def configure_scene(env, env_ids):
    env.collision_manifest = []
    for slot in range(env.num_envs):
        path = f"/World/envs/env_{slot}"
        meshes = TASK.apply_collision(None, env_path=path)
        for name, paths in meshes.items():
            if not paths:
                raise RuntimeError(f"No collision meshes configured: {path}/{name}")
            env.collision_manifest.append({"env_id": slot, "object": name, "meshes": paths})


def restore_task_state(env, states, env_ids):
    for slot in env_ids:
        set_joint_enabled(f"/World/envs/env_{slot}/{THREAD_JOINT_NAME}", True)


def make_cfg(**kwargs):
    if kwargs.get("scene_cfg") is None:
        kwargs["scene_cfg"] = TASK.make_scene_cfg({name: knob.default for name, knob in TASK.knobs.items()})
    cfg = build_cfg(scene, **kwargs)
    cfg.events.collisions.func = configure_scene
    cfg.events.collisions.params = {}
    cfg.restore_task_state = restore_task_state
    return cfg
