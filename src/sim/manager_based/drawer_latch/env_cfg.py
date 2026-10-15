# SPDX-License-Identifier: Apache-2.0
"""Native batched drawer_latch scene and sliding joints."""

from sim.envs import drawer_latch as scene
from sim.envs.drawer_latch import reset_drive
from sim.manager_based.common.env_cfg import build_cfg
from sim.tasks.drawer_latch import TASK


def configure_scene(env, env_ids):
    env.collision_manifest = []
    for slot in range(env.num_envs):
        path = f"/World/envs/env_{slot}"
        meshes = TASK.apply_collision(None, env_path=path)
        if not meshes["drawer"]:
            raise RuntimeError(f"No drawer collision meshes configured: {path}")
        env.collision_manifest.append({"env_id": slot, "object": "drawer", "meshes": meshes["drawer"]})


def restore_task_state(env, states, env_ids):
    for slot, state in zip(env_ids, states, strict=True):
        reset_drive(f"/World/envs/env_{slot}", float(state["task_vars"]["yaw_deg"]))


def make_cfg(**kwargs):
    if kwargs.get("scene_cfg") is None:
        kwargs["scene_cfg"] = TASK.make_scene_cfg({name: knob.default for name, knob in TASK.knobs.items()})
    cfg = build_cfg(scene, **kwargs)
    cfg.events.collisions.func = configure_scene
    cfg.events.collisions.params = {}
    cfg.restore_task_state = restore_task_state
    return cfg
