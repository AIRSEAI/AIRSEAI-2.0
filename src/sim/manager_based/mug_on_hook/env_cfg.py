"""Native batched mug_on_hook scene and per-environment contacts."""

from sim.envs import mug_on_hook as scene
from sim.manager_based.common.env_cfg import build_cfg
from sim.tasks.mug_on_hook import TASK


def configure_scene(env, env_ids):
    env.collision_manifest = []
    for slot in range(env.num_envs):
        path = f"/World/envs/env_{slot}"
        meshes = TASK.apply_collision(None, env_path=path)
        for name, paths in meshes.items():
            if not paths:
                raise RuntimeError(f"No collision meshes configured: {path}/{name}")
            env.collision_manifest.append({"env_id": slot, "object": name, "meshes": paths})


def make_cfg(**kwargs):
    if kwargs.get("scene_cfg") is None:
        kwargs["scene_cfg"] = TASK.make_scene_cfg({name: knob.default for name, knob in TASK.knobs.items()})
    cfg = build_cfg(scene, **kwargs)
    cfg.events.collisions.func = configure_scene
    cfg.events.collisions.params = {}
    return cfg
