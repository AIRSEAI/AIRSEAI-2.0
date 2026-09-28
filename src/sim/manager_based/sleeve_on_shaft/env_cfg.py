"""Native batched sleeve_on_shaft scene and per-environment contacts."""

from sim.envs import sleeve_on_shaft as scene
from sim.envs.bimanual_tactile import TACTILE_SENSORS
from sim.envs.sleeve_on_shaft import MATERIALS, material_prims
from sim.manager_based.common.env_cfg import build_cfg
from sim.manager_based.common.tactile import SelectableTactileSensor
from sim.tasks.sleeve_on_shaft import TASK


class TaskTactileSensor(SelectableTactileSensor):
    target_names = tuple(prim for material in MATERIALS for prim in material_prims(material.name))


def configure_scene(env, env_ids):
    env.collision_manifest = []
    for slot in range(env.num_envs):
        path = f"/World/envs/env_{slot}"
        meshes = TASK.apply_collision(None, env_path=path)
        for name, paths in meshes.items():
            if not paths:
                raise RuntimeError(f"No collision meshes configured: {path}/{name}")
            env.collision_manifest.append({"env_id": slot, "object": name, "meshes": paths})
        TASK.author_joints(path)


def make_cfg(**kwargs):
    if kwargs.get("scene_cfg") is None:
        kwargs["scene_cfg"] = TASK.make_scene_cfg({name: knob.default for name, knob in TASK.knobs.items()})
    # Keep parked shells at the joint's rest separation before physics starts.
    for material in MATERIALS:
        a, b = material_prims(material.name)
        first, second = (getattr(kwargs["scene_cfg"], prim) for prim in (a, b))
        if first.init_state.pos == second.init_state.pos:
            x, y, z = first.init_state.pos
            first.init_state.pos = (x, y + scene.SPLIT_GAP / 2, z)
            second.init_state.pos = (x, y - scene.SPLIT_GAP / 2, z)
    cfg = build_cfg(scene, **kwargs)
    cfg.events.collisions.func = configure_scene
    cfg.events.collisions.params = {}
    for name in TACTILE_SENSORS:
        getattr(cfg.scene, name).class_type = TaskTactileSensor
    return cfg
