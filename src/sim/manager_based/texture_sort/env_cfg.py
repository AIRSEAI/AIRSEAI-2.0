# SPDX-License-Identifier: Apache-2.0
"""texture_sort uses the shared native-batch environment configuration."""

from sim.envs import texture_sort as scene
from sim.envs.bimanual_tactile import TACTILE_SENSORS
from sim.manager_based.common.env_cfg import build_cfg
from sim.manager_based.common.tactile import SelectableTactileSensor


class TextureTactileSensor(SelectableTactileSensor):
    target_names = ("puck_smooth", "puck_rough")


def restore_task_state(env, states, env_ids):
    for slot, state in zip(env_ids, states, strict=True):
        target = "puck_" + state["task_vars"]["target_key"]
        for name in TACTILE_SENSORS:
            env.scene[name].select_target([slot], target)


def make_cfg(**kwargs):
    cfg = build_cfg(scene, **kwargs)
    for name in TACTILE_SENSORS:
        getattr(cfg.scene, name).class_type = TextureTactileSensor
    cfg.restore_task_state = restore_task_state
    return cfg
