# SPDX-License-Identifier: Apache-2.0
"""Construct native batched tactile environments."""


def make_env_cfg(task, **kwargs):
    from importlib import import_module

    from sim.tasks.registry import TASKS

    entry = TASKS.get(task)
    if entry is None or entry.manager_cfg is None:
        raise ValueError(f"No manager-based environment for {task!r}")
    return import_module(entry.manager_cfg).make_cfg(**kwargs)


def make_env(task, **kwargs):
    from isaaclab.envs import ManagerBasedEnv

    xr_state_only = bool(kwargs.get("xr_state_only", False))
    cfg = make_env_cfg(task, **kwargs)
    cfg.wait_for_textures = False
    env = ManagerBasedEnv(cfg)
    if not xr_state_only:
        # Capture optical zero-contact references before any task motion/restore.
        # Nominal objects are on the table, away from the home-position fingers.
        from sim.envs.bimanual_tactile import TACTILE_SENSORS

        for _ in range(2):
            env.sim.render()
        for name in TACTILE_SENSORS:
            env.scene[name].get_initial_render()
    return env
