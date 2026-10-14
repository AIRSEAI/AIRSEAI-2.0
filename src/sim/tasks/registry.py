"""The task registry: the one place a task is registered.

Read by the expert runner, the evaluator, the dataset collector, the
environment viewer and the installer's asset step. An entry names the task's
modules and the knobs its id pins; a task id that pins a knob runs its
parent's module with that knob fixed, and the pin overrides the CLI.

Nothing here imports Isaac: the paths are strings, resolved by whoever needs
them once the app is running.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

__all__ = ["TaskEntry", "TASKS", "load_task", "load_scene"]


@dataclass(frozen=True)
class TaskEntry:
    module: str  # sim.tasks.<name>, the expert
    scene: str  # sim.envs.<name>, the scene and its preview
    assets: str | None = None  # generator script, None when the scene is primitives
    pins: dict = field(default_factory=dict)
    manager_cfg: str | None = None

    @property
    def name(self) -> str:
        """The name the task's modules share. Two ids that differ only in a
        pinned knob share it, and so share a scene and an asset directory."""
        return self.module.rsplit(".", 1)[1]


TASKS: dict[str, TaskEntry] = {
    "glassware_tray_transfer": TaskEntry(
        module="sim.tasks.glassware_tray_transfer",
        scene="sim.envs.glassware_tray_transfer",
        assets="benchmark/run/make_glassware_tray_transfer_assets.py",
        manager_cfg="sim.manager_based.glassware_tray_transfer.env_cfg",
    ),
    "stack_cube": TaskEntry(
        module="sim.tasks.stack_cube",
        scene="sim.envs.stack_cube",
        manager_cfg="sim.manager_based.stack_cube.env_cfg",
    ),
    "blind_peg_insertion": TaskEntry(
        module="sim.tasks.blind_peg_insertion",
        scene="sim.envs.blind_peg_insertion",
        assets="benchmark/run/make_blind_peg_insertion_assets.py",
        manager_cfg="sim.manager_based.blind_peg_insertion.env_cfg",
    ),
    "texture_sort": TaskEntry(
        module="sim.tasks.texture_sort",
        scene="sim.envs.texture_sort",
        assets="benchmark/run/make_texture_sort_assets.py",
        manager_cfg="sim.manager_based.texture_sort.env_cfg",
    ),
    "handover": TaskEntry(
        module="sim.tasks.handover",
        scene="sim.envs.handover",
        assets="benchmark/run/make_handover_assets.py",
        manager_cfg="sim.manager_based.handover.env_cfg",
    ),
    "loose_parts_box_transfer": TaskEntry(
        module="sim.tasks.loose_parts_box_transfer",
        scene="sim.envs.loose_parts_box_transfer",
        assets="benchmark/run/make_loose_parts_box_transfer_assets.py",
        manager_cfg="sim.manager_based.loose_parts_box_transfer.env_cfg",
    ),
    "mug_on_hook": TaskEntry(
        module="sim.tasks.mug_on_hook",
        scene="sim.envs.mug_on_hook",
        assets="benchmark/run/make_mug_on_hook_assets.py",
        manager_cfg="sim.manager_based.mug_on_hook.env_cfg",
    ),
    "cap_unscrew": TaskEntry(
        module="sim.tasks.cap_unscrew",
        scene="sim.envs.cap_unscrew",
        assets="benchmark/run/make_cap_unscrew_assets.py",
        manager_cfg="sim.manager_based.cap_unscrew.env_cfg",
    ),
    "drawer_latch": TaskEntry(
        module="sim.tasks.drawer_latch",
        scene="sim.envs.drawer_latch",
        assets="benchmark/run/make_drawer_latch_assets.py",
        manager_cfg="sim.manager_based.drawer_latch.env_cfg",
    ),
}


def load_task(task_id: str):
    """-> (TASK object, pinned knobs). Import happens here because the task
    modules import isaaclab, which needs Kit running."""
    entry = TASKS[task_id]
    return importlib.import_module(entry.module).TASK, dict(entry.pins)


def load_scene(task_id: str):
    """-> the task's scene module, without importing its expert. Same reason
    for the late import as `load_task`."""
    return importlib.import_module(TASKS[task_id].scene)
