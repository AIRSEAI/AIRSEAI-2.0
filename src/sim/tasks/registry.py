"""The task registry: id -> (task module, pinned knobs).

Read by the expert runner, the evaluator and the dataset collector. A task
id may pin a knob, running its module with that knob fixed; the pin
overrides the CLI.
"""

from __future__ import annotations

import importlib

__all__ = ["TASKS", "load_task"]

TASKS: dict[str, tuple[str, dict]] = {
    "stack_cube":            ("sim.tasks.stack_cube", {}),
    "blind_peg_insertion":   ("sim.tasks.blind_peg_insertion", {}),
    "texture_sort":          ("sim.tasks.texture_sort", {}),
    "sleeve_on_shaft":       ("sim.tasks.sleeve_on_shaft", {}),
    "handover":              ("sim.tasks.handover", {}),
    "cap_unscrew":           ("sim.tasks.cap_unscrew", {}),
}


def load_task(task_id: str):
    """-> (TASK object, pinned knobs). Import happens here because the task
    modules import isaaclab, which needs Kit running."""
    module, pins = TASKS[task_id]
    return importlib.import_module(module).TASK, dict(pins)
