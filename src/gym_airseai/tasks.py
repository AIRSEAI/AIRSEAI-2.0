"""The task ids this package registers.

Read from the simulation's own registry when it is importable, which is the
case in a checkout. A lerobot process that has only this package installed
falls back to the list below, so `gym.make` still works there.
"""

from __future__ import annotations

__all__ = ["TASK_IDS", "VARIANTS"]

VARIANTS = ("vision", "tacff", "tacrgb")

try:
    from sim.tasks.registry import TASKS as _TASKS

    TASK_IDS: tuple[str, ...] = tuple(sorted(_TASKS))
except ImportError:  # installed without the simulation
    TASK_IDS = (
        "blind_peg_insertion",
        "cap_unscrew",
        "handover",
        "mug_on_hook",
        "sleeve_on_shaft",
        "stack_cube",
        "texture_sort",
    )
