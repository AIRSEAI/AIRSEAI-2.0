"""Experts: one `sim.tasks.spec.Task` per benchmark task.

Each module defines a task's phases, handover gate, termination test and
verdict, driven by both `benchmark/tasks/run_expert.py` and
`benchmark/eval/evaluate.py`, and ends with a module-level `TASK` that
`sim.tasks.registry` names. Its scene is `sim.envs.<task>`.
"""
