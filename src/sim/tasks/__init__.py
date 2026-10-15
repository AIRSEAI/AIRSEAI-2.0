# SPDX-License-Identifier: Apache-2.0
"""Experts: one `sim.tasks.spec.Task` per benchmark task.

Each module defines a task's phases, handover gate, termination test and
verdict, driven by both the expert runner and the evaluator, and ends with a
module-level `TASK` that `sim.tasks.registry` names. Its scene is
`sim.envs.<task>`.
"""
