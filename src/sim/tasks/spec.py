"""The `Task` interface: one object that defines a task's episode for both
the expert runner and the evaluator.

A task subclasses `Task` and exposes a module-level `TASK`. The expert runner
drives the whole phase list and records a dataset; the evaluator drives it up
to `handover_phase` (None: t = 0), then hands control to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.criteria.trace import EpisodeTrace, Verdict

__all__ = ["Knob", "Task", "RetryReset"]


@dataclass(frozen=True)
class Knob:
    """A task parameter exposed on the runner CLI. Defaults are the task's
    anchor configuration; the registry may pin a different value (as
    `blind_peg_dim` pins `light_scale`)."""
    default: Any
    help: str
    type: type = float


class RetryReset(Exception):
    """Raised by `reset` when the drawn initial state is unusable (e.g. a
    standing object fell during settling). The runner re-draws."""


class Task:
    # ---- identity ----------------------------------------------------------
    name: str = ""
    # Natural-language instruction recorded into the dataset and given to a
    # language-conditioned policy. It must not name the hidden variable:
    # "put the block in the matching cup", not "put the rough block in the
    # right cup".
    instruction: str = ""
    knobs: dict[str, Knob] = {}
    phases: list[tuple[str, float]] = []        # (name, timeout_s), the expert
    handover_phase: str | None = None           # policy takes over here; None = t=0
    # Phases driven in joint space rather than through IK: for these,
    # `phase_target` returns (joint-name -> radians, grip, None) and the
    # runner calls `Harness.step_joint_toward`. Empty for every task that
    # only ever commands the TCP.
    joint_phases: tuple[str, ...] = ()
    horizon_s: float = 25.0                     # policy control after handover
    report_name: str = ""                       # <task>_scripted.json
    # third-person debug camera (eye, target); None = no scene cam
    scene_cam: tuple | None = None

    # ---- scene -------------------------------------------------------------
    def make_scene_cfg(self, knobs: dict) -> Any:
        raise NotImplementedError

    def apply_collision(self, h) -> None:
        """SDF / triangle / CCD on the task's objects; before `h.start()`."""

    def task_id(self, knobs: dict) -> str:
        return self.name

    # ---- episode -----------------------------------------------------------
    def reset(self, h, rng: np.random.Generator, ep: int, knobs: dict) -> dict:
        """Randomise, settle, select the arm. Returns the episode ctx (must
        include 'side'). May raise RetryReset."""
        raise NotImplementedError

    def signals(self, h, ctx: dict) -> dict:
        """Per-step scalars. Must include 'areas' ({pad: mm2})."""
        return {"areas": h.pad_areas()}

    def phase_target(self, h, ctx: dict, name: str, t_phase: float,
                     sig: dict, knobs: dict):
        """-> (target, grip_m, wrist_quat | None) for the active arm.

        For IK phases `target` is xyz. For names in `joint_phases` it is a
        joint-name -> radians dict and the runner calls `step_joint_toward`.
        """
        raise NotImplementedError

    def phase_done(self, h, ctx: dict, name: str, t_phase: float,
                   sig: dict, target, knobs: dict) -> bool:
        raise NotImplementedError

    def next_phase(self, ctx: dict, name: str) -> str | None:
        """The phase to run after `name` has finished (done or timed out).
        None means the next entry in `phases`. A task that retries -- re-open
        and descend again on a thin grasp -- returns an earlier phase name
        here, and bounds the retries itself in `ctx`."""
        return None

    def handover_ok(self, h, ctx: dict, sig: dict) -> bool:
        """Gate at `handover_phase`: is the state fit to hand to a policy?"""
        return True

    def should_terminate(self, ctx: dict, hist: dict) -> bool:
        """Early stop during policy control (evaluator only)."""
        return False

    # ---- scoring -----------------------------------------------------------
    def build_trace(self, h, ctx: dict, hist: dict) -> EpisodeTrace:
        raise NotImplementedError

    def verdict(self, trace: EpisodeTrace, ctx: dict, knobs: dict) -> Verdict:
        raise NotImplementedError

    def episode_extra(self, ctx: dict, hist: dict) -> dict:
        """Task-specific fields for the episode record."""
        return {}

    def report_extra(self, knobs: dict) -> dict:
        """Task-specific fields for the run report."""
        return {}
