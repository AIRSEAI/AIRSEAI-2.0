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

__all__ = [
    "Knob",
    "Task",
    "RetryReset",
]


@dataclass(frozen=True)
class Knob:
    """A task parameter exposed on the runner CLI. Defaults are the task's
    anchor configuration; the registry may pin a different value for a
    variant, and the pin overrides the CLI."""

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
    phases: list[tuple[str, float]] = []  # (name, timeout_s), the expert
    handover_phase: str | None = None  # policy takes over here; None = t=0
    # Phases driven in joint space rather than through IK: for these,
    # `phase_target` returns (joint-name -> radians, grip, None) and the
    # runner calls `Harness.step_joint_toward`. Empty for every task that
    # only ever commands the TCP.
    joint_phases: tuple[str, ...] = ()
    horizon_s: float = 25.0  # policy control after handover
    restore_contact_clearance_m: float = 0.0  # lift before restore to clear old contact anchors
    report_name: str = ""  # <task>_scripted.json
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
    def sample_init_state(self, robot: dict, rng, state_id: int, knobs: dict) -> dict:
        """Draw a local-coordinate start without advancing physics."""
        raise NotImplementedError(f"{self.name} has no vector reset sampler")

    def begin_restored_episode(self, h, state: dict, knobs: dict) -> dict:
        """Build controller context after the scheduler completes a reset."""
        raise NotImplementedError(f"{self.name} has no vector episode context")

    def validate_init_state(self, state: dict) -> None:
        """Validate task-specific fields before any environment consumes the file."""

    def reset_seed_index(self, episode: int) -> int:
        """Random stream index for a global episode ID."""
        return episode

    def prepare_expert(self, ctx: dict, seed: int, episode: int) -> dict:
        """Configure expert controls after reset and return recording metadata."""
        return {}

    def reset(self, h, rng: np.random.Generator, ep: int, knobs: dict) -> dict:
        """Randomise, settle, select the arm. Returns the episode ctx (must
        include 'side'). May raise RetryReset."""
        raise NotImplementedError

    def signals(self, h, ctx: dict) -> dict:
        """Per-step scalars. Must include 'areas' ({pad: mm2})."""
        return {"areas": h.pad_areas()}

    def phase_target(self, h, ctx: dict, name: str, t_phase: float, sig: dict, knobs: dict):
        """-> (target, grip_m, wrist_quat | None) for the active arm.

        For IK phases `target` is xyz. For names in `joint_phases` it is a
        joint-name -> radians dict and the runner calls `step_joint_toward`.
        """
        raise NotImplementedError

    def phase_done(self, h, ctx: dict, name: str, t_phase: float, sig: dict, target, knobs: dict) -> bool:
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

    completion_on_success: bool = False

    def policy_should_terminate(self, h, ctx: dict, hist: dict, knobs: dict) -> bool:
        """Use the task verdict for completion-enabled tasks."""
        if self.completion_on_success:
            if len(hist["t"]) < 2:
                return False
            return self.verdict(self.build_trace(h, ctx, hist), ctx, knobs).success
        return self.should_terminate(ctx, hist)

    def should_terminate(self, ctx: dict, hist: dict) -> bool:
        """Early stop during policy control (evaluator only)."""
        return False

    def init_state_objects(self) -> tuple[str, ...]:
        """The scene objects whose settled pose defines the start. Objects
        parked out of the way are included: where they are parked matters."""
        raise NotImplementedError

    def snapshot_init_state(self, h, ctx: dict) -> dict:
        """-> what `reset` drew and cannot be measured back off the scene.

        The robot and the objects are recorded by the caller; this returns
        only the rest ('task_vars'), such as which arm was picked or an
        offset applied later in the episode.
        """
        raise NotImplementedError

    def restore_init_state(self, h, state: dict, knobs: dict) -> dict:
        """Put `state` back and return a ctx equivalent to `reset`'s.

        The scene is already restored when this is called; what remains is
        the task's own settling and the ctx it would have built.
        """
        raise NotImplementedError

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
