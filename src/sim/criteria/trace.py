"""The recorded quantities a verdict is computed from, and the verdict itself.

`EpisodeTrace` holds privileged simulator state a policy does not observe
(object poses, per-pad contact areas, phase boundaries), separate from the
dataset. Every field is time-major on a shared clock `t`, so criteria can
slice by time and give the same verdict on a decimated trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = ["EpisodeTrace", "FailureMode", "Verdict"]


# `str, Enum` rather than `StrEnum`, which needs Python 3.11+.
class FailureMode(str, Enum):  # noqa: UP042
    """The failure taxonomy shared by every task.

    Flat and closed, so failure distributions compare across tasks. Finer
    detail goes in the verdict's `metrics`, not in a new mode.
    """

    NONE = "none"                    # success
    SLIP = "slip"                    # object moved within the grasp
    OVERFORCE = "overforce"          # force/torque budget exceeded
    MISTIMING = "mistiming"          # phases correct, temporal relation wrong
    COORDINATION = "coordination"    # two arms fought each other, or one lagged
    GRASP_FAILURE = "grasp_failure"  # never achieved a load-bearing contact
    UNFINISHED = "unfinished"        # ran out of time in a recoverable state


@dataclass
class EpisodeTrace:
    """Privileged per-step state for one episode.

    Parameters
    ----------
    t
        (T,) seconds since episode start, strictly increasing.
    pad_area
        pad name -> (T,) contact area in mm^2. Area is the load-bearing test
        that force alone fails: a pad can read force from a passing brush.
    object_pos
        object name -> (T, 3) world position, metres.
    object_quat
        object name -> (T, 4) world orientation, (w, x, y, z).
    phase
        (T,) integer phase index from the controller's state machine, or None
        for a learned policy (criteria that need phases must say so).
    phase_names
        Index -> name, for readable mistiming reports.
    extra
        Task-specific scalars over time (applied torque, insertion depth, ...).
        Anything a single task needs and no other does goes here rather than
        growing this dataclass.
    """

    t: np.ndarray
    pad_area: dict[str, np.ndarray] = field(default_factory=dict)
    object_pos: dict[str, np.ndarray] = field(default_factory=dict)
    object_quat: dict[str, np.ndarray] = field(default_factory=dict)
    phase: np.ndarray | None = None
    phase_names: tuple[str, ...] = ()
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    # -- derived views -----------------------------------------------------

    def in_contact(self, pad: str, area_mm2: float = 5.0) -> np.ndarray:
        """(T,) boolean, load-bearing contact on this pad.

        5 mm^2 is the default bar for calling a grasp closed; it is a
        threshold, so it is an argument.
        """
        return np.asarray(self.pad_area[pad], dtype=float) > area_mm2

    def window(self, t0: float, t1: float) -> np.ndarray:
        """(T,) boolean mask for the half-open time window [t0, t1)."""
        t = np.asarray(self.t, dtype=float)
        return (t >= t0) & (t < t1)

    def phase_window(self, name: str) -> np.ndarray:
        """(T,) boolean mask for the steps spent in a named phase."""
        if self.phase is None or name not in self.phase_names:
            return np.zeros(len(self.t), dtype=bool)
        return np.asarray(self.phase) == self.phase_names.index(name)

    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0


@dataclass
class Verdict:
    """The outcome of one episode under one criterion.

    `metrics` carries both the measured quantities and the thresholds they were
    judged against, so a report is self-describing and a sensitivity analysis
    can tell which threshold produced which verdict.
    """

    success: bool
    failure: FailureMode = FailureMode.NONE
    metrics: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.success and self.failure is not FailureMode.NONE:
            raise ValueError(f"success verdict carries failure {self.failure}")
        if not self.success and self.failure is FailureMode.NONE:
            raise ValueError("failure verdict must name a mode")

    def as_dict(self) -> dict:
        return {"success": bool(self.success),
                "failure": self.failure.value,
                "metrics": {k: float(v) for k, v in self.metrics.items()},
                "notes": list(self.notes)}
