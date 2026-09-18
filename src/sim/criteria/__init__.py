"""Success criteria and failure classification, computed from state, not pixels.

Every criterion is a pure numpy function of a recorded `EpisodeTrace`, called
identically by the expert runner and the policy evaluator, and returns a
`Verdict` whose failure is one of the `FailureMode` taxonomy. No Isaac import.

Thresholds are keyword arguments and are echoed in the verdict's `metrics`,
so traces can be re-judged with other values.
"""

from __future__ import annotations

from sim.criteria.detectors import (
    detect_drop,
    detect_grasp_failure,
    detect_mistiming,
    detect_slip,
)
from sim.criteria.probes import insertion_depth, tilt_deg, tip_position
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict

__all__ = [
    "EpisodeTrace",
    "FailureMode",
    "Verdict",
    "detect_drop",
    "detect_grasp_failure",
    "detect_mistiming",
    "detect_slip",
    "insertion_depth",
    "tilt_deg",
    "tip_position",
]
