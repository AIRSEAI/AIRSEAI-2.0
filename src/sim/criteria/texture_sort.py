"""texture_sort criterion: the puck ends in the bin its surface calls for."""

from __future__ import annotations

from sim.criteria.detectors import detect_grasp_failure
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict

__all__ = ["verdict"]


def verdict(trace: EpisodeTrace, *, chosen_bin: int, correct_bin: int,
            pads: tuple[str, ...]) -> Verdict:
    """Classify by surface, place in the matching bin.

    `chosen_bin` is the bin the puck ended in, -1 for neither. A wrong bin
    reports UNFINISHED; the failure taxonomy is closed.
    """
    dropped, m_grasp = detect_grasp_failure(trace, pads)
    metrics = {"chosen_bin": float(chosen_bin), "correct_bin": float(correct_bin),
               **m_grasp}
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if chosen_bin != correct_bin:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)
