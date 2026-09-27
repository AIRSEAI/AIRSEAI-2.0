"""Sustained task position completion, with prior grasp evidence."""

from __future__ import annotations

import numpy as np

from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict

SUCCESS_HOLD_S = 1.0
GRASP_HOLD_S = 0.5
CONTACT_AREA_MM2 = 5.0


def sustained_completion(trace: EpisodeTrace, positioned: np.ndarray, *,
                        pads: tuple[str, ...],
                        hold_s: float = SUCCESS_HOLD_S) -> Verdict:
    """Latch completion after position persists for the hold time."""
    t = np.asarray(trace.t, dtype=float)
    if hold_s <= 0 or not pads:
        raise ValueError("Positive hold time and grasp pads are required")
    if len(t) < 2:
        return Verdict(False, FailureMode.UNFINISHED)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Trace timestamps must be finite and strictly increasing")
    positioned = np.asarray(positioned, dtype=bool)
    if positioned.shape != t.shape:
        raise ValueError("Completion signals must share the trace clock")
    areas = np.stack([np.asarray(trace.pad_area[p], dtype=float) for p in pads])
    if areas.shape != (len(pads), len(t)) or not np.isfinite(areas).all():
        raise ValueError("Contact signals must be finite and share the trace clock")
    grasped = np.any(areas > CONTACT_AREA_MM2, axis=0)
    grasp_run = longest_grasp = success_run = longest_success = 0.0
    success_time = None
    # Use observed intervals; the first sample alone contributes no duration.
    for i, dt in enumerate(np.diff(t, prepend=t[0])):
        grasp_run = grasp_run + dt if grasped[i] and grasped[max(0, i - 1)] else 0.0
        longest_grasp = max(longest_grasp, grasp_run)
        ready = positioned[i]
        previous = positioned[max(0, i - 1)]
        success_run = success_run + dt if ready and previous else 0.0
        longest_success = max(longest_success, success_run)
        if (success_time is None and success_run + 1e-9 >= hold_s
                and longest_grasp + 1e-9 >= GRASP_HOLD_S):
            success_time = float(t[i])
    metrics = {"completion_hold_s": longest_success, "required_hold_s": hold_s,
               "grasp_longest_hold_s": longest_grasp, "grasp_required_hold_s": GRASP_HOLD_S,
               "contact_area_mm2": CONTACT_AREA_MM2}
    if success_time is not None:
        metrics["success_time_s"] = success_time
        return Verdict(True, metrics=metrics)
    failure = FailureMode.GRASP_FAILURE if longest_grasp + 1e-9 < GRASP_HOLD_S else FailureMode.UNFINISHED
    return Verdict(False, failure, metrics)
