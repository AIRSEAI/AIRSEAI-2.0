"""drawer_latch criterion: the drawer stays open after release.

Reads::

    extra["extension_m"], extra["hand_clearance_m"]
"""

from __future__ import annotations

import numpy as np

from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.drawer_latch import LATCH_TOLERANCE

__all__ = ["verdict"]

RELEASE_HOLD_S = 1.0
CONTACT_AREA_MM2 = 5.0
GRASP_HOLD_S = 0.30
HAND_CLEARANCE_M = 0.060
POSITION_TOLERANCE_M = LATCH_TOLERANCE + 0.002


def verdict(
    trace: EpisodeTrace,
    *,
    detent_m: float,
    pads: tuple[str, ...],
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    release_hold_s: float = RELEASE_HOLD_S,
    contact_area_mm2: float = CONTACT_AREA_MM2,
    grasp_hold_s: float = GRASP_HOLD_S,
    hand_clearance_m: float = HAND_CLEARANCE_M,
) -> Verdict:
    """Require a grasp and a final interval of unsupported opening.

    Precedence: GRASP_FAILURE, UNFINISHED.
    """
    t = np.asarray(trace.t, dtype=float)
    x = np.asarray(trace.extra["extension_m"], dtype=float)
    if len(t) < 2:
        return Verdict(False, FailureMode.UNFINISHED)
    if x.shape != t.shape or not np.isfinite(x).all():
        raise ValueError("Drawer extension must be finite and share the trace clock")
    clearance = np.asarray(trace.extra["hand_clearance_m"], dtype=float)
    if clearance.shape != t.shape or not np.isfinite(clearance).all():
        raise ValueError("Hand clearance must be finite and share the trace clock")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Trace timestamps must be strictly increasing")
    area = np.stack([np.asarray(trace.pad_area[p], dtype=float) for p in pads])
    if area.shape != (len(pads), len(t)) or not np.isfinite(area).all():
        raise ValueError("Pad areas must be finite and share the trace clock")
    touching = np.any(area > contact_area_mm2, axis=0)
    positioned = np.abs(x - detent_m) <= position_tolerance_m
    clear = clearance >= hand_clearance_m
    grasp_run = release_run = longest_grasp = longest_release = 0.0
    success_time = None
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        grasp_run = grasp_run + dt if touching[i] and touching[i - 1] else 0.0
        longest_grasp = max(longest_grasp, grasp_run)
        released = (
            positioned[i]
            and positioned[i - 1]
            and clear[i]
            and clear[i - 1]
            and not touching[i]
            and not touching[i - 1]
        )
        release_run = release_run + dt if released else 0.0
        longest_release = max(longest_release, release_run)
        if success_time is None and longest_grasp >= grasp_hold_s and release_run >= release_hold_s:
            success_time = float(t[i])
    metrics = {
        "final_extension_m": float(x[-1]),
        "max_extension_m": float(np.max(x)),
        "peak_to_final_retraction_m": float(np.max(x) - x[-1]),
        "never_pulled_5mm": bool(np.max(x) < 0.005),
        "detent_m": float(detent_m),
        "position_tolerance_m": float(position_tolerance_m),
        "released_hold_s": longest_release,
        "terminal_released_hold_s": release_run,
        "final_hand_clearance_m": float(clearance[-1]),
        "required_release_hold_s": float(release_hold_s),
        "grasp_hold_s": longest_grasp,
    }
    if success_time is not None and release_run >= release_hold_s:
        metrics["success_time_s"] = success_time
        return Verdict(True, metrics=metrics)
    mode = FailureMode.GRASP_FAILURE if longest_grasp < grasp_hold_s else FailureMode.UNFINISHED
    return Verdict(False, mode, metrics)
