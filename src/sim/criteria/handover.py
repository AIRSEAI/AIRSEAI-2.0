# SPDX-License-Identifier: Apache-2.0
"""handover criterion: the sleeve changes hands and is seated over the key.

Reads::

    object_pos["sleeve"]
    extra["seat_depth_m"], extra["gap_m"], extra["over_target"]
"""

from __future__ import annotations

import numpy as np

from sim.criteria.detectors import detect_grasp_failure, detect_mistiming
from sim.criteria.probes import seat_depth
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.handover import DEFAULT_TARGET_BORE, MIN_SEAT, gap_window

__all__ = ["seated_in_cup", "verdict"]

# Angle of the sleeve from upright above which it is not standing in the cup,
# degrees.
MAX_TILT_DEG = 10.0


def seated_in_cup(pos, quat, *, length: float, cup_xy, rim_z: float, xy_tol: float,
                  max_tilt_deg: float = MAX_TILT_DEG) -> float:
    """How far the sleeve's lower rim is below a cup's rim, zero unless it
    stands centred in the cup."""
    return seat_depth(pos, quat, length=length, target_xy=cup_xy, target_z=rim_z,
                      xy_tol=xy_tol, max_tilt_deg=max_tilt_deg)


def _side_area(trace: EpisodeTrace, pads: tuple[str, ...]) -> np.ndarray:
    """(T,) the smaller of a gripper's two pad areas: both pads must bear
    load for the gripper to hold the sleeve."""
    present = [np.asarray(trace.pad_area[p], dtype=float) for p in pads if p in trace.pad_area]
    return np.min(present, axis=0) if present else np.zeros(len(trace.t))


def verdict(trace: EpisodeTrace, *, bore_m: float = DEFAULT_TARGET_BORE,
            min_seat_m: float = MIN_SEAT, drop_height_m: float = 0.05,
            area_mm2: float = 5.0, giver_pads: tuple[str, ...],
            receiver_pads: tuple[str, ...]) -> Verdict:
    """The sleeve ends seated over the target's key and let go.

    Judged on the final sample. Precedence: giver never held `GRASP_FAILURE`
    -> seated and let go, success -> fell `COORDINATION` -> gap below
    `gap_window(bore_m)` `OVERFORCE` -> above it `COORDINATION` ->
    `UNFINISHED`. The gap is the median while the receiver holds the sleeve
    over the target; the release timing is reported as `gap_s`, not judged.
    """
    n = len(trace.t)
    sides = EpisodeTrace(t=trace.t, pad_area={"giver": _side_area(trace, giver_pads),
                                              "receiver": _side_area(trace, receiver_pads)})
    never_held, m_grasp = detect_grasp_failure(sides, ("giver",), area_mm2=area_mm2)
    _, m_time = detect_mistiming(sides, "receiver", "giver", float("inf"), area_mm2)
    z = np.asarray(trace.object_pos["sleeve"], dtype=float)[:, 2]
    fell = bool(z.min() < z[0] - drop_height_m)
    gap = np.asarray(trace.extra.get("gap_m", np.zeros(n)), dtype=float)
    over = np.asarray(trace.extra.get("over_target", np.zeros(n)), dtype=float) > 0.5
    seat = np.asarray(trace.extra.get("seat_depth_m", np.zeros(n)), dtype=float)
    released = all(float(np.asarray(a, dtype=float)[-1]) <= area_mm2
                   for a in trace.pad_area.values())
    lo, hi = gap_window(bore_m)
    held = sides.in_contact("receiver", area_mm2)
    last_over = int(np.flatnonzero(over)[-1]) if over.any() else -1
    held_over = over & held
    if held_over.any():
        gap_at_target = float(np.median(gap[held_over]))
    else:
        gap_at_target = float(gap[last_over]) if last_over >= 0 else float("nan")
    metrics = {**m_time, **m_grasp,
               "sleeve_drop_m": float(z[0] - z.min()), "drop_height_m": drop_height_m,
               "gap_at_target_m": gap_at_target, "gap_lo_m": lo, "gap_hi_m": hi,
               "gap_final_m": float(gap[-1]),
               "seat_depth_m": float(seat[-1]), "peak_seat_depth_m": float(seat.max()),
               "min_seat_m": min_seat_m, "released": float(released)}
    if never_held:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if released and float(seat[-1]) >= min_seat_m:
        return Verdict(True, metrics=metrics)
    if fell:
        return Verdict(False, FailureMode.COORDINATION, metrics)
    # A sleeve never brought over the target has no gap to judge: NaN fails
    # both comparisons.
    if gap_at_target < lo:
        return Verdict(False, FailureMode.OVERFORCE, metrics)
    if gap_at_target > hi:
        return Verdict(False, FailureMode.COORDINATION, metrics)
    return Verdict(False, FailureMode.UNFINISHED, metrics)
