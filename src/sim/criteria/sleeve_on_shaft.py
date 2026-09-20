"""sleeve_on_shaft criterion: the split sleeve ends seated on the shaft.

Reads::

    extra["seat_depth_m"], extra["bore_margin_m"]
    pad_area                                    the carrying gripper's pads
"""

from __future__ import annotations

import numpy as np

from sim.criteria.detectors import detect_grasp_failure
from sim.criteria.probes import seat_depth
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.sleeve_on_shaft import MIN_SEAT

__all__ = ["seated_on_shaft", "verdict"]

# Angle of the sleeve from upright above which it is not on the shaft, degrees.
MAX_TILT_DEG = 20.0


def seated_on_shaft(pos, quat, *, height: float, shaft_xy, shaft_top_z: float,
                    bore_r: float, max_tilt_deg: float = MAX_TILT_DEG) -> float:
    """How far the sleeve's lower rim is below the shaft's top, zero unless it
    is on the shaft.

    The reverse of an insertion: the bore travels onto the shaft. The rim must
    be below the shaft top, its centre within the bore radius of the shaft
    axis, and the sleeve near upright.
    """
    return seat_depth(pos, quat, length=height, target_xy=shaft_xy,
                      target_z=shaft_top_z, xy_tol=bore_r, max_tilt_deg=max_tilt_deg)


def verdict(trace: EpisodeTrace, *, min_seat_m: float = MIN_SEAT,
            area_mm2: float = 5.0, pads: tuple[str, ...]) -> Verdict:
    """A split sleeve, carried in the gripper, ends up seated on the shaft.

    `extra["seat_depth_m"]` is the sleeve's lower rim below the shaft top,
    zero unless the sleeve is around the shaft and near upright
    (`seated_on_shaft`). Judged on the FINAL value: a sleeve that went on and
    was pulled back off has not been seated.

    `extra["bore_margin_m"]` is how much wider the bore is than the shaft
    along the squeeze axis: split gap plus twice the clearance, minus how far
    the grip has closed the shells. Negative means the grip shut the bore
    below the shaft diameter.

    In simulation the shaft's chamfer can wedge a compliant grip back open, so
    any sample with the sleeve on the shaft (seat depth above zero) and a
    negative bore margin is `OVERFORCE`, whatever the final seat.

    Precedence: never held `GRASP_FAILURE` -> forced on through a shut bore
    `OVERFORCE` -> seated, success -> not seated with the bore shut while the
    pads held the sleeve `OVERFORCE` -> otherwise `UNFINISHED`.
    """
    dropped, m_grasp = detect_grasp_failure(trace, pads, area_mm2=area_mm2)
    notes: list[str] = []
    seat = trace.extra.get("seat_depth_m")
    margin = trace.extra.get("bore_margin_m")
    metrics = {**m_grasp, "min_seat_m": min_seat_m}
    if seat is None:
        notes.append("no seat_depth_m recorded; cannot judge a seat")
        seat_arr = np.zeros(len(trace.t))
    else:
        seat_arr = np.asarray(seat, dtype=float)
    seat_final = float(seat_arr[-1])
    metrics["seat_depth_m"] = seat_final
    forced_on = shut_while_held = False
    if margin is not None:
        margin_arr = np.asarray(margin, dtype=float)
        metrics["min_bore_margin_m"] = float(np.min(margin_arr))
        on_shaft = seat_arr > 0.0
        if on_shaft.any():
            metrics["min_bore_margin_on_shaft_m"] = float(np.min(margin_arr[on_shaft]))
            forced_on = metrics["min_bore_margin_on_shaft_m"] < 0.0
        held = np.zeros(len(trace.t), dtype=bool)
        for p in pads:
            held |= trace.in_contact(p, area_mm2)
        if held.any():
            metrics["min_bore_margin_held_m"] = float(np.min(margin_arr[held]))
            shut_while_held = metrics["min_bore_margin_held_m"] < 0.0
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics, tuple(notes))
    if forced_on:
        return Verdict(False, FailureMode.OVERFORCE, metrics, tuple(notes))
    if seat_final >= min_seat_m:
        return Verdict(True, metrics=metrics, notes=tuple(notes))
    if shut_while_held:
        return Verdict(False, FailureMode.OVERFORCE, metrics, tuple(notes))
    return Verdict(False, FailureMode.UNFINISHED, metrics, tuple(notes))
