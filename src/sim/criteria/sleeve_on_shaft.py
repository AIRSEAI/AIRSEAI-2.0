"""sleeve_on_shaft criterion: the split sleeve ends seated on the shaft.

Reads::

    extra["seat_depth_m"], extra["bore_margin_m"]
    extra["sep_m"], extra["rim_offset_m"], extra["sleeve_tilt_deg"]
"""

from __future__ import annotations

import numpy as np

from sim.criteria.detectors import detect_grasp_failure
from sim.criteria.probes import seat_depth
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.sleeve_on_shaft import DEFAULT_CLEARANCE, MIN_SEAT, SHAFT_HEIGHT, SPLIT_GAP

__all__ = ["seated_on_shaft", "verdict"]

# Angle of the sleeve from upright above which it is not on the shaft, degrees.
MAX_TILT_DEG = 20.0


def seated_on_shaft(pos, quat, *, height: float, shaft_xy, shaft_top_z: float,
                    bore_r: float, max_tilt_deg: float = MAX_TILT_DEG) -> float:
    """How far the sleeve's lower rim is below the shaft's top, zero unless
    it is around the shaft and near upright."""
    return seat_depth(pos, quat, length=height, target_xy=shaft_xy,
                      target_z=shaft_top_z, xy_tol=bore_r, max_tilt_deg=max_tilt_deg)


def verdict(trace: EpisodeTrace, *, min_seat_m: float = MIN_SEAT,
            clearance_m: float = DEFAULT_CLEARANCE, max_split_dev_m: float = 0.001,
            max_offset_extra_m: float = 0.001, max_overseat_m: float = 0.001,
            max_final_tilt_deg: float = 5.0, area_mm2: float = 5.0,
            pads: tuple[str, ...]) -> Verdict:
    """The sleeve ends seated on the shaft, whole, centred and upright.

    Judged on the final sample: rim at least `min_seat_m` below the shaft
    top and not below its base, shells at their rest separation, rim within
    the radial clearance of the shaft axis, tilt within `max_final_tilt_deg`.
    Precedence: never held `GRASP_FAILURE` -> seated, success -> bore shut
    while on the shaft or while held `OVERFORCE` -> `UNFINISHED`.
    The notes name the final-state checks that failed or had no signal.
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
    invalid = _final_state_faults(trace, seat_final, metrics, notes, clearance_m=clearance_m,
                                  max_split_dev_m=max_split_dev_m,
                                  max_offset_extra_m=max_offset_extra_m,
                                  max_overseat_m=max_overseat_m,
                                  max_final_tilt_deg=max_final_tilt_deg)
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics, tuple(notes))
    if seat_final >= min_seat_m and not invalid:
        return Verdict(True, metrics=metrics, notes=tuple(notes))
    if forced_on or shut_while_held:
        return Verdict(False, FailureMode.OVERFORCE, metrics, tuple(notes))
    return Verdict(False, FailureMode.UNFINISHED, metrics, tuple(notes))


def _final_state_faults(trace: EpisodeTrace, seat_final: float, metrics: dict,
                        notes: list[str], *, clearance_m: float, max_split_dev_m: float,
                        max_offset_extra_m: float, max_overseat_m: float,
                        max_final_tilt_deg: float) -> list[str]:
    """Names of the final-state checks that fail; records each measure."""
    faults: list[str] = []
    metrics["max_seat_m"] = SHAFT_HEIGHT + max_overseat_m
    if seat_final > SHAFT_HEIGHT + max_overseat_m:
        faults.append("sunk")
    for key, name, measure, limit in (
            ("sep_m", "split", lambda v: abs(v - SPLIT_GAP), max_split_dev_m),
            ("rim_offset_m", "off_centre", lambda v: v, clearance_m + max_offset_extra_m),
            ("sleeve_tilt_deg", "tilted", lambda v: v, max_final_tilt_deg)):
        series = trace.extra.get(key)
        if series is None:
            notes.append(f"no {key} recorded; {name} check skipped")
            continue
        value = measure(float(np.asarray(series, dtype=float)[-1]))
        metrics[f"final_{name}"] = value
        metrics[f"max_final_{name}"] = limit
        if value > limit:
            faults.append(name)
    if faults:
        notes.append("final state: " + ", ".join(faults))
    return faults
