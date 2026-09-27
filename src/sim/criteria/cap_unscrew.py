"""cap_unscrew criterion: the cap reaches the tray.

Reads::

    object_pos["cap"], object_pos["bottle"], object_quat["cap"]
    extra["thread_deg"], extra["bottle_deg"]
"""

from __future__ import annotations

import numpy as np

from sim.criteria.detectors import detect_grasp_failure
from sim.criteria.probes import axis_from_quat
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.cap_unscrew import BOTTLE_H, CAP_H, CAP_R, TRAY_HEIGHT, TRAY_INNER_R

__all__ = ["cap_centre", "in_tray", "on_bottle", "verdict"]

# How far the cap's centre may sit off the bottle's axis and still count as
# on the bottle, metres: inside the neck's radius.
ON_BOTTLE_XY_TOL = 0.010

# The bottle's turn, over the thread's, above which the holding hand let it
# spin.
SPUN_RATIO = 1.0


def cap_centre(pos, quat) -> np.ndarray:
    """World position of the cap's centre: its origin is on the skirt's rim."""
    return np.asarray(pos, dtype=float) + axis_from_quat(quat, (0.0, 0.0, CAP_H / 2.0))


def in_tray(
    centre, tray_pos, *, tray_r: float = TRAY_INNER_R, max_above_m: float = TRAY_HEIGHT + CAP_R
) -> bool:
    """Is the cap's centre inside the tray's bore and no higher above its base
    than a cap lying in it on its side?"""
    c, t = np.asarray(centre, dtype=float), np.asarray(tray_pos, dtype=float)
    return bool(np.linalg.norm(c[:2] - t[:2]) <= tray_r and c[2] - t[2] <= max_above_m)


def on_bottle(centre, bottle_pos, *, xy_tol: float = ON_BOTTLE_XY_TOL) -> bool:
    """Is the cap's centre on the bottle's axis, level with the neck or above?"""
    c, b = np.asarray(centre, dtype=float), np.asarray(bottle_pos, dtype=float)
    return bool(np.linalg.norm(c[:2] - b[:2]) <= xy_tol and c[2] >= b[2] + BOTTLE_H - CAP_H)


def verdict(
    trace: EpisodeTrace,
    *,
    tray_pos,
    tray_r: float = TRAY_INNER_R,
    max_above_m: float = TRAY_HEIGHT + CAP_R,
    area_mm2: float = 5.0,
    turner_pads: tuple[str, ...],
    holder_pads: tuple[str, ...],
) -> Verdict:
    """The cap reaches the tray at some point in the episode.

    Judged over the whole episode, not on the final sample: the task is
    turning the cap off the bottle, which a cap in the tray has been through
    whether or not it stays there. Precedence: the turning hand never held
    `GRASP_FAILURE` -> reached the tray, success -> ends on the bottle: the
    bottle turned further than the thread `COORDINATION`, else `UNFINISHED`
    -> anywhere else `SLIP`.
    """
    pos = np.asarray(trace.object_pos["cap"], dtype=float)
    quat = np.asarray(trace.object_quat["cap"], dtype=float)
    bottle = np.asarray(trace.object_pos["bottle"], dtype=float)
    n = len(trace.t)
    thread = np.asarray(trace.extra.get("thread_deg", np.zeros(n)), dtype=float)
    spun = np.asarray(trace.extra.get("bottle_deg", np.zeros(n)), dtype=float)
    never_held, m_grasp = detect_grasp_failure(trace, turner_pads, area_mm2=area_mm2)
    t = np.asarray(tray_pos, dtype=float)
    centres = np.asarray([cap_centre(p, q) for p, q in zip(pos, quat, strict=True)])
    centre = centres[-1]
    reached = [in_tray(c, t, tray_r=tray_r, max_above_m=max_above_m) for c in centres]
    released = all(
        float(np.asarray(trace.pad_area[p], dtype=float)[-1]) <= area_mm2
        for p in turner_pads + holder_pads
        if p in trace.pad_area
    )
    metrics = {
        **m_grasp,
        "cap_to_tray_xy_m": float(np.linalg.norm(centre[:2] - t[:2])),
        "closest_to_tray_xy_m": float(np.linalg.norm(centres[:, :2] - t[:2], axis=1).min()) if n else 0.0,
        "cap_above_tray_m": float(centre[2] - t[2]),
        "tray_r_m": tray_r,
        "max_above_m": max_above_m,
        "in_tray_ever": float(any(reached)),
        "in_tray_final": float(reached[-1]) if n else 0.0,
        "thread_turn_deg": float(thread[-1]) if n else 0.0,
        "peak_thread_turn_deg": float(thread.max()) if n else 0.0,
        "bottle_turn_deg": float(np.abs(spun).max()) if n else 0.0,
        "released": float(released),
    }
    if never_held:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if any(reached):
        return Verdict(True, metrics=metrics)
    if on_bottle(centre, bottle[-1]):
        if metrics["bottle_turn_deg"] > SPUN_RATIO * max(metrics["peak_thread_turn_deg"], 1.0):
            return Verdict(False, FailureMode.COORDINATION, metrics)
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(False, FailureMode.SLIP, metrics)
