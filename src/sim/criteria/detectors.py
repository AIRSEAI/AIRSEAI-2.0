# SPDX-License-Identifier: Apache-2.0
"""Failure detectors: shared, task-independent, computed from state.

Each detector answers one question about an episode and returns the evidence
(a metrics dict) alongside the answer. Detectors are independent; a task
criterion in `sim.criteria.<task>` decides which apply and in what precedence.

All detectors judge on geometry (contact area, object pose, timing), since
TacSL's force magnitudes are uncalibrated.
"""

from __future__ import annotations

import numpy as np

from sim.criteria.trace import EpisodeTrace

__all__ = ["detect_grasp_failure", "detect_mistiming", "detect_slip"]


def detect_slip(trace: EpisodeTrace, obj: str, pads: tuple[str, ...],
                max_slip_m: float = 0.005, area_mm2: float = 5.0
                ) -> tuple[bool, dict[str, float]]:
    """Did the object move relative to the grasp while held?

    Measured as the object's displacement while at least one pad is in
    load-bearing contact, relative to the grasp frame, so carrying the object
    does not register as slip.
    """
    held = np.zeros(len(trace.t), dtype=bool)
    for p in pads:
        held |= trace.in_contact(p, area_mm2)
    if held.sum() < 2:
        return False, {"slip_m": 0.0, "max_slip_m": max_slip_m,
                       "held_steps": float(held.sum())}

    pos = np.asarray(trace.object_pos[obj], dtype=float)[held]
    # Grasp frame: `extra["grasp_frame_pos"]` when recorded, else the object's
    # first held position.
    ref = trace.extra.get("grasp_frame_pos")
    if ref is not None:
        rel = pos - np.asarray(ref, dtype=float)[held]
    else:
        rel = pos - pos[0]
    slip = float(np.linalg.norm(rel - rel[0], axis=1).max())
    return slip > max_slip_m, {"slip_m": slip, "max_slip_m": max_slip_m,
                               "held_steps": float(held.sum())}


def detect_grasp_failure(trace: EpisodeTrace, pads: tuple[str, ...],
                         min_hold_s: float = 0.5, area_mm2: float = 5.0
                         ) -> tuple[bool, dict[str, float]]:
    """Was there ever a load-bearing contact held for `min_hold_s`?

    Distinct from slip: never closing on the object and closing then losing it
    are different failures.
    """
    if not pads or len(trace.t) < 2:
        return True, {"longest_hold_s": 0.0, "min_hold_s": min_hold_s}
    held = np.zeros(len(trace.t), dtype=bool)
    for p in pads:
        held |= trace.in_contact(p, area_mm2)
    # Each sample owns the interval ahead of it, so n samples at 20 Hz last
    # n/20 s; this keeps the hold length invariant under decimation.
    t = np.asarray(trace.t, dtype=float)
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2]))
    longest, run = 0.0, 0.0
    for h, d in zip(held, dt, strict=True):
        run = run + d if h else 0.0
        longest = max(longest, run)
    return longest < min_hold_s, {"longest_hold_s": float(longest),
                                  "min_hold_s": min_hold_s}


def detect_mistiming(trace: EpisodeTrace, first: str, second: str,
                     max_gap_s: float, area_mm2: float = 5.0
                     ) -> tuple[bool, dict[str, float]]:
    """In a handover, did the release follow the receiving grasp in time?

    `first` is the pad that must *gain* contact (the receiver) and `second` the
    pad that must then *lose* it (the giver). Two ways to fail: the giver lets
    go before the receiver has load (negative gap -- the object falls), or holds
    on far too long (gap beyond `max_gap_s` -- the arms fight). Both are
    mistiming; the sign is in the metrics.
    """
    gain = trace.in_contact(first, area_mm2)
    lose = trace.in_contact(second, area_mm2)
    if not gain.any() or not lose.any():
        return True, {"gap_s": float("nan"), "max_gap_s": max_gap_s}
    t_gain = float(trace.t[np.argmax(gain)])              # first True
    t_release = float(trace.t[len(lose) - 1 - np.argmax(lose[::-1])])  # last True
    gap = t_release - t_gain
    return not (0.0 <= gap <= max_gap_s), {"gap_s": gap, "max_gap_s": max_gap_s}
