"""blind_peg_insertion criterion: the peg ends in the occluded hole.

Reads::

    extra["insertion_depth_m"], extra["peg_tilt_deg"]
    pad_area                                    the carrying gripper's pads
"""

from __future__ import annotations

import numpy as np

from sim.criteria.detectors import detect_grasp_failure
from sim.criteria.probes import seat_depth
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.blind_peg_insertion import MIN_DEPTH

__all__ = ["insertion_depth", "verdict"]

# Angle of the peg from upright above which a grasp counts as lost, degrees.
MAX_TILT_DEG = 20.0


def insertion_depth(pos, quat, *, length: float, hole_xy, mouth_z: float,
                    hole_radius: float, xy_margin: float = 0.002,
                    max_tilt_deg: float = MAX_TILT_DEG) -> float:
    """How far the tip is *into the hole*, zero unless it actually is.

    A peg dropped on the table beside the hole is below the mouth but neither
    over it nor upright, so it reads zero.
    """
    return seat_depth(pos, quat, length=length, target_xy=hole_xy, target_z=mouth_z,
                      xy_tol=hole_radius + xy_margin, max_tilt_deg=max_tilt_deg)


def verdict(trace: EpisodeTrace, *, min_depth_m: float = MIN_DEPTH,
            max_tilt_deg: float = MAX_TILT_DEG, pads: tuple[str, ...]) -> Verdict:
    """Insert into an occluded hole under injected pose error.

    Depth comes from `extra["insertion_depth_m"]`, zero whenever the peg is
    not in the hole (`insertion_depth`). Judged on the final depth, not the
    deepest moment.

    The jam test is on tilt, not lateral force: tilt separates a peg still
    upright in the grasp from one that has rotated out of it.
    """
    depth = np.asarray(trace.extra.get("insertion_depth_m", [0.0]), dtype=float)
    tilt = np.asarray(trace.extra.get("peg_tilt_deg", [0.0]), dtype=float)
    dropped, m_grasp = detect_grasp_failure(trace, pads)
    metrics = {"depth_m": float(depth[-1]), "peak_depth_m": float(depth.max()),
               "min_depth_m": min_depth_m,
               "peak_tilt_deg": float(tilt.max()), "max_tilt_deg": max_tilt_deg,
               **m_grasp}
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if float(tilt.max()) > max_tilt_deg:
        return Verdict(False, FailureMode.SLIP, metrics)
    if float(depth[-1]) < min_depth_m:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)
