"""stack_cube criterion: cube A ends stacked on cube B.

Reads::

    object_pos["cube_a"], object_pos["cube_b"]
"""

from __future__ import annotations

import numpy as np

from sim.criteria.detectors import detect_grasp_failure
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.stack_cube import CUBE_SIZE

__all__ = ["verdict"]


def verdict(trace: EpisodeTrace, *, cube_size: float = CUBE_SIZE,
            area_mm2: float = 5.0, pads: tuple[str, ...]) -> Verdict:
    """Cube A stacked on cube B.

    A must end within half a cube of B in xy and one cube up, and the pads
    must have held a load-bearing contact during the carry.
    """
    a = np.asarray(trace.object_pos["cube_a"], dtype=float)[-1]
    b = np.asarray(trace.object_pos["cube_b"], dtype=float)[-1]
    xy_err = float(np.linalg.norm(a[:2] - b[:2]))
    z_err = float(abs(a[2] - (b[2] + cube_size)))
    stacked = xy_err < cube_size / 2 and z_err < cube_size / 2
    dropped, m_grasp = detect_grasp_failure(trace, pads, area_mm2=area_mm2)
    metrics = {"xy_err_m": xy_err, "z_err_m": z_err, "cube_size_m": cube_size,
               **m_grasp}
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if not stacked:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)
