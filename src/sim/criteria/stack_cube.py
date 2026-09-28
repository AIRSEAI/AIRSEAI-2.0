"""Sustained stack completion. Reads object_pos cube_a and cube_b."""

from __future__ import annotations

import numpy as np

from sim.criteria.completion import SUCCESS_HOLD_S, sustained_completion
from sim.criteria.trace import EpisodeTrace, Verdict
from sim.dims.stack_cube import CUBE_SIZE

__all__ = ["verdict"]


def verdict(
    trace: EpisodeTrace,
    *,
    cube_size: float = CUBE_SIZE,
    pads: tuple[str, ...],
    stack_hold_s: float = SUCCESS_HOLD_S,
) -> Verdict:
    """Latch sustained stacking; prior grasp is required before completion."""
    a = np.asarray(trace.object_pos["cube_a"], dtype=float)
    b = np.asarray(trace.object_pos["cube_b"], dtype=float)
    xy = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
    z = np.abs(a[:, 2] - b[:, 2] - cube_size)
    result = sustained_completion(
        trace, (xy < cube_size / 2) & (z < cube_size / 2), pads=pads, hold_s=stack_hold_s
    )
    if len(a):
        result.metrics.update(xy_err_m=float(xy[-1]), z_err_m=float(z[-1]), cube_size_m=cube_size)
    return result
