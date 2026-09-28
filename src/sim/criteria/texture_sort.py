"""Sustained puck placement. Reads object_pos target (mesh base position)."""

from __future__ import annotations

import numpy as np

from sim.criteria.completion import SUCCESS_HOLD_S, sustained_completion
from sim.criteria.trace import EpisodeTrace, Verdict
from sim.dims.texture_sort import CUP_INNER_R

__all__ = ["verdict"]


def verdict(
    trace: EpisodeTrace,
    *,
    bin_xy: np.ndarray,
    correct_bin: int,
    pads: tuple[str, ...],
    hold_s: float = SUCCESS_HOLD_S,
) -> Verdict:
    """Latch sustained XY placement in the correct cup with prior grasp."""
    pos = np.asarray(trace.object_pos["target"], dtype=float)
    distances = np.linalg.norm(pos[:, None, :2] - np.asarray(bin_xy)[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    positioned = (nearest == correct_bin) & (distances[:, correct_bin] < CUP_INNER_R)
    result = sustained_completion(trace, positioned, pads=pads, hold_s=hold_s)
    if len(pos):
        closest = int(nearest[-1])
        chosen = closest if distances[-1, closest] < CUP_INNER_R else -1
        result.metrics.update(
            chosen_bin=float(chosen), correct_bin=float(correct_bin), bin_radius_m=CUP_INNER_R
        )
    return result
