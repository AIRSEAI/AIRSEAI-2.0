"""Optional per-joint bounds for human VR commands in simulator units (radians)."""

from __future__ import annotations

import math

import torch


def bound_joint_target(
    proposed: torch.Tensor,
    previous: torch.Tensor,
    soft_limits: torch.Tensor,
    max_step_rad: float,
) -> tuple[torch.Tensor, bool]:
    """Bound finite targets; reject invalid inputs or an out-of-limits previous command."""
    if not math.isfinite(max_step_rad) or max_step_rad <= 0:
        raise ValueError("max_step_rad must be finite and positive")
    if proposed.shape != previous.shape or soft_limits.shape != (*proposed.shape, 2):
        raise ValueError("joint target and soft limit shapes disagree")

    lower, upper = soft_limits[..., 0], soft_limits[..., 1]
    valid = (
        torch.isfinite(proposed).all()
        & torch.isfinite(previous).all()
        & torch.isfinite(soft_limits).all()
        & (lower <= upper).all()
        & (previous >= lower).all()
        & (previous <= upper).all()
    )
    if not bool(valid.item()):
        return previous.clone(), False

    within_limits = torch.clamp(proposed, min=lower, max=upper)
    delta = torch.clamp(within_limits - previous, min=-max_step_rad, max=max_step_rad)
    bounded = torch.clamp(previous + delta, min=lower, max=upper)
    return bounded, True
