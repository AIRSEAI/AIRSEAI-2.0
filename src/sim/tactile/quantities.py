"""Geometric quantities derived from TacSL's per-taxel output.

`contact_area` counts occupied taxels for touching/holding logic;
`quat_apply` / `quat_apply_inverse` mirror isaaclab.utils.math so this module
imports without Kit. Quaternions are (w, x, y, z); lengths are metres unless a
name says `_mm`.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "contact_area",
    "quat_apply",
    "quat_apply_inverse",
]


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` by `quat`. Both broadcast over leading dims; quat is (..., 4)."""
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` by the conjugate of `quat`."""
    conj = torch.cat([quat[..., 0:1], -quat[..., 1:4]], dim=-1)
    return quat_apply(conj, vec)


def contact_area(
    penetration_depth: torch.Tensor,
    pitch_x: float,
    pitch_y: float,
    threshold: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Contact area from the taxel occupancy.

    This is a counting measurement, so it is quantised by the taxel pitch and
    only becomes accurate once a decent number of taxels are active.

    Args:
        penetration_depth: (N, P) SDF penetration per taxel; zero when clear.
        pitch_x: taxel spacing along the pad's first axis, in metres.
        pitch_y: taxel spacing along the pad's second axis, in metres.
        threshold: depth above which a taxel counts as touching.

    Returns:
        Dict with `n_active` (N,), `area_mm2` (N,) and `equiv_radius_mm` (N,).
    """
    active = (penetration_depth > threshold).sum(dim=-1)
    area_m2 = active.to(penetration_depth.dtype) * pitch_x * pitch_y
    return {
        "n_active": active,
        "area_mm2": area_m2 * 1.0e6,
        "equiv_radius_mm": torch.sqrt(area_m2 / math.pi) * 1.0e3,
    }
