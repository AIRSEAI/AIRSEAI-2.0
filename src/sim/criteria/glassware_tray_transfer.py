"""Check final tray placement and the standing state of all three cups."""

from __future__ import annotations

import numpy as np

from sim.criteria.probes import tilt_deg
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.glassware_tray_transfer import (
    CUP_NAMES,
    CUP_RADIUS,
    RIM_THICKNESS,
    TRAY_LENGTH,
    TRAY_THICKNESS,
    TRAY_WIDTH,
)

__all__ = ["cup_standing_on_tray", "verdict"]

GOAL_XY_TOL = 0.020
GOAL_Z_TOL = 0.012
MAX_TRAY_TILT_DEG = 8.0
MAX_CUP_TILT_DEG = 15.0
CUP_Z_TOL = 0.010
HOLD_S = 1.0
CONTACT_AREA_MM2 = 5.0


def cup_standing_on_tray(cup_pos, cup_quat, tray_pos, tray_quat):
    """Check that the upright cup base rests within the tray deck."""
    p = np.asarray(cup_pos, dtype=float)
    tray = np.asarray(tray_pos, dtype=float)
    w, x, y, z = np.asarray(tray_quat, dtype=float)
    q_inv = np.array([w, -x, -y, -z])
    v = p - tray
    t = 2.0 * np.cross(q_inv[1:], v)
    local = v + q_inv[0] * t + np.cross(q_inv[1:], t)
    standing = (
        abs(local[0]) <= TRAY_LENGTH / 2 - RIM_THICKNESS - CUP_RADIUS
        and abs(local[1]) <= TRAY_WIDTH / 2 - RIM_THICKNESS - CUP_RADIUS
        and abs(local[2] - TRAY_THICKNESS) <= CUP_Z_TOL
        and tilt_deg(cup_quat) <= MAX_CUP_TILT_DEG
    )
    return bool(standing), local


def verdict(
    trace: EpisodeTrace,
    *,
    goal_xy,
    goal_z,
    support_size,
    right_pads: tuple[str, ...],
    left_pads: tuple[str, ...],
    goal_xy_tol: float = GOAL_XY_TOL,
    goal_z_tol: float = GOAL_Z_TOL,
    max_tilt_deg: float = MAX_TRAY_TILT_DEG,
    hold_s: float = HOLD_S,
) -> Verdict:
    """Judge sustained placement with all cups standing after release."""
    t = np.asarray(trace.t, dtype=float)
    if len(t) < 2:
        return Verdict(False, FailureMode.UNFINISHED)
    tray = np.asarray(trace.object_pos["tray"], dtype=float)
    quat = np.asarray(trace.object_quat["tray"], dtype=float)
    areas = {p: np.asarray(trace.pad_area[p], dtype=float) for p in right_pads + left_pads}
    held = {
        "right": np.all(np.stack([areas[p] > CONTACT_AREA_MM2 for p in right_pads]), axis=0),
        "left": np.all(np.stack([areas[p] > CONTACT_AREA_MM2 for p in left_pads]), axis=0),
    }
    grasp_s = {side: float(np.sum(np.diff(t, prepend=t[0]) * mask)) for side, mask in held.items()}
    xy = np.linalg.norm(tray[:, :2] - np.asarray(goal_xy), axis=1)
    dz = np.abs(tray[:, 2] - float(goal_z))
    tilt = np.array([tilt_deg(q) for q in quat])
    corners = np.array(
        [[x, y, 0.0] for x in (-TRAY_LENGTH / 2, TRAY_LENGTH / 2) for y in (-TRAY_WIDTH / 2, TRAY_WIDTH / 2)]
    )
    corner_offsets = 2.0 * np.cross(quat[:, None, 1:], corners)
    world_corners = (
        tray[:, None, :]
        + corners
        + quat[:, None, :1] * corner_offsets
        + np.cross(quat[:, None, 1:], corner_offsets)
    )
    margins = np.asarray(support_size) / 2 - np.abs(world_corners[:, :, :2] - np.asarray(goal_xy))
    footprint_inside = np.all(margins >= 0.0, axis=(1, 2))
    cups = {
        name: np.array(
            [
                cup_standing_on_tray(p, q, tray_p, tray_q)[0]
                for p, q, tray_p, tray_q in zip(
                    trace.object_pos[name], trace.object_quat[name], tray, quat, strict=True
                )
            ]
        )
        for name in CUP_NAMES
    }
    all_standing = np.logical_and.reduce(list(cups.values()))
    released = ~np.any(np.stack(list(areas.values())) > CONTACT_AREA_MM2, axis=0)
    positioned = (
        (xy <= goal_xy_tol)
        & (dz <= goal_z_tol)
        & (tilt <= max_tilt_deg)
        & footprint_inside
        & all_standing
        & released
    )
    run = longest = 0.0
    for i in range(1, len(t)):
        run = run + (t[i] - t[i - 1]) if positioned[i] and positioned[i - 1] else 0.0
        longest = max(longest, run)
    success = run >= hold_s
    metrics = {
        "tray_goal_xy_m": float(xy[-1]),
        "tray_goal_z_m": float(dz[-1]),
        "tray_tilt_deg": float(tilt[-1]),
        "tray_footprint_inside": float(footprint_inside[-1]),
        "tray_edge_margin_m": float(np.min(margins[-1])),
        "standing_cups": float(sum(cups[name][-1] for name in CUP_NAMES)),
        "released": float(released[-1]),
        "completion_hold_s": run,
        "longest_completion_hold_s": longest,
        "right_grasp_s": grasp_s["right"],
        "left_grasp_s": grasp_s["left"],
    }
    metrics.update(
        {f"{name}_upright_deg": float(tilt_deg(trace.object_quat[name][-1])) for name in CUP_NAMES}
    )
    return Verdict(success, FailureMode.NONE if success else FailureMode.UNFINISHED, metrics)
