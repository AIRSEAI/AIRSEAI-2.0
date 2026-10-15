# SPDX-License-Identifier: Apache-2.0
"""loose_parts_box_transfer criterion: final shelf height without spilled parts.

Reads::

    object_pos["parts_box"], extra["escaped"]
"""

from __future__ import annotations

import numpy as np

from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.loose_parts_box_transfer import BOX_LENGTH, BOX_WIDTH

GOAL_TOL_Z = 0.006
HOLD_S = 1.0


def part_escaped(box_pos, box_quat, part_pos, radius: float, *, box_length: float = BOX_LENGTH) -> bool:
    """Whether an item centre has cleared a wall or fallen below the box."""
    q = np.asarray(box_quat, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    local = rotation.T @ (np.asarray(part_pos) - np.asarray(box_pos))
    return bool(
        abs(local[0]) > box_length / 2 + radius / 2
        or abs(local[1]) > BOX_WIDTH / 2 + radius / 2
        or local[2] < -radius / 2
    )


def verdict(
    trace: EpisodeTrace,
    *,
    goal_xy,
    goal_surface_z: float,
    hold_s: float = HOLD_S,
    tol_z: float = GOAL_TOL_Z,
) -> Verdict:
    """Require final shelf height for one second and no escaped contents."""
    t = np.asarray(trace.t, dtype=float)
    if len(t) < 2:
        return Verdict(False, FailureMode.UNFINISHED)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Trace timestamps must increase")
    pos = np.asarray(trace.object_pos["parts_box"], dtype=float)
    escaped = np.asarray(trace.extra["escaped"], dtype=bool)
    if pos.shape != (len(t), 3) or escaped.shape != t.shape:
        raise ValueError("Trace signals must share the same clock")
    goal = np.asarray(goal_xy, dtype=float)
    placed = np.abs(pos[:, 2] - goal_surface_z) <= tol_z
    run = longest = 0.0
    success_time = None
    for i in range(1, len(t)):
        run = run + (t[i] - t[i - 1]) if placed[i] and placed[i - 1] else 0.0
        longest = max(longest, run)
        if success_time is None and run + 1e-9 >= hold_s:
            success_time = float(t[i])
    metrics = {
        "goal_error_x_m": float(abs(pos[-1, 0] - goal[0])),
        "goal_error_y_m": float(abs(pos[-1, 1] - goal[1])),
        "goal_error_z_m": float(abs(pos[-1, 2] - goal_surface_z)),
        "goal_hold_s": run,
        "longest_goal_hold_s": longest,
        "required_hold_s": hold_s,
        "escaped": float(escaped.any()),
    }
    if escaped.any():
        metrics["first_escape_s"] = float(t[np.argmax(escaped)])
        return Verdict(False, FailureMode.SLIP, metrics)
    if run + 1e-9 >= hold_s:
        metrics["success_time_s"] = success_time
        return Verdict(True, metrics=metrics)
    return Verdict(False, FailureMode.UNFINISHED, metrics)
