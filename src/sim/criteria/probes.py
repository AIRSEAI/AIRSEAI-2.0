"""Pose geometry shared by every task, computed the same way on both sides.

The expert runner and the policy evaluator both derive the `EpisodeTrace`
inputs of a task's criterion from object positions and orientations, so those
derivations live here and have one implementation. Plain numpy in metres,
degrees and radians; no torch, no scene handles.

The gate a single task puts on `seat_depth` lives in `sim.criteria.<task>`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["axis_from_quat", "seat_depth", "tilt_deg", "tilt_from_horizontal_deg",
           "tip_position"]


def axis_from_quat(quat, local=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotate a body-local axis into the world by a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=float).reshape(4))
    v = np.asarray(local, dtype=float)
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def tilt_deg(quat) -> float:
    """Angle between the body's local +z and world +z, in degrees.

    For a held peg or sleeve this is the binding indicator: it separates an
    object still upright in the grasp from one that has rotated out of it.
    """
    up = axis_from_quat(quat)
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def tilt_from_horizontal_deg(quat, local=(0.0, 1.0, 0.0)) -> float:
    """Angle a body-local axis makes with the horizontal plane, in degrees.

    The "is it level" measure for a bar carried horizontally, where `tilt_deg`
    would already read 90. Zero means level; 90 means hanging. Unsigned, so
    the threshold is symmetric in which end dips.
    """
    a = axis_from_quat(quat, local)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return 0.0
    return float(np.degrees(np.arcsin(np.clip(abs(a[2]) / n, 0.0, 1.0))))


def tip_position(pos, quat, length: float) -> np.ndarray:
    """World position of a cylinder's lower tip, honouring its orientation.

    Not `pos_z - length/2`, which holds only while the object stands upright.
    """
    pos = np.asarray(pos, dtype=float).reshape(3)
    return pos + axis_from_quat(quat, (0.0, 0.0, -length / 2.0))


def seat_depth(pos, quat, *, length: float, target_xy, target_z: float,
               xy_tol: float, max_tilt_deg: float) -> float:
    """How far a cylinder's lower tip is below `target_z`, zero unless it is
    over the target and near upright.

    Three gates, all required: the tip below the reference height, its centre
    within `xy_tol` of the target's axis in xy, and the body within
    `max_tilt_deg` of upright. An object dropped on the table beside the
    target satisfies only the first.

    The gate reads the tip, not the body's middle: an object seated and
    leaning a few degrees against a bore moves its middle further than its
    tip.
    """
    tip = tip_position(pos, quat, length)
    if tilt_deg(quat) > max_tilt_deg:
        return 0.0
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    if float(np.linalg.norm(tip[:2] - target_xy)) > xy_tol:
        return 0.0
    return max(0.0, target_z - float(tip[2]))
