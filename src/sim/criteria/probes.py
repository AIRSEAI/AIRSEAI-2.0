"""Task quantities that both the expert and the policy rollout have to compute.

These derive the `EpisodeTrace` inputs of `sim.criteria.tasks` from object
positions and orientations, so the expert and the policy rollout compute them
identically. Plain numpy in metres, degrees and radians; no torch or scene
handles.
"""

from __future__ import annotations

import numpy as np

__all__ = ["axis_from_quat", "insertion_depth", "seated_in_cup", "seated_on_shaft",
           "tilt_deg", "tilt_from_horizontal_deg", "tip_position"]


def axis_from_quat(quat, local=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotate a body-local axis into the world by a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=float).reshape(4))
    v = np.asarray(local, dtype=float)
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def tilt_deg(quat) -> float:
    """Angle between the body's local +z and world +z, in degrees.

    For a held peg or plug this is the binding indicator: it separates an
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


def insertion_depth(pos, quat, *, length: float, hole_xy, mouth_z: float,
                    hole_radius: float, xy_margin: float = 0.002,
                    max_tilt_deg: float = 20.0) -> float:
    """How far the tip is *into the hole*, zero unless it actually is.

    Gated on three things: the tip must be below the mouth, within the hole's
    radius of its centre in xy, and the object must be near upright. An object
    dropped on the table beside the hole satisfies only the first.
    """
    tip = tip_position(pos, quat, length)
    if tilt_deg(quat) > max_tilt_deg:
        return 0.0
    hole_xy = np.asarray(hole_xy, dtype=float).reshape(2)
    if float(np.linalg.norm(tip[:2] - hole_xy)) > hole_radius + xy_margin:
        return 0.0
    return max(0.0, mouth_z - float(tip[2]))


def seated_in_cup(pos, quat, *, length: float, cup_xy, rim_z: float, xy_tol: float,
                  max_tilt_deg: float = 10.0) -> float:
    """How far a sleeve's lower rim is below a cup's rim, zero unless the
    sleeve stands centred in the cup.

    Gated like `insertion_depth`: the lower rim must be below the cup's rim,
    its centre within `xy_tol` of the cup's centre, and the sleeve near
    upright. The gate reads the lower rim, not the sleeve's middle: a seated
    sleeve leaning a few degrees against the bore moves its middle further
    than its rim. A sleeve resting on something in the cup off its centre, or
    lying across the cup, reads zero.
    """
    rim = tip_position(pos, quat, length)
    if tilt_deg(quat) > max_tilt_deg:
        return 0.0
    cup_xy = np.asarray(cup_xy, dtype=float).reshape(2)
    if float(np.linalg.norm(rim[:2] - cup_xy)) > xy_tol:
        return 0.0
    return max(0.0, rim_z - float(rim[2]))


def seated_on_shaft(pos, quat, *, height: float, shaft_xy, shaft_top_z: float,
                    bore_r: float, max_tilt_deg: float = 20.0) -> float:
    """How far a sleeve's lower rim is below a shaft's top, zero unless it is on the shaft.

    The reverse of `insertion_depth`: the hole travels onto the peg. The rim
    must be below the shaft top, its centre within the bore radius of the
    shaft axis, and the sleeve near upright.
    """
    rim = tip_position(pos, quat, height)
    if tilt_deg(quat) > max_tilt_deg:
        return 0.0
    shaft_xy = np.asarray(shaft_xy, dtype=float).reshape(2)
    if float(np.linalg.norm(rim[:2] - shaft_xy)) > bore_r:
        return 0.0
    return max(0.0, shaft_top_z - float(rim[2]))
