"""mug_on_hook criterion: the mug ends hanging on the rail's peg.

Reads::

    object_pos["mug"], object_quat["mug"], extra["grasp_frame_pos"]
"""

from __future__ import annotations

import numpy as np

from sim.criteria.detectors import detect_grasp_failure, detect_slip
from sim.criteria.probes import axis_from_quat
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict
from sim.dims.mug_on_hook import EYE_FLOOR_Z, EYE_HEIGHT, EYE_X, MUG_OUTER_R

__all__ = ["eye_margin", "verdict"]

# How far the mug may travel within the grasp before it counts as slipped, m.
# Above the seating descent: the hand drives down against a peg that will not
# move, so a mug that seats correctly rides that far up in the gel by design.
MAX_SLIP = 0.030


def eye_margin(
    mug_pos,
    mug_quat,
    peg_root,
    peg_tip,
    *,
    eye_x: float = EYE_X,
    eye_floor: float = EYE_FLOOR_Z,
    eye_height: float = EYE_HEIGHT,
    body_r: float = MUG_OUTER_R,
) -> float:
    """Return the peg crossing clearance inside the handle eye, in metres.

    Transform the peg segment into the mug frame and intersect its y=0 plane.
    Clearance is positive inside the eye and negative outside it.
    """
    pos = np.asarray(mug_pos, dtype=float).reshape(3)
    w, x, y, z = (float(v) for v in np.asarray(mug_quat, dtype=float).reshape(4))
    conj = (w, -x, -y, -z)  # world -> body is the inverse turn
    a = axis_from_quat(conj, np.asarray(peg_root, dtype=float).reshape(3) - pos)
    b = axis_from_quat(conj, np.asarray(peg_tip, dtype=float).reshape(3) - pos)

    dy = b[1] - a[1]
    if abs(dy) < 1e-9:
        return -float(eye_height)
    s = -a[1] / dy
    if not 0.0 <= s <= 1.0:
        return -float(eye_height)
    p = a + s * (b - a)
    return float(min(p[0] - body_r, body_r + eye_x - p[0], p[2] - eye_floor, eye_floor + eye_height - p[2]))


def verdict(
    trace: EpisodeTrace,
    *,
    peg_root,
    peg_tip,
    pads: tuple[str, ...],
    eye_x: float = EYE_X,
    max_slip_m: float = MAX_SLIP,
    area_mm2: float = 5.0,
) -> Verdict:
    """Require a valid grasp and a final pose with the peg inside the handle.

    Outside the handle, contact and slip determine the failure category.
    """
    pos = np.asarray(trace.object_pos["mug"], dtype=float)
    quat = np.asarray(trace.object_quat["mug"], dtype=float)
    margin = eye_margin(pos[-1], quat[-1], peg_root, peg_tip, eye_x=eye_x)

    never_held, m_grasp = detect_grasp_failure(trace, pads, area_mm2=area_mm2)
    slipped, m_slip = detect_slip(trace, "mug", pads, max_slip_m=max_slip_m, area_mm2=area_mm2)
    still_held = any(bool(trace.in_contact(p, area_mm2)[-1]) for p in pads)
    metrics = {
        "eye_margin_m": margin,
        "eye_x_m": float(eye_x),
        "final_mug_z_m": float(pos[-1][2]),
        "still_held": float(still_held),
        **m_grasp,
        **m_slip,
    }

    if never_held:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if margin > 0.0:
        return Verdict(True, metrics=metrics)
    if still_held:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    if slipped:
        return Verdict(False, FailureMode.SLIP, metrics)
    return Verdict(False, FailureMode.MISTIMING, metrics)
