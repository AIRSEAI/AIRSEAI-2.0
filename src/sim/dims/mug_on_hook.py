"""mug_on_hook dimensions: the mug and its handle, and the rail's peg.

After editing, regenerate the assets::

    python benchmark/run/make_mug_on_hook_assets.py --headless
"""

from __future__ import annotations

import math

__all__ = [
    "EYE_FLOOR_Z",
    "EYE_HEIGHT",
    "EYE_X",
    "HANDLE_CORNER_R",
    "HANDLE_ROOT_R",
    "HANDLE_SECTION",
    "MUG_COLOR",
    "MUG_FLOOR",
    "MUG_HEIGHT",
    "MUG_INNER_R",
    "MUG_OUTER_R",
    "MUG_WALL",
    "PEG_COLOR",
    "PEG_FALL_DEG",
    "PEG_LENGTH",
    "PEG_R",
    "PEG_ROOT_Z",
    "RAIL_CAP_H",
    "RAIL_CAP_OVERHANG",
    "RAIL_COLOR",
    "RAIL_HEIGHT",
    "RAIL_THICKNESS",
    "RAIL_WIDTH",
    "STOP_BALL_R",
    "ball_window",
    "eye_top_z",
    "hang_offset_x",
    "hang_offset_z",
    "peg_axis_z",
    "peg_span",
    "stop_lift",
    "x_window",
]

# ----- the mug ---------------------------------------------------------------
# Lengths are in metres.
MUG_OUTER_R = 0.019
MUG_WALL = 0.003
MUG_INNER_R = MUG_OUTER_R - MUG_WALL
MUG_HEIGHT = 0.080
MUG_FLOOR = 0.009
MUG_COLOR = (0.88, 0.87, 0.83)  # glazed off-white

# ----- the handle ------------------------------------------------------------
HANDLE_SECTION = 0.006
HANDLE_CORNER_R = 0.008  # path corner radius at the two outer corners
HANDLE_ROOT_R = 0.017  # where the path's ends sit, inside the wall

EYE_X = 0.019
EYE_HEIGHT = 0.026

EYE_FLOOR_Z = 0.040


def eye_top_z() -> float:
    """Height of the eye's roof above the mug base, in metres."""
    return EYE_FLOOR_Z + EYE_HEIGHT


def hang_offset_z() -> float:
    """Vertical offset from the peg axis to the hanging mug base, in metres."""
    return PEG_R - eye_top_z()


def hang_offset_x(eye_x: float = EYE_X) -> float:
    """Horizontal offset from the peg axis to the hanging mug axis, in metres."""
    return -(MUG_OUTER_R + eye_x / 2.0)


def x_window(eye_x: float = EYE_X) -> float:
    """Half-width of the x band in which the peg clears the eye, metres."""
    return (eye_x - 2.0 * PEG_R) / 2.0


# ----- the wall rail ---------------------------------------------------------
RAIL_WIDTH = 0.090  # along x
RAIL_THICKNESS = 0.010  # along y
RAIL_HEIGHT = 0.125
RAIL_CAP_H = 0.008  # the strip along the board's top edge
RAIL_CAP_OVERHANG = 0.006  # how far it stands proud of the board's face
RAIL_COLOR = (0.34, 0.24, 0.16)  # stained wood

PEG_R = 0.0025
PEG_LENGTH = 0.070
STOP_BALL_R = 0.0070
PEG_FALL_DEG = 20.0  # downward slope toward the free end, degrees
PEG_ROOT_Z = 0.115  # above the table, at the board's face
PEG_COLOR = (0.34, 0.36, 0.40)  # brushed steel


def peg_axis_z(along: float) -> float:
    """Height of the peg's axis above the table, `along` metres out from the board."""
    return PEG_ROOT_Z - along * math.tan(math.radians(PEG_FALL_DEG))


def peg_span() -> float:
    """How far the peg reaches out from the board in y, metres."""
    return PEG_LENGTH * math.cos(math.radians(PEG_FALL_DEG))


def stop_lift() -> float:
    """Height of the stop ball above the rod, in metres."""
    return STOP_BALL_R - PEG_R


def ball_window(eye_x: float = EYE_X) -> float:
    """Half-width of the sideways clearance between the eye and stop ball, in metres."""
    return (eye_x - 2.0 * STOP_BALL_R) / 2.0
