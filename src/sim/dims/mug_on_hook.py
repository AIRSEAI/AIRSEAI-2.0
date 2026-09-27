"""mug_on_hook dimensions: the mug and its handle, the rail's peg, and the coaster.

After editing, regenerate the assets::

    python benchmark/run/make_mug_on_hook_assets.py --headless
"""

from __future__ import annotations

import math

__all__ = [
    "COASTER_COLOR",
    "COASTER_FLOOR",
    "COASTER_HEIGHT",
    "COASTER_INNER_R",
    "COASTER_OUTER_R",
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
    "PEG_LENGTH",
    "PEG_R",
    "STOP_BALL_R",
    "PEG_ROOT_Z",
    "PEG_FALL_DEG",
    "RAIL_CAP_H",
    "RAIL_CAP_OVERHANG",
    "RAIL_COLOR",
    "RAIL_HEIGHT",
    "RAIL_THICKNESS",
    "RAIL_WIDTH",
    "eye_top_z",
    "hang_offset_x",
    "hang_offset_z",
    "ball_window",
    "peg_axis_z",
    "peg_span",
    "stop_lift",
    "x_window",
]

# ----- the mug ---------------------------------------------------------------
# A straight-walled hollow mug, metres. The body is narrow enough for the fully
# open jaws (46 mm gel gap) to close on it with about 4 mm a side, and tall
# enough that the gels sit clear of the finger housing's 34 mm reach below them.
MUG_OUTER_R = 0.019
MUG_WALL = 0.003
MUG_INNER_R = MUG_OUTER_R - MUG_WALL
MUG_HEIGHT = 0.080
MUG_FLOOR = 0.009
MUG_COLOR = (0.88, 0.87, 0.83)         # glazed off-white

# ----- the handle ------------------------------------------------------------
# A round-section loop in the mug's xz plane, standing off the +x side, swept
# along a rounded path whose ends are buried in the body. Handle and body are
# two closed meshes of one rigid body: their union is not itself watertight,
# and the SDF cooker needs each mesh closed.
#
# The section is round, so both walls a mis-aimed peg can strike -- the body's
# outside and the handle's outer bar -- are curved, and the contact carries a
# sideways component. That is the eye's lead-in, and why the peg needs no
# taper.
HANDLE_SECTION = 0.006
HANDLE_CORNER_R = 0.008                # path corner radius at the two outer corners
HANDLE_ROOT_R = 0.017                  # where the path's ends sit, inside the wall

# The eye: the hole between the body's wall and the handle's inner face. EYE_X
# is its clear width, EYE_HEIGHT its clear height. Both the rod and the stop
# ball on its end have to thread it; `ball_window()` is the tighter of the two.
EYE_X = 0.015
EYE_HEIGHT = 0.026

# The eye's floor above the mug's base, metres; the rest of the handle follows
# from this, the eye's size and the section. As high as the rim allows: a hung
# mug turns about the peg until its mass is under the contact, and how far it
# turns is set by how far the eye's roof stands above that mass.
EYE_FLOOR_Z = 0.040


def eye_top_z() -> float:
    """Height of the eye's roof above the mug's base, metres.

    The surface a hung mug rests on the peg by, so it fixes how far below the
    peg the mug hangs.
    """
    return EYE_FLOOR_Z + EYE_HEIGHT


def hang_offset_z() -> float:
    """Drop from the peg's axis to the base of a mug hanging on it, metres.

    The eye's roof rests on the peg's top, so the axis sits inside the eye,
    one radius below that roof.
    """
    return PEG_R - eye_top_z()


def hang_offset_x(eye_x: float = EYE_X) -> float:
    """Offset from the peg's axis to the axis of a mug hanging on it, metres.

    Negative: the body hangs on the near side, the eye centred on the peg.
    """
    return -(MUG_OUTER_R + eye_x / 2.0)


def x_window(eye_x: float = EYE_X) -> float:
    """Half-width of the x band in which the peg clears the eye, metres."""
    return (eye_x - 2.0 * PEG_R) / 2.0


# ----- the wall rail ---------------------------------------------------------
# A backboard standing on the table with a capping strip, and one peg. Metres.
RAIL_WIDTH = 0.090                     # along x
RAIL_THICKNESS = 0.010                 # along y
RAIL_HEIGHT = 0.125
RAIL_CAP_H = 0.008                     # the strip along the board's top edge
RAIL_CAP_OVERHANG = 0.006              # how far it stands proud of the board's face
RAIL_COLOR = (0.34, 0.24, 0.16)        # stained wood

# The peg: a round rod of one diameter leaving the board's face, ending in a
# ball. The rod falls towards that free end, so a hung mug runs down it and is
# caught by the ball; the ball stands `stop_lift()` proud of the rod, and a mug
# arriving with enough speed rides over that step and off the end.
PEG_R = 0.0025
PEG_LENGTH = 0.070
STOP_BALL_R = 0.0040
# Fall towards the free end, degrees, and the friction of the rod it works
# against. Steep with the friction nearly cancelling it: the run down the rod
# then stays slow, while a mug dropped onto the rod turns a large part of its
# fall into motion along it. That difference is what the stop ball reads.
PEG_FALL_DEG = 20.0
PEG_ROOT_Z = 0.115                     # above the table, at the board's face
PEG_COLOR = (0.34, 0.36, 0.40)         # brushed steel


def peg_axis_z(along: float) -> float:
    """Height of the peg's axis above the table, `along` metres out from the board."""
    return PEG_ROOT_Z - along * math.tan(math.radians(PEG_FALL_DEG))


def peg_span() -> float:
    """How far the peg reaches out from the board in y, metres."""
    return PEG_LENGTH * math.cos(math.radians(PEG_FALL_DEG))


def stop_lift() -> float:
    """How far the stop ball stands proud of the rod, metres.

    A hung mug rests its eye's roof on the rod, so this is the height the
    handle has to climb to pass the ball, and the energy barrier that decides
    whether the mug is caught or carries on off the end.
    """
    return STOP_BALL_R - PEG_R


def ball_window(eye_x: float = EYE_X) -> float:
    """Half-width of the x band in which the stop ball clears the eye, metres.

    Tighter than `x_window()`, so this is what bounds the aim: the eye has to
    pass over the ball on its way onto the rod.
    """
    return (eye_x - 2.0 * STOP_BALL_R) / 2.0


# ----- the coaster -----------------------------------------------------------
# The shallow dish the mug starts in, metres. It keeps the mug upright and
# bounds where it can stand, so the jitter stays inside the dish.
COASTER_OUTER_R = 0.032
COASTER_INNER_R = 0.026
COASTER_HEIGHT = 0.006
COASTER_FLOOR = 0.003
COASTER_COLOR = (0.24, 0.19, 0.14)     # cork
