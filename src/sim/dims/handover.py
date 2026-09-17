"""handover dimensions: the spring-loaded split sleeve and the two cups.

After editing, regenerate the assets::

    python benchmark/run/make_handover_assets.py --headless
"""

from __future__ import annotations

import math

__all__ = [
    "CUP_CHAMFER",
    "CUP_COLOR",
    "CUP_FLOOR",
    "CUP_HEIGHT",
    "CUP_OUTER_R",
    "DEFAULT_TARGET_BORE",
    "GAP_CLEARANCE",
    "GAP_MIN",
    "KEY_CHAMFER",
    "KEY_HALF_LENGTH",
    "KEY_HEIGHT",
    "KEY_OVERLAP",
    "KEY_THICKNESS",
    "MIN_SEAT",
    "REST_GAP",
    "SHELL_MESH_NAMES",
    "SLEEVE_COLOR",
    "SLEEVE_INNER_R",
    "SLEEVE_LENGTH",
    "SLEEVE_OUTER_R",
    "SLEEVE_WALL",
    "SOURCE_BORE_R",
    "SPAWN_GAP",
    "TARGET_BORE_SWEEP",
    "gap_window",
    "target_mesh_name",
]

# The sleeve, metres: two rigid half-shells cut on the plane through its
# axis, joined by a spring that pushes them apart. It stands upright in the
# cups. Narrow enough that, opened into the target's window, it still clears
# the fully open jaws by a few millimetres a side, so a hand letting go can
# slide off it. Long enough for a grasp above the source cup's rim and for two
# grasps one above the other at the exchange.
SLEEVE_OUTER_R = 0.018
SLEEVE_WALL = 0.003
SLEEVE_INNER_R = SLEEVE_OUTER_R - SLEEVE_WALL
SLEEVE_LENGTH = 0.080
SLEEVE_COLOR = (0.78, 0.80, 0.82)

SHELL_MESH_NAMES = ("sleeve_half_a", "sleeve_half_b")

# Shell separation along the closing axis, metres. The sleeve stands in the
# source cup at SPAWN_GAP, its spring pressing both shells on the bore; free,
# the spring opens it to REST_GAP; the joint's limit stops it at GAP_MIN
# before the shells meet.
SPAWN_GAP = 0.002
REST_GAP = 0.010
GAP_MIN = 0.0005

# Clearance a fit needs, metres.
GAP_CLEARANCE = 0.0005

# The cups, metres: a hollow cylinder with a floor, open at the top, the rim
# chamfered. The target also has a key standing on its floor.
# Each is as shallow as holding its sleeve allows: the giver's grasp must
# sit above the source rim and still reach once the sleeve is drawn clear,
# and the receiver's grasp, below the giver's on the same sleeve, must sit
# above the target rim.
CUP_OUTER_R = 0.032
CUP_HEIGHT = {"source": 0.025, "target": 0.020}
CUP_FLOOR = 0.005
CUP_CHAMFER = 0.002

# Source bore radius: a hair over the sleeve at its spawn gap, so the spring
# holds the shells against the bore and friction resists a straight pull.
SOURCE_BORE_R = (2 * SLEEVE_OUTER_R + SPAWN_GAP + 0.0001) / 2

# Target bore diameters, the task's knob; one mesh each. The wider the bore,
# the wider the window of held gaps that seats.
TARGET_BORE_SWEEP = (0.041, 0.042, 0.044, 0.048)
DEFAULT_TARGET_BORE = 0.042

# The target's key, metres: a flat bar standing on the floor along the
# sleeve's split axis. The two rigid shells open the bore only along that
# axis, so the key's length sets the narrowest gap that seats: each end
# reaches KEY_OVERLAP past the closed sleeve's inner wall.
KEY_THICKNESS = 0.004
KEY_OVERLAP = 0.001
KEY_HALF_LENGTH = math.sqrt(SLEEVE_INNER_R ** 2 - (KEY_THICKNESS / 2) ** 2) + KEY_OVERLAP
KEY_HEIGHT = 0.010                   # above the floor's top
KEY_CHAMFER = 0.002                  # 45-degree lead-in at the top of each end

# One colour per cup, so the two roles are told apart at a glance.
CUP_COLOR = {"source": (0.85, 0.55, 0.15), "target": (0.15, 0.55, 0.60)}

# Success bar: the sleeve's lower rim this far below the target's rim, metres.
MIN_SEAT = 0.010


def gap_window(bore_d: float) -> tuple[float, float]:
    """(narrowest, widest) held gap that seats in a target of bore `bore_d`.

    Split across a diameter, the sleeve's width along the split axis grows by
    the gap and its width across it does not. The widest gap is where the
    outer width along the split meets the bore; the narrowest is where the
    inner arcs, at the key's half thickness off the axis, clear its ends.
    """
    half_chord = math.sqrt(SLEEVE_INNER_R ** 2 - (KEY_THICKNESS / 2) ** 2)
    return (2 * (KEY_HALF_LENGTH - half_chord) + GAP_CLEARANCE,
            bore_d - GAP_CLEARANCE - 2 * SLEEVE_OUTER_R)


def target_mesh_name(bore_d: float) -> str:
    """File stem of a target cup mesh, e.g. `cup_target_b42.0mm`."""
    return f"cup_target_b{bore_d * 1e3:.1f}mm"
