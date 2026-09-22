"""sleeve_on_shaft dimensions: the split sleeve, the shaft it slides onto, and the seat bar.

After editing, regenerate the assets::

    python benchmark/run/make_sleeve_on_shaft_assets.py --headless
"""

from __future__ import annotations

__all__ = [
    "BASE_HEIGHT",
    "BASE_RADIUS",
    "CLEARANCE_SWEEP",
    "DEFAULT_CLEARANCE",
    "JOINT_TRAVEL",
    "MIN_SEAT",
    "SHAFT_CHAMFER",
    "SHAFT_COLOR",
    "SHAFT_HEIGHT",
    "SHELL_MESH_NAMES",
    "SLEEVE_COLOR",
    "SLEEVE_HEIGHT",
    "SLEEVE_INNER_R",
    "SLEEVE_OUTER_R",
    "SLEEVE_WALL",
    "SPLIT_GAP",
    "shaft_mesh_name",
    "shaft_radius",
]

# The sleeve, metres. Outer diameter plus SPLIT_GAP must fit between the open
# gel faces; the height keeps the finger housing clear of the shaft base once
# the sleeve is seated.
SLEEVE_OUTER_R = 0.020
SLEEVE_WALL = 0.002
SLEEVE_INNER_R = SLEEVE_OUTER_R - SLEEVE_WALL
SLEEVE_HEIGHT = 0.070
SLEEVE_COLOR = (0.30, 0.62, 0.78)      # one appearance for every material

# The two half-shells: authored with their flat face on the xz plane and
# spawned SPLIT_GAP apart along y, the gripper's closing axis, joined by a
# spring that may close JOINT_TRAVEL before its limit. JOINT_TRAVEL must
# exceed SPLIT_GAP + 2 * clearance, so a hard squeeze can shut the bore
# below the shaft before the limit stops it.
SHELL_MESH_NAMES = ("sleeve_half_a", "sleeve_half_b")
SPLIT_GAP = 0.0005
JOINT_TRAVEL = 0.0045

# The shaft stands on a round base; radial clearance to the sleeve bore is the
# task's geometric knob. One shaft mesh is generated per clearance.
BASE_RADIUS = 0.030
BASE_HEIGHT = 0.006
SHAFT_HEIGHT = 0.022                   # above the base top
SHAFT_CHAMFER = 0.0008
SHAFT_COLOR = (0.55, 0.55, 0.58)
DEFAULT_CLEARANCE = 0.001
CLEARANCE_SWEEP = (0.0005, 0.001, 0.0015)

# Success bar: the sleeve's lower rim this far below the shaft top, metres.
MIN_SEAT = 0.015


def shaft_radius(clearance: float) -> float:
    """Shaft radius for a radial clearance to the sleeve bore, metres."""
    return SLEEVE_INNER_R - clearance


def shaft_mesh_name(clearance: float) -> str:
    """File stem of the shaft mesh for a clearance, e.g. `shaft_c1.00mm`."""
    return f"shaft_c{clearance * 1e3:.2f}mm"
