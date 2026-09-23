"""cap_unscrew dimensions: the bottle, its screw cap, and the tray.

After editing, regenerate the assets::

    python benchmark/run/make_cap_unscrew_assets.py --headless
"""

from __future__ import annotations

__all__ = [
    "BOTTLE_BODY_H",
    "BOTTLE_COLOR",
    "BOTTLE_H",
    "BOTTLE_R",
    "CAP_BORE_DEPTH",
    "CAP_BORE_R",
    "CAP_COLOR",
    "CAP_H",
    "CAP_R",
    "NECK_H",
    "NECK_R",
    "TRAY_COLOR",
    "TRAY_FLOOR",
    "TRAY_HEIGHT",
    "TRAY_INNER_R",
    "TRAY_OUTER_R",
]

# The bottle, metres: a solid of revolution, body then neck, standing on its
# base. The body is narrow enough for the fully open jaws (46 mm gel gap) to
# close on it with room to spare; the neck is what the cap screws onto. The
# neck is long enough that fingers pitched down onto the cap clear the body,
# which is wider than the cap.
BOTTLE_R = 0.019
BOTTLE_BODY_H = 0.085
NECK_R = 0.013
NECK_H = 0.028
BOTTLE_H = BOTTLE_BODY_H + NECK_H
BOTTLE_COLOR = (0.30, 0.52, 0.72)

# The cap, metres: an inverted cup over the neck. Its bore clears the neck by
# more than one SDF voxel of the bottle (about 0.8 mm at resolution 128), so
# the sampled neck does not read as touching the bore; the thread joint keeps
# the cap on its axis. The roof rests on the neck's top, which leaves the
# skirt well clear of the body's shoulder. Tall enough for a full pad contact.
CAP_R = 0.017
CAP_H = 0.020
CAP_BORE_R = NECK_R + 0.0015
CAP_BORE_DEPTH = 0.014
CAP_COLOR = (0.82, 0.28, 0.20)

# The tray the cap goes into, metres: a shallow round dish, floor and rim.
# Wide enough that a cap set down by a hand a little off centre lands
# inside, and shallow enough that the fingers holding the cap by its middle
# stay clear of the rim on the way out.
TRAY_OUTER_R = 0.055
TRAY_INNER_R = 0.050
TRAY_HEIGHT = 0.008
TRAY_FLOOR = 0.003
TRAY_COLOR = (0.35, 0.55, 0.40)
