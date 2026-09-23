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

# The bottle, metres: a body and a narrower neck, revolved about z. The body
# fits the fully open jaws; the neck is long enough to keep fingers on the
# cap clear of the body.
BOTTLE_R = 0.019
BOTTLE_BODY_H = 0.085
NECK_R = 0.013
NECK_H = 0.028
BOTTLE_H = BOTTLE_BODY_H + NECK_H
BOTTLE_COLOR = (0.30, 0.52, 0.72)

# The cap, metres: an inverted cup whose roof rests on the neck's top. The
# bore clears the neck by more than one SDF voxel of the bottle, below which
# the sampled neck reads as touching it.
CAP_R = 0.017
CAP_H = 0.020
CAP_BORE_R = NECK_R + 0.0015
CAP_BORE_DEPTH = 0.014
CAP_COLOR = (0.82, 0.28, 0.20)

# The tray the cap goes into, metres: a round dish, wide enough for a cap let
# go a little off centre and shallow enough to keep the fingers above its rim.
TRAY_OUTER_R = 0.055
TRAY_INNER_R = 0.050
TRAY_HEIGHT = 0.008
TRAY_FLOOR = 0.003
TRAY_COLOR = (0.35, 0.55, 0.40)
