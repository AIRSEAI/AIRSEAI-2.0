"""handover_cup_transfer dimensions: the two cups and the seating bar.

After editing, regenerate the assets::

    python benchmark/run/make_handover_cup_transfer_assets.py --headless
"""

from __future__ import annotations

__all__ = [
    "CUP_BORE_R",
    "CUP_CHAMFER",
    "CUP_COLOR",
    "CUP_FLOOR",
    "CUP_HEIGHT",
    "CUP_OUTER_R",
    "MIN_SEAT",
]

# The cups, metres: one mesh per colour, differing only in bore.
CUP_OUTER_R = 0.016
CUP_HEIGHT = 0.040
CUP_FLOOR = 0.005
CUP_CHAMFER = 0.003          # 45-degree lead-in at the rim

# Bore radius per cup. The source cup's bore is tight, so the bar stands
# upright for the grasp; the destination's is wide, so the drop is forgiving.
CUP_BORE_R = {"amber": 0.004, "teal": 0.010}

# Display colour per cup; the colour is the only visible difference.
CUP_COLOR = {"amber": (0.85, 0.55, 0.15), "teal": (0.15, 0.55, 0.60)}

# Success bar: the bar's foot this far below the destination rim, metres.
MIN_SEAT = 0.010
