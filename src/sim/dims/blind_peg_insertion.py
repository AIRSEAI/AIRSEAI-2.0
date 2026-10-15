# SPDX-License-Identifier: Apache-2.0
"""blind_peg_insertion dimensions: the peg, the board, and the clearance sweep.

After editing, regenerate the assets::

    python benchmark/run/make_blind_peg_insertion_assets.py --headless
"""

from __future__ import annotations

__all__ = [
    "BOARD_THICKNESS",
    "BOARD_X",
    "BOARD_Y",
    "CLEARANCE_SWEEP",
    "DEFAULT_CLEARANCE",
    "HOLE_DEPTH",
    "MIN_DEPTH",
    "PEG_DIAMETER",
    "PEG_LENGTH",
    "hole_diameter",
]

# The peg: a plain cylinder, metres.
PEG_DIAMETER = 0.010
PEG_LENGTH = 0.080

# Board thickness, metres. The table closes the hole's bottom, so this is also
# the hole depth; keep it above MIN_DEPTH.
BOARD_THICKNESS = 0.040
HOLE_DEPTH = BOARD_THICKNESS

# How far into the hole the peg must end for the insertion to count, metres.
MIN_DEPTH = 0.025

# Plate footprint, metres, hole at the centre. Large enough to cover the
# expert's search lattice plus the hole jitter.
BOARD_X, BOARD_Y = 0.100, 0.075

# Radial clearance between peg and hole, metres. One authored board per value.
DEFAULT_CLEARANCE = 0.0010
CLEARANCE_SWEEP = (0.0005, 0.0010, 0.0015, 0.0020, 0.0030)


def hole_diameter(clearance: float = DEFAULT_CLEARANCE) -> float:
    """Hole diameter for a radial clearance, metres: the clearance is per side."""
    return PEG_DIAMETER + 2.0 * clearance
