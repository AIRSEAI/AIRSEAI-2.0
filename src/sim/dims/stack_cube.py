"""stack_cube dimensions: the cube.

The cubes are primitives, so there is nothing to regenerate.
"""

from __future__ import annotations

__all__ = [
    "CUBE_SIZE",
]

# Cube edge, metres. Keep it tall enough that a grasp at the gel centre's
# lowest reachable height (about 34 mm above the table, set by the finger
# housing) still lands on the cube's side.
CUBE_SIZE = 0.045
