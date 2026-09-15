"""Geometry of the GelSight R1.5 gel pad in the robot asset.

A pad position that misses the gel produces an identically-zero force field
with no error message, so these numbers come from the asset itself.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PAD", "PadGeometry", "taxel_pitch"]


@dataclass(frozen=True)
class PadGeometry:
    """Where the gel is, in world coordinates, with the finger at z=0.5.

    The pad is a 24 x 32 x 3 mm slab whose sensing normal is +z. `centre_y` is
    the midpoint of the y span, not a normal offset.
    """

    centre_x: float = 0.0
    centre_y: float = 0.06776
    surface_z: float = 0.50225
    span_x: float = 0.02398
    span_y: float = 0.03200
    thickness: float = 0.00306
    # Past this the sphere centre crosses the taxel plane and the force field
    # turns over: it is no longer indentation, it is the sphere swallowing the
    # sensing points.
    max_usable_indent_mm: float = 2.5


PAD = PadGeometry()


def taxel_pitch(grid: tuple[int, int],
                pad: PadGeometry = PAD,
                margin: float = 0.003) -> tuple[float, float]:
    """Taxel spacing in metres for a (rows, cols) grid, as (row_pitch, col_pitch).

    Rows run along **x** and cols along **y**. TacSL does not simply inset the
    bbox by `margin`: it insets asymmetrically so that the sensed region takes
    the aspect ratio of the requested grid, which makes `margin` a lower bound
    rather than the actual edge distance. The sensed fraction of the bbox is
    therefore fixed.
    """
    rows, cols = grid
    # Fraction of the pad bbox that TacSL populates. Fixed across grids because
    # the inset follows the grid's aspect ratio.
    frac_x, frac_y = 0.7445, 0.7445
    del margin  # kept in the signature for callers; see the docstring
    return (pad.span_x * frac_x / max(rows - 1, 1),
            pad.span_y * frac_y / max(cols - 1, 1))
