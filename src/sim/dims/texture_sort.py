"""texture_sort dimensions: the two puck classes, their surface relief, and the cup.

After editing, regenerate the assets::

    python benchmark/run/make_texture_sort_assets.py --headless
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CUP_COLOR",
    "CUP_FLOOR",
    "CUP_HEIGHT",
    "CUP_INNER_R",
    "CUP_WALL",
    "OBJECT_HEIGHT",
    "OBJECT_RADIUS",
    "OBJECT_SPECS",
    "ObjectSpec",
]

# The puck: a standing cylinder, metres. Both classes share these; only the
# surface differs.
OBJECT_RADIUS = 0.019
OBJECT_HEIGHT = 0.060

# The sorting cup: a hollow cylinder, metres, open at the top.
CUP_INNER_R = 0.040
CUP_WALL = 0.004
CUP_HEIGHT = 0.040
CUP_FLOOR = 0.004
CUP_COLOR = (0.38, 0.38, 0.40)      # authored into the USD; the scene recolours each bin


@dataclass(frozen=True)
class ObjectSpec:
    """One puck class: what the mesh generator and the criterion both need."""

    name: str
    bin_index: int          # criterion's `correct_bin`: 0 smooth, 1 textured
    radius: float
    height: float
    dimple_depth: float     # radial modulation amplitude, metres; 0 = smooth
    k_theta: int            # dimples around the circumference
    k_z: int                # dimples along the height
    colour: tuple[float, float, float]


# The two classes. `rough` carries shallow dimples the gel can resolve; the
# two colours differ by less than render noise.
OBJECT_SPECS = {
    "smooth": ObjectSpec("puck_smooth", 0, OBJECT_RADIUS, OBJECT_HEIGHT,
                         0.0, 0, 0, (0.90, 0.90, 0.88)),
    "rough": ObjectSpec("puck_rough", 1, OBJECT_RADIUS, OBJECT_HEIGHT,
                        0.0009, 24, 12, (0.88, 0.88, 0.90)),
}
