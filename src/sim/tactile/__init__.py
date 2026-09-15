"""Tactile support code on top of TacSL, without modifying it.

Holds the geometry the envs, experts and criteria need around TacSL's tactile
RGB and per-taxel force field:

    quantities  contact area and quaternion helpers (pure geometry)
    sdf         the SDF collider TacSL requires on any object it must sense
    pad         measured GelSight R1.5 gel geometry and taxel pitch
"""

from .pad import PAD, PadGeometry, taxel_pitch
from .quantities import contact_area, quat_apply, quat_apply_inverse
from .sdf import apply_ccd, apply_sdf_collision, apply_triangle_mesh_collision

__all__ = [
    "PAD",
    "PadGeometry",
    "apply_ccd",
    "apply_sdf_collision",
    "apply_triangle_mesh_collision",
    "contact_area",
    "quat_apply",
    "quat_apply_inverse",
    "taxel_pitch",
]
