# SPDX-License-Identifier: Apache-2.0
"""The tacff image: the native force-field channels packed as one RGB frame.

Isaac-free, so dataset conversion and evaluation use the same scales.

    R   = clip(depth_mm / DEPTH_MAX_MM, 0, 1)                   depth >= 0
    G,B = clip(ff_shear / (2 * SHEAR_MAX) + 0.5, 0, 1)      (x, y), signed

Shear is signed, so its [-SHEAR_MAX, +SHEAR_MAX] range (width 2*SHEAR_MAX)
is centred on 0.5; no contact therefore renders as (0, 0.5, 0.5), not black.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DEPTH_MAX_MM", "SHEAR_MAX", "tacff_image"]

DEPTH_MAX_MM = 3.0     # gel thickness bound; depth saturates physically here
SHEAR_MAX = 6.0        # |ff_shear| bound: mu * F_n <= 2 * depth -> 6 in x1e3 units


def tacff_image(depth_mm: np.ndarray, shear: np.ndarray) -> np.ndarray:
    """(H,W) depth in mm + (H,W,2) ff_shear (x1e3) -> (H,W,3) uint8."""
    r = np.clip(depth_mm / DEPTH_MAX_MM, 0.0, 1.0) * 255.0
    gb = np.clip(shear / (2.0 * SHEAR_MAX) + 0.5, 0.0, 1.0) * 255.0
    return np.concatenate([r[..., None], gb], axis=-1).round().astype(np.uint8)
