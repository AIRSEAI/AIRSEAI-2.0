"""Shared overview camera and collision settings for scene previews."""

from __future__ import annotations

from dataclasses import dataclass

ANIM_EYE_DIR = (0.722, -0.604, 0.328)
ANIM_EYE_DIST = 1.20
ANIM_AIM_CHEST = (0.0, 0.0, 0.45)


@dataclass(frozen=True)
class PreviewSpec:
    sdf_prims: tuple[str, ...] = ()
    tri_prims: tuple[str, ...] = ()
    work_target: tuple[float, float, float] = (0.0, -0.15, 0.10)
    sdf_resolution: int = 256

    @property
    def look_at(self) -> tuple[float, float, float]:
        return tuple((a + b) / 2 for a, b in zip(self.work_target, ANIM_AIM_CHEST, strict=True))

    @property
    def eye(self) -> tuple[float, float, float]:
        return tuple(a + ANIM_EYE_DIST * b for a, b in zip(self.look_at, ANIM_EYE_DIR, strict=True))
