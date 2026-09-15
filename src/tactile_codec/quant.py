"""float field (mm) <-> uint16, per named, frozen profile.

Two profiles exist and neither may drift:

`daimon_v1`
    What every real-robot dataset already on disk was written with:
    ``clip(x * 1000 + 30000, 0, 65535)`` for depth, shear and deformation
    alike. 1 count = 1 um, representable range -30.000 .. +35.535 mm.
    Frozen verbatim so existing recordings stay decodable.

`sim_v1`
    The simulation profile, using the same 1 um resolution but sane offsets:
    depth is non-negative so its offset is 0; shear/deformation are signed so
    they centre on 32768. Typical signals (penetration <= 3 mm, shear
    <= 2.2 mm) sit an order of magnitude inside the representable range.

Saturation is counted, not silenced: `quantize` returns how many values the
clip touched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["DAIMON_V1", "PROFILES", "SIM_V1", "QuantProfile",
           "dequantize", "quantize"]


@dataclass(frozen=True)
class QuantProfile:
    """uint16 mapping for one channel family: ``u16 = clip(x*scale + offset)``."""

    name: str
    scale: float          # counts per unit (1000.0 => 1 count = 1 um for mm data)
    offset: float         # counts at x == 0
    unit: str = "mm"

    @property
    def lo(self) -> float:
        """Smallest representable value, in `unit`."""
        return (0.0 - self.offset) / self.scale

    @property
    def hi(self) -> float:
        """Largest representable value, in `unit`."""
        return (65535.0 - self.offset) / self.scale

    def to_dict(self) -> dict:
        """For the dataset's info.json, so readers never guess constants."""
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> QuantProfile:
        return QuantProfile(**d)


# The real robot's constants, frozen.
DAIMON_V1_DEPTH = QuantProfile("daimon_v1_depth", scale=1000.0, offset=30000.0)
DAIMON_V1_SHEAR = QuantProfile("daimon_v1_shear", scale=1000.0, offset=30000.0)
DAIMON_V1_DEFORMATION = QuantProfile("daimon_v1_deformation", scale=1000.0,
                                     offset=30000.0)
# Convenience alias: on the real robot all three families share one mapping.
DAIMON_V1 = DAIMON_V1_DEPTH

# Simulation: same resolution, offsets that match the signals' signs.
SIM_V1_DEPTH = QuantProfile("sim_v1_depth", scale=1000.0, offset=0.0)
SIM_V1_SHEAR = QuantProfile("sim_v1_shear", scale=1000.0, offset=32768.0)
SIM_V1_DEFORMATION = QuantProfile("sim_v1_deformation", scale=1000.0,
                                  offset=32768.0)
SIM_V1 = SIM_V1_DEPTH

PROFILES: dict[str, QuantProfile] = {
    p.name: p for p in (
        DAIMON_V1_DEPTH, DAIMON_V1_SHEAR, DAIMON_V1_DEFORMATION,
        SIM_V1_DEPTH, SIM_V1_SHEAR, SIM_V1_DEFORMATION,
    )
}


def quantize(x: np.ndarray, profile: QuantProfile) -> tuple[np.ndarray, int]:
    """float (in `profile.unit`) -> uint16. Returns (array, n_saturated).

    Clipping is silent at the value level (a reader cannot tell a true
    35.535 mm from an overflow), so callers are expected to check the count.
    """
    raw = np.asarray(x, dtype=np.float64) * profile.scale + profile.offset
    n_sat = int(np.count_nonzero((raw < 0.0) | (raw > 65535.0)))
    return np.clip(np.rint(raw), 0.0, 65535.0).astype(np.uint16), n_sat


def dequantize(u: np.ndarray, profile: QuantProfile) -> np.ndarray:
    """uint16 -> float32 in `profile.unit`. Inverse of `quantize` up to 0.5 count."""
    return ((np.asarray(u, dtype=np.float32) - profile.offset)
            / profile.scale).astype(np.float32)
