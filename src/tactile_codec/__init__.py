"""Quantisation profiles for tactile fields, shared by sim and real robot.

One table, both directions. `to_dict()` is what goes into the dataset's
`info.json`, so a reader never has to guess.
"""

from .quant import (
    DAIMON_V1,
    PROFILES,
    SIM_V1,
    QuantProfile,
    dequantize,
    quantize,
)

__all__ = ["DAIMON_V1", "PROFILES", "SIM_V1", "QuantProfile",
           "dequantize", "quantize"]
