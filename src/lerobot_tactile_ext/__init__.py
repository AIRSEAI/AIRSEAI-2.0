# SPDX-License-Identifier: Apache-2.0
"""Tactile ``data`` dtype for LeRobot.

``data`` features are lossless 16-bit tactile fields stored as ffv1/mkv.
Upstream LeRobot neither encodes nor decodes them; this extension provides
both:

``codec``
    ffv1/mkv encoder and decoder for uint16 fields, plus incremental
    per-episode streams.
``writer``
    a LeRobot v3.0 dataset writer that mixes video, data and vector
    features and records the quantisation profile in the feature info.

Bit-exactness is the contract: decode(encode(x)) == x, uint16 for uint16.
Anything lossy belongs in a different dtype.
"""

from .codec import decode_data_frames, encode_data_frames

__all__ = ["decode_data_frames", "encode_data_frames"]
