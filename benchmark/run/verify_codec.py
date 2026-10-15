#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that the tactile codec round-trips data. No Kit, no GPU, no robot.

  bit_exact_*            uint16 in == uint16 out, for 1ch, 2ch and 3ch data.
  zero_saturation        quantising realistic sim signals under `sim_v1`
                         clips zero values.
  quant_roundtrip_half_count
                         dequantize(quantize(x)) is within half a count
                         (0.5 um) everywhere in the representable range.

The compression ratio is recorded, not asserted; the inputs are synthetic
fields.

    python benchmark/run/verify_codec.py --out ~/codec_out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from lerobot_tactile_ext import decode_data_frames, encode_data_frames
from tactile_codec import PROFILES, dequantize, quantize

FPS = 60


def synthetic_fields(rng: np.ndarray, n: int = 40, h: int = 288, w: int = 384):
    """Contact-like float fields in mm: a wandering Gaussian dent + noise floor.

    Contact-shaped rather than white noise, so the compression ratio is
    representative of real datasets.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.zeros((n, h, w), np.float32)
    shear = np.zeros((n, h, w, 2), np.float32)
    for i in range(n):
        cy, cx = h * (0.4 + 0.2 * np.sin(i / 7)), w * (0.5 + 0.2 * np.cos(i / 9))
        r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (0.02 * h * w)
        dent = 2.5 * np.exp(-r2) * min(i / 10, 1.0)          # <= 2.5 mm
        # Penetration is non-negative by construction (TacSL clamps at zero),
        # so the noise floor is clipped too; negative values would saturate
        # the depth profile's clip.
        depth[i] = np.clip(dent + rng.normal(0, 1e-3, (h, w)), 0.0, None)
        shear[i, ..., 0] = 0.8 * dent * np.sin(i / 5)
        shear[i, ..., 1] = -0.5 * dent * np.cos(i / 6)
    return depth, shear


def roundtrip(name: str, u16: np.ndarray, out_dir: Path, checks: dict,
              original_channels: int | None = None) -> None:
    path = out_dir / f"{name}.mkv"
    stats = encode_data_frames(u16, path, FPS)
    back = decode_data_frames(path, original_channels=original_channels)
    ok = bool(back.shape == u16.squeeze().shape if u16.ndim == 4 and u16.shape[-1] == 1
              else back.shape == u16.shape) and bool(np.array_equal(
                  back, u16.squeeze() if (u16.ndim == 4 and u16.shape[-1] == 1) else u16))
    checks[f"bit_exact_{name}"] = {
        "pass": ok, "frames": stats["frames"], "pix_fmt": stats["pix_fmt"],
        "compression_ratio": round(stats["compression_ratio"], 2),
        "encoded_mb": round(stats["bytes"] / 1e6, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tactile codec roundtrip gate.")
    ap.add_argument("--out", default=os.path.expanduser("~/codec_out"))
    a = ap.parse_args()
    out_dir = Path(os.path.expanduser(a.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    checks: dict = {}

    depth_mm, shear_mm = synthetic_fields(rng)

    # --- quantisation ----------------------------------------------------
    dq, d_sat = quantize(depth_mm, PROFILES["sim_v1_depth"])
    sq, s_sat = quantize(shear_mm, PROFILES["sim_v1_shear"])
    checks["zero_saturation"] = {
        "pass": d_sat == 0 and s_sat == 0,
        "depth_saturated": d_sat, "shear_saturated": s_sat,
        "depth_range_mm": [float(depth_mm.min()), float(depth_mm.max())],
        "shear_range_mm": [float(shear_mm.min()), float(shear_mm.max())],
    }

    err_d = np.abs(dequantize(dq, PROFILES["sim_v1_depth"]) - depth_mm).max()
    err_s = np.abs(dequantize(sq, PROFILES["sim_v1_shear"]) - shear_mm).max()
    # daimon_v1 too: same numbers must survive the real robot's mapping
    ddq, _ = quantize(depth_mm, PROFILES["daimon_v1_depth"])
    err_dd = np.abs(dequantize(ddq, PROFILES["daimon_v1_depth"]) - depth_mm).max()
    # Half a count, plus 1 nm of slack: the dequantized value is float32 and
    # carries ~2e-8 mm of representation error on top of the mathematically
    # exact half-count bound.
    half_count = 0.5 / 1000.0
    checks["quant_roundtrip_half_count"] = {
        "pass": bool(max(err_d, err_s, err_dd) <= half_count + 1e-6),
        "max_err_mm": {"sim_depth": float(err_d), "sim_shear": float(err_s),
                       "daimon_depth": float(err_dd)},
    }

    # --- codec bit-exactness, all channel layouts ------------------------
    roundtrip("depth_1ch", dq, out_dir, checks)
    roundtrip("shear_2ch", sq, out_dir, checks, original_channels=2)
    rgbish = rng.integers(0, 65536, (12, 96, 128, 3), dtype=np.uint16)
    roundtrip("random_3ch", rgbish, out_dir, checks)

    report = {"profiles": {k: v.to_dict() for k, v in PROFILES.items()},
              "checks": checks}
    with open(out_dir / "tactile_codec_roundtrip.json", "w") as f:
        json.dump(report, f, indent=2)

    print("===== TACTILE CODEC ROUNDTRIP =====", flush=True)
    ok_all = True
    for k, v in checks.items():
        ok_all &= v["pass"]
        extra = ""
        if "compression_ratio" in v:
            extra = f"  ratio {v['compression_ratio']}x, {v['encoded_mb']} MB"
        print(f"  {k:28s} {'PASS' if v['pass'] else 'FAIL'}{extra}", flush=True)
    print("CHECK_COMPLETE", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
