#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add the human-viewable preview streams to a recorded dataset, in place.

For each tactile ``data`` stream, writes an ``observation.images.*`` mp4 that
renders it for the eye, as the real robot's recordings carry. Display
constants:

  depth        (depth_mm * 0.25 * 255) -> "hot" colormap
  shear /      green arrows on black, one per 16-px cell, arrow length =
  deformation  displacement_mm * 20 px

No Kit, no GPU:

    python benchmark/run/render_previews.py --dataset ~/datasets/sim_stack_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot_tactile_ext import decode_data_frames
from lerobot_tactile_ext.codec import Mp4Stream
from tactile_codec import QuantProfile, dequantize

# Real recordings carry shear/deformation (displacement, mm); sim recordings
# carry ff_shear (TacSL force field, x1e3). Whichever is present is drawn.
FIELDS = ("depth", "shear", "deformation", "ff_shear")


def hot_colormap(v01: np.ndarray) -> np.ndarray:
    """(H,W) in [0,1] -> (H,W,3) uint8, OpenCV COLORMAP_HOT-style ramp."""
    r = np.clip(3.0 * v01, 0, 1)
    g = np.clip(3.0 * v01 - 1.0, 0, 1)
    b = np.clip(3.0 * v01 - 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def draw_arrows(disp_mm: np.ndarray, step: int = 16,
                gain: float = 20.0) -> np.ndarray:
    """(H,W,2) mm -> green arrows on black, PIL line drawing."""
    from PIL import Image, ImageDraw

    h, w = disp_mm.shape[:2]
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = disp_mm[y, x] * gain
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue
            x1, y1 = x + dx, y + dy
            d.line([(x, y), (x1, y1)], fill=(0, 255, 0), width=1)
            # arrow head: two short strokes back from the tip
            v = np.array([dx, dy])
            n = np.linalg.norm(v)
            u = v / n
            for side in (np.array([-u[1], u[0]]), np.array([u[1], -u[0]])):
                tip = np.array([x1, y1]) - u * min(4.0, n * 0.4) + side * 2.0
                d.line([(x1, y1), tuple(tip)], fill=(0, 255, 0), width=1)
    return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render tactile preview streams.")
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    root = Path(a.dataset).expanduser()
    info = json.load(open(root / "meta" / "info.json"))
    fps = int(info["fps"])

    sensors = sorted({k.split(".")[2].removesuffix("_tactile")
                      for k in info["features"]
                      if k.startswith("observation.depth.")})
    made = []
    for s in sensors:
        for field in FIELDS:
            src_key = f"observation.{field}.{s}_tactile"
            if src_key not in info["features"]:
                continue
            spec = info["features"][src_key]
            u16 = decode_data_frames(
                root / "videos" / src_key / "chunk-000" / "file-000.mkv",
                original_channels=spec["info"]["original_channels"])
            phys = dequantize(u16, QuantProfile(**spec["info"]["quant"]))

            dst_key = f"observation.images.{field}_{s}"
            out = root / "videos" / dst_key / "chunk-000" / "file-000.mp4"
            stream = Mp4Stream(out, fps)
            for i in range(phys.shape[0]):
                if field == "depth":
                    stream.add(hot_colormap(np.clip(phys[i] * 0.25, 0, 1)))
                else:
                    stream.add(draw_arrows(phys[i]))
            st = stream.close()
            h, w = phys.shape[1:3]
            info["features"][dst_key] = {
                "dtype": "video", "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": h, "video.width": w,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.fps": fps,
                         "video.channels": 3, "has_audio": False,
                         # display constants, so the preview is reproducible
                         "preview_of": src_key,
                         "display": ("hot(depth*0.25)" if field == "depth"
                                     else "arrows(disp*20px, step 16)")},
            }
            made.append((dst_key, st["frames"]))
            print(f"[previews] {dst_key}: {st['frames']} frames", flush=True)

    # episodes parquet: the new streams share their source stream's timing
    import pyarrow as pa
    import pyarrow.parquet as pq
    for ep_file in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        t = pq.read_table(ep_file)
        cols = {n: t[n] for n in t.column_names}
        for dst_key, _ in made:
            src_key = info["features"][dst_key]["info"]["preview_of"]
            for c in ("chunk_index", "file_index", "from_timestamp",
                      "to_timestamp"):
                name = f"videos/{dst_key}/{c}"
                if name not in cols:
                    cols[name] = t[f"videos/{src_key}/{c}"]
        pq.write_table(pa.table(cols), ep_file)

    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[previews] added {len(made)} streams to {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
