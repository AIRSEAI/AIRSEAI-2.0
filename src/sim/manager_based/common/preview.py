"""Synchronous review grid: images and metadata share the same simulation tick."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def grid_frame(frames, sim_time, state_ids=None, statuses=None):
    n, h, w = frames.shape[:3]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = Image.new("RGB", (cols * w, rows * (h + 30)), "black")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for slot, frame in enumerate(frames):
        x, y = (slot % cols) * w, (slot // cols) * (h + 30)
        canvas.paste(Image.fromarray(frame[..., :3].astype("uint8")), (x, y + 30))
        label = f"env {slot} | sim {sim_time:.3f}s"
        if state_ids is not None:
            label += f" | state {state_ids[slot]}"
        if statuses is not None:
            label += f" | {statuses[slot]}"
        draw.text((x + 8, y + 4), label, fill="white", font=font)
    return np.asarray(canvas)
