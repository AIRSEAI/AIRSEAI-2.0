#!/usr/bin/env python3
"""Generate the two rigid bodies of the tool drawer.

python benchmark/run/make_drawer_latch_assets.py --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.assets.meshio import write_usd_parts
from sim.assets.provision import task_asset_dir
from sim.dims.drawer_latch import (
    CASE_SIZE,
    DRAWER_SIZE,
    HANDLE_CENTER_Z,
    HANDLE_DEPTH,
    HANDLE_HEIGHT,
    HANDLE_STANDOFF,
    HANDLE_WIDTH,
)

parser = argparse.ArgumentParser(description="Generate drawer_latch USD assets.")
parser.add_argument("--out", default=str(task_asset_dir("drawer_latch")))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


def box(center, size):
    x, y, z = center
    sx, sy, sz = (v / 2 for v in size)
    pts = [
        (x - sx, y - sy, z - sz), (x + sx, y - sy, z - sz),
        (x + sx, y + sy, z - sz), (x - sx, y + sy, z - sz),
        (x - sx, y - sy, z + sz), (x + sx, y - sy, z + sz),
        (x + sx, y + sy, z + sz), (x - sx, y + sy, z + sz),
    ]
    tris = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return pts, tris


def part(name, center, size, color):
    pts, tris = box(center, size)
    return name, pts, tris, color, "convexHull"


case_color = (0.38, 0.43, 0.48)
drawer_color = (0.78, 0.33, 0.12)
handle_color = (0.13, 0.15, 0.17)
cx, cy, cz = CASE_SIZE
dx, dy, dz = DRAWER_SIZE
case = [
    part("side_left", (0, -cy / 2 + 0.004, cz / 2), (cx, 0.008, cz - 0.004), case_color),
    part("side_right", (0, cy / 2 - 0.004, cz / 2), (cx, 0.008, cz - 0.004), case_color),
    part("top", (0, 0, cz - 0.004), (cx, cy, 0.008), case_color),
    part("bottom", (0, 0, 0.004), (cx, cy - 0.016, 0.008), case_color),
    part("back", (cx / 2 - 0.004, 0, cz / 2), (0.008, cy - 0.016, cz - 0.004), case_color),
]
front = -cx / 2 - 0.001
drawer = [
    part("handle", (front - HANDLE_STANDOFF, 0, HANDLE_CENTER_Z), (HANDLE_DEPTH, HANDLE_WIDTH, HANDLE_HEIGHT), handle_color),
    part("front", (front, 0, 0.010 + dz / 2), (0.010, 0.200, dz), drawer_color),
    part("floor", (-0.008, 0, 0.019), (dx, dy, 0.008), drawer_color),
    part("side_left", (-0.008, -0.092, (0.019 + cz - 0.041) / 2), (dx, 0.006, cz - 0.060), drawer_color),
    part("side_right", (-0.008, 0.092, (0.019 + cz - 0.041) / 2), (dx, 0.006, cz - 0.060), drawer_color),
    part("back", (0.089, 0, (0.019 + cz - 0.041) / 2), (0.006, dy, cz - 0.060), drawer_color),
    part("mount_left", (front - HANDLE_STANDOFF / 2, -0.052, HANDLE_CENTER_Z), (HANDLE_STANDOFF, 0.015, 0.016), handle_color),
    part("mount_right", (front - HANDLE_STANDOFF / 2, 0.052, HANDLE_CENTER_Z), (HANDLE_STANDOFF, 0.015, 0.016), handle_color),
]
out = Path(args.out).expanduser()
write_usd_parts(out / "case.usd", case, rigid=True, root_name="case")
write_usd_parts(out / "drawer.usd", drawer, rigid=True, root_name="drawer", friction=(1.2, 1.0))
app.close()
