#!/usr/bin/env python3
"""Generate blind_peg_insertion's USDs: one hole board per clearance in the sweep.

Each board is a BOARD_X x BOARD_Y x BOARD_THICKNESS plate with a through hole
at the centre, diameter `hole_diameter(clearance)`, its top edge chamfered
45 degrees as a lead-in. Dimensions come from `sim.dims.blind_peg_insertion`;
the files land in `assets/usds/blind_peg_insertion/board_hole_<d>mm.usd`.

    python benchmark/run/make_blind_peg_insertion_assets.py --headless
    python benchmark/run/make_blind_peg_insertion_assets.py --headless --clearances 0.001 --keep_obj
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.assets.meshio import orient_outward, write_obj, write_usd
from sim.assets.provision import task_asset_dir
from sim.dims.blind_peg_insertion import (
    BOARD_THICKNESS,
    BOARD_X,
    BOARD_Y,
    CLEARANCE_SWEEP,
    hole_diameter,
)

parser = argparse.ArgumentParser(description="Generate blind_peg_insertion's hole boards.")
parser.add_argument("--clearances", type=float, nargs="*", default=None,
                    help="radial clearances in metres; default is the sweep")
parser.add_argument("--out", default=str(task_asset_dir("blind_peg_insertion")))
parser.add_argument("--sections", type=int, default=96,
                    help="vertices around the hole")
parser.add_argument("--keep_obj", action="store_true",
                    help="also write an OBJ beside each USD, for a mesh viewer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Kit must be running before `write_usd` can import `pxr`.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

CHAMFER = 0.003             # 45 degrees: 3 mm radial over 3 mm deep


def _angles(sections: int) -> list[float]:
    """Evenly spaced angles plus the four rectangle-corner directions, so the
    plate's edges come out straight."""
    hx, hy = BOARD_X / 2, BOARD_Y / 2
    corners = [math.atan2(sy * hy, sx * hx)
               for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    even = [2 * math.pi * i / sections for i in range(sections)]
    return sorted({round(a % (2 * math.pi), 9) for a in even + corners})


def _rect_point(t: float) -> tuple[float, float]:
    """Where a ray from the centre at angle `t` leaves the rectangle."""
    hx, hy = BOARD_X / 2, BOARD_Y / 2
    c, s = math.cos(t), math.sin(t)
    scale = min(hx / abs(c) if abs(c) > 1e-12 else math.inf,
                hy / abs(s) if abs(s) > 1e-12 else math.inf)
    return scale * c, scale * s


def build_board(clearance: float, sections: int):
    """(points, triangles) for the plate, centred on the origin.

    Five rings of vertices per sampled angle, bottom to top::

        0  inner_bottom   hole wall, lower end
        1  inner_top      where the chamfer starts
        2  chamfer_top    the chamfer's mouth, at the top face
        3  outer_bottom   plate edge, lower
        4  outer_top      plate edge, upper
    """
    r = hole_diameter(clearance) / 2.0
    h = BOARD_THICKNESS / 2.0
    z1 = h - CHAMFER
    ts = _angles(sections)
    n = len(ts)

    pts: list[tuple[float, float, float]] = []
    for t in ts:
        c, s = math.cos(t), math.sin(t)
        rx, ry = _rect_point(t)
        rc = r + CHAMFER
        pts += [(r * c, r * s, -h), (r * c, r * s, z1),
                (rc * c, rc * s, h),
                (rx, ry, -h), (rx, ry, h)]

    def v(i: int, ring: int) -> int:
        return (i % n) * 5 + ring

    tris: list[tuple[int, int, int]] = []

    def quad(a, b, c_, d):
        tris.append((a, b, c_))
        tris.append((a, c_, d))

    for i in range(n):
        j = i + 1
        quad(v(i, 2), v(i, 4), v(j, 4), v(j, 2))   # top face, +z
        quad(v(i, 0), v(j, 0), v(j, 3), v(i, 3))   # bottom face, -z
        quad(v(i, 0), v(i, 1), v(j, 1), v(j, 0))   # hole wall, inward
        quad(v(i, 1), v(i, 2), v(j, 2), v(j, 1))   # the 45 deg chamfer
        quad(v(i, 3), v(j, 3), v(j, 4), v(i, 4))   # plate edge, outward
    return pts, tris


def main() -> int:
    out = Path(args_cli.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for c in args_cli.clearances or list(CLEARANCE_SWEEP):
        d_mm = round(hole_diameter(c) * 1000, 2)
        pts, tris = build_board(c, args_cli.sections)
        tris, vol = orient_outward(pts, tris, f"board {d_mm:.2f} mm")
        usd = out / f"board_hole_{d_mm:.2f}mm.usd"
        # "none" keeps the triangles, so the hole survives; the board is
        # kinematic, which PhysX allows for a triangle-mesh collider.
        write_usd(usd, pts, tris, (0.35, 0.38, 0.42), rigid=True,
                  root_name="board", approximation="none")
        if args_cli.keep_obj:
            write_obj(usd.with_suffix(".obj"), pts, tris)
        print(f"[boards] clearance {c * 1000:.2f} mm -> hole {d_mm:.2f} mm, "
              f"{len(pts)} verts, {len(tris)} tris, volume {vol * 1e6:.2f} cm3 -> {usd.name}",
              flush=True)
    print("BOARDS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)       # skip Kit's own shutdown, which can hang
