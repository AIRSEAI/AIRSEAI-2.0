#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate texture_sort's USDs: the two pucks and the sorting cup.

Each puck is a flat-capped cylinder whose side radius is
`R - a * (0.5 - 0.5 * cos(k_theta * theta) * cos(k_z * 2 pi z / H))`, a regular
field of shallow dimples (`a = 0` is the smooth puck). The cup is a hollow
cylinder revolved from a profile. Dimensions come from `sim.dims.texture_sort`;
the files land in `assets/usds/texture_sort/{puck_smooth,puck_rough,sort_cup}.usd`.

    python benchmark/run/make_texture_sort_assets.py --headless
    python benchmark/run/make_texture_sort_assets.py --headless --keep_obj
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
from sim.dims.texture_sort import (
    CUP_COLOR,
    CUP_FLOOR,
    CUP_HEIGHT,
    CUP_INNER_R,
    CUP_WALL,
    OBJECT_SPECS,
)

parser = argparse.ArgumentParser(description="Generate texture_sort's pucks and cup.")
parser.add_argument("--out", default=str(task_asset_dir("texture_sort")))
parser.add_argument("--rings", type=int, default=64,
                    help="vertex rows along a puck's height")
parser.add_argument("--sectors", type=int, default=120,
                    help="vertices around the circumference")
parser.add_argument("--keep_obj", action="store_true",
                    help="also write an OBJ beside each USD, for a mesh viewer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Kit must be running before `write_usd` can import `pxr`.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The cup's cross-section, (r, z) from the axis out and back to the axis.
CUP_PROFILE = [(0.0, 0.0),
               (CUP_INNER_R + CUP_WALL, 0.0),
               (CUP_INNER_R + CUP_WALL, CUP_HEIGHT),
               (CUP_INNER_R, CUP_HEIGHT),
               (CUP_INNER_R, CUP_FLOOR),
               (0.0, CUP_FLOOR)]


def build_puck(radius: float, height: float, amplitude: float, k_t: int,
               k_z: int, rings: int, sectors: int):
    """(points, triangles) for a dimpled cylinder, base on z = 0.

    The side is a grid of `rings + 1` rows by `sectors` columns with the
    radius modulated per vertex; the two rows nearest a cap carry no dimples,
    so each cap is a planar fan at the unmodulated radius.
    """
    pts: list[tuple[float, float, float]] = []
    for i in range(rings + 1):
        z = height * i / rings
        edge = i == 0 or i == rings
        for j in range(sectors):
            th = 2.0 * math.pi * j / sectors
            dimple = 0.0 if edge else (
                0.5 - 0.5 * math.cos(k_t * th) * math.cos(k_z * 2.0 * math.pi * z / height))
            r = radius - amplitude * dimple
            pts.append((r * math.cos(th), r * math.sin(th), z))
    bottom = len(pts)
    pts.append((0.0, 0.0, 0.0))
    top = len(pts)
    pts.append((0.0, 0.0, height))

    def v(i: int, j: int) -> int:
        return i * sectors + (j % sectors)

    tris: list[tuple[int, int, int]] = []
    for i in range(rings):
        for j in range(sectors):
            a, b, c, d = v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)
            tris.append((a, b, c))
            tris.append((a, c, d))
    for j in range(sectors):
        tris.append((bottom, v(0, j + 1), v(0, j)))
        tris.append((top, v(rings, j), v(rings, j + 1)))
    return pts, tris


def revolve(profile, sectors: int):
    """Revolve a 2-D profile [(r, z), ...] about the z axis.

    The profile is an open polyline from the axis, out and around, and back
    to the axis; both endpoints must have r == 0 and become the two pole
    vertices that close the surface.
    """
    if profile[0][0] != 0.0 or profile[-1][0] != 0.0:
        raise ValueError("profile must start and end on the axis (r = 0)")
    mid = profile[1:-1]
    pts: list[tuple[float, float, float]] = [(0.0, 0.0, profile[0][1])]
    for r, z in mid:
        for j in range(sectors):
            a = 2.0 * math.pi * j / sectors
            pts.append((r * math.cos(a), r * math.sin(a), z))
    pts.append((0.0, 0.0, profile[-1][1]))
    top = len(pts) - 1

    def v(i: int, j: int) -> int:
        return 1 + i * sectors + (j % sectors)

    tris: list[tuple[int, int, int]] = []
    for j in range(sectors):                       # bottom pole fan
        tris.append((0, v(0, j + 1), v(0, j)))
    for i in range(len(mid) - 1):                  # the bands between rings
        for j in range(sectors):
            a, b, c, d = v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)
            tris.append((a, b, c))
            tris.append((a, c, d))
    for j in range(sectors):                       # top pole fan
        tris.append((top, v(len(mid) - 1, j), v(len(mid) - 1, j + 1)))
    return pts, tris


def main() -> int:
    out = Path(args_cli.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for spec in OBJECT_SPECS.values():
        pts, tris = build_puck(spec.radius, spec.height, spec.dimple_depth,
                               spec.k_theta, spec.k_z,
                               args_cli.rings, args_cli.sectors)
        tris, vol = orient_outward(pts, tris, spec.name)
        usd = out / f"{spec.name}.usd"
        write_usd(usd, pts, tris, spec.colour, rigid=True, root_name="body")
        if args_cli.keep_obj:
            write_obj(usd.with_suffix(".obj"), pts, tris)
        # The dimples take a little volume off, so `rough` reads a few
        # percent under the plain cylinder.
        want = math.pi * spec.radius ** 2 * spec.height
        print(f"[objects] {spec.name}: r {spec.radius * 1e3:.1f} mm, h "
              f"{spec.height * 1e3:.0f} mm, dimple {spec.dimple_depth * 1e3:.2f} mm, "
              f"{len(pts)} verts, {len(tris)} tris, volume {vol * 1e6:.2f} cm3 "
              f"(cylinder {want * 1e6:.2f}) -> {usd.name}", flush=True)

    pts, tris = revolve(CUP_PROFILE, args_cli.sectors)
    tris, vol = orient_outward(pts, tris, "sort_cup")
    usd = out / "sort_cup.usd"
    # Rigid, so the scene can make the cups kinematic and re-place them.
    write_usd(usd, pts, tris, CUP_COLOR, rigid=True, root_name="body")
    if args_cli.keep_obj:
        write_obj(usd.with_suffix(".obj"), pts, tris)
    solid = math.pi * (CUP_INNER_R + CUP_WALL) ** 2 * CUP_HEIGHT
    print(f"[objects] sort_cup: {len(pts)} verts, {len(tris)} tris, "
          f"volume {vol * 1e6:.2f} cm3 (solid cylinder would be "
          f"{solid * 1e6:.2f}) -> {usd.name}", flush=True)
    print("OBJECTS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)       # skip Kit's own shutdown, which can hang
