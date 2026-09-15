#!/usr/bin/env python3
"""Generate handover_cup_transfer's USDs: one cup per colour.

Each cup is a cylinder of radius CUP_OUTER_R and height CUP_HEIGHT, bored
from the top down to a CUP_FLOOR-thick floor with the bore radius
`CUP_BORE_R[colour]`, its rim chamfered 45 degrees as a lead-in. Dimensions
and colours come from `sim.dims.handover_cup_transfer`; the files land in
`assets/usds/handover_cup_transfer/cup_<colour>.usd`.

    python benchmark/run/make_handover_cup_transfer_assets.py --headless
    python benchmark/run/make_handover_cup_transfer_assets.py --headless --keep_obj
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.assets.meshio import orient_outward, write_obj, write_usd
from sim.assets.provision import task_asset_dir
from sim.dims.handover_cup_transfer import (
    CUP_BORE_R,
    CUP_CHAMFER,
    CUP_COLOR,
    CUP_FLOOR,
    CUP_HEIGHT,
    CUP_OUTER_R,
)

parser = argparse.ArgumentParser(description="Generate handover_cup_transfer's cups.")
parser.add_argument("--out", default=str(task_asset_dir("handover_cup_transfer")))
parser.add_argument("--sections", type=int, default=48,
                    help="vertices around the cup")
parser.add_argument("--keep_obj", action="store_true",
                    help="also write an OBJ beside each USD, for a mesh viewer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Kit must be running before `write_usd` can import `pxr`.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def build_cup(sections: int, bore_r: float):
    """(points, triangles) for the cup, centred on the origin in xy, base at z = 0.

    Five rings of vertices per sampled angle, bottom to top::

        0  outer_bottom   R_o, z 0
        1  outer_top      R_o, z H
        2  bore_bottom    R_i, z t_floor        (the floor's upper face)
        3  bore_top       R_i, z H - chamfer
        4  rim            R_i + chamfer, z H

    plus one apex vertex each for the base disc and the floor disc.
    """
    ts = [2 * math.pi * i / sections for i in range(sections)]
    n = len(ts)
    pts: list[tuple[float, float, float]] = []
    for t in ts:
        c, s = math.cos(t), math.sin(t)
        pts += [(CUP_OUTER_R * c, CUP_OUTER_R * s, 0.0),
                (CUP_OUTER_R * c, CUP_OUTER_R * s, CUP_HEIGHT),
                (bore_r * c, bore_r * s, CUP_FLOOR),
                (bore_r * c, bore_r * s, CUP_HEIGHT - CUP_CHAMFER),
                ((bore_r + CUP_CHAMFER) * c, (bore_r + CUP_CHAMFER) * s,
                 CUP_HEIGHT)]
    base_apex = len(pts)
    pts.append((0.0, 0.0, 0.0))
    floor_apex = len(pts)
    pts.append((0.0, 0.0, CUP_FLOOR))

    def v(i: int, ring: int) -> int:
        return (i % n) * 5 + ring

    tris: list[tuple[int, int, int]] = []

    def quad(a, b, c_, d):
        tris.append((a, b, c_))
        tris.append((a, c_, d))

    for i in range(n):
        j = i + 1
        quad(v(i, 0), v(j, 0), v(j, 1), v(i, 1))   # outer wall, outward
        quad(v(i, 4), v(i, 1), v(j, 1), v(j, 4))   # top annulus, +z
        quad(v(i, 3), v(i, 4), v(j, 4), v(j, 3))   # the 45 deg rim chamfer
        quad(v(i, 2), v(i, 3), v(j, 3), v(j, 2))   # bore wall, inward
        tris.append((base_apex, v(j, 0), v(i, 0)))  # base disc, -z
        tris.append((floor_apex, v(i, 2), v(j, 2)))  # floor disc, +z
    return pts, tris


def check_closed(tris) -> None:
    """Raise unless every edge is shared by exactly two consistently wound triangles."""
    directed: Counter = Counter()
    for a, b, c in tris:
        directed[(a, b)] += 1
        directed[(b, c)] += 1
        directed[(c, a)] += 1
    if any(v != 1 for v in directed.values()):
        raise AssertionError("winding is inconsistent: a directed edge repeats")
    und = Counter(tuple(sorted(e)) for e in directed)
    if any(v != 2 for v in und.values()):
        raise AssertionError("mesh is not closed")


def main() -> int:
    out = Path(args_cli.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name, colour in CUP_COLOR.items():
        bore = CUP_BORE_R[name]
        pts, tris = build_cup(args_cli.sections, bore)
        check_closed(tris)
        tris, vol = orient_outward(pts, tris, f"cup {name}")
        want = (math.pi * CUP_OUTER_R ** 2 * CUP_HEIGHT
                - math.pi * bore ** 2 * (CUP_HEIGHT - CUP_FLOOR))
        if abs(vol - want) > 0.03 * want:
            raise AssertionError(f"cup {name}: volume {vol * 1e6:.2f} cm3, expected {want * 1e6:.2f}")
        usd = out / f"cup_{name}.usd"
        # "none" keeps the triangles, so the bore survives; the cup is
        # kinematic, which PhysX allows for a triangle-mesh collider.
        write_usd(usd, pts, tris, colour, rigid=True, root_name="cup", approximation="none")
        if args_cli.keep_obj:
            write_obj(usd.with_suffix(".obj"), pts, tris)
        print(f"[cups] {name}: bore {bore * 2e3:.0f} mm, outer {CUP_OUTER_R * 2e3:.0f} mm, "
              f"height {CUP_HEIGHT * 1e3:.0f} mm, {len(pts)} verts, colour {colour} -> {usd.name}",
              flush=True)
    print("CUPS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)       # skip Kit's own shutdown, which can hang
