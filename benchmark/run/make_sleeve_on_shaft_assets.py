#!/usr/bin/env python3
"""Generate sleeve_on_shaft USD assets: sleeve half-shells and shafts.

python benchmark/run/make_sleeve_on_shaft_assets.py --headless
python benchmark/run/make_sleeve_on_shaft_assets.py --headless --keep_obj
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.assets.meshio import check_closed, orient_outward, write_obj, write_usd
from sim.assets.provision import task_asset_dir
from sim.dims.sleeve_on_shaft import (
    BASE_HEIGHT,
    BASE_RADIUS,
    CLEARANCE_SWEEP,
    SHAFT_CHAMFER,
    SHAFT_COLOR,
    SHAFT_HEIGHT,
    SHELL_MESH_NAMES,
    SLEEVE_COLOR,
    SLEEVE_HEIGHT,
    SLEEVE_INNER_R,
    SLEEVE_OUTER_R,
    shaft_mesh_name,
    shaft_radius,
)

parser = argparse.ArgumentParser(description="Generate sleeve_on_shaft's shells and shafts.")
parser.add_argument("--out", default=str(task_asset_dir("sleeve_on_shaft")))
parser.add_argument("--sectors", type=int, default=96, help="mesh segments over each shell's half circle")
parser.add_argument(
    "--keep_obj", action="store_true", help="also write an OBJ beside each USD, for a mesh viewer"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Kit must be running before `write_usd` can import `pxr`.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Friction authored into the shells: `UsdFileCfg` has no `physics_material`.
SHELL_FRICTION = (1.1, 0.9)
# The shaft is lower-friction so a sleeve in clearance slides rather than
# catching on a light touch of the wall.
SHAFT_FRICTION = (0.4, 0.3)


def build_half_shell(sectors: int, sign: float):
    """(points, triangles) for half a tube, centred on the origin, z in +-H/2.

    `sign` picks the half: +1 keeps y >= 0, -1 mirrors it. Four vertex rings
    per sampled angle: outer bottom, outer top, inner bottom, inner top.
    """
    h = SLEEVE_HEIGHT / 2
    ro, ri = SLEEVE_OUTER_R, SLEEVE_INNER_R
    n = sectors + 1
    pts: list[tuple[float, float, float]] = []
    for j in range(n):
        a = math.pi * j / sectors
        c, s = math.cos(a), sign * math.sin(a)
        pts += [(ro * c, ro * s, -h), (ro * c, ro * s, h), (ri * c, ri * s, -h), (ri * c, ri * s, h)]

    def v(j: int, ring: int) -> int:
        return j * 4 + ring

    tris = []
    for j in range(sectors):
        k = j + 1
        tris += [(v(j, 0), v(k, 0), v(k, 1)), (v(j, 0), v(k, 1), v(j, 1))]  # outer wall
        tris += [(v(j, 2), v(j, 3), v(k, 3)), (v(j, 2), v(k, 3), v(k, 2))]  # inner wall
        tris += [(v(j, 1), v(k, 1), v(k, 3)), (v(j, 1), v(k, 3), v(j, 3))]  # top rim
        tris += [(v(j, 0), v(j, 2), v(k, 2)), (v(j, 0), v(k, 2), v(k, 0))]  # bottom rim
    for j in (0, sectors):  # cut faces
        tris += [(v(j, 0), v(j, 1), v(j, 3)), (v(j, 0), v(j, 3), v(j, 2))]
    # The two cut faces close the half-tube; their winding is fixed below by
    # `orient_outward` only if the whole mesh came out inverted, so make the
    # far face consistent with the near one here.
    tris[-2:] = [(c, b, a) for a, b, c in tris[-2:]]
    if sign < 0:
        tris = [(a, c, b) for a, b, c in tris]
    return pts, tris


def build_shaft(sections: int, radius: float):
    """(points, triangles) for base disc + chamfered shaft, base bottom at z = 0.

    Rings per sampled angle, bottom to top: base outer bottom, base outer top,
    shaft foot (on the base top), shaft chamfer start, shaft top edge.
    """
    zb, zs = BASE_HEIGHT, BASE_HEIGHT + SHAFT_HEIGHT
    rt = radius - SHAFT_CHAMFER
    n = sections
    pts: list[tuple[float, float, float]] = []
    for i in range(n):
        t = 2 * math.pi * i / n
        c, s = math.cos(t), math.sin(t)
        pts += [
            (BASE_RADIUS * c, BASE_RADIUS * s, 0.0),
            (BASE_RADIUS * c, BASE_RADIUS * s, zb),
            (radius * c, radius * s, zb),
            (radius * c, radius * s, zs - SHAFT_CHAMFER),
            (rt * c, rt * s, zs),
        ]
    bottom_apex = len(pts)
    pts.append((0.0, 0.0, 0.0))
    top_apex = len(pts)
    pts.append((0.0, 0.0, zs))

    def v(i: int, ring: int) -> int:
        return (i % n) * 5 + ring

    tris = []
    for i in range(n):
        k = i + 1
        tris += [(bottom_apex, v(k, 0), v(i, 0))]  # base bottom
        tris += [(v(i, 0), v(k, 0), v(k, 1)), (v(i, 0), v(k, 1), v(i, 1))]  # base side
        tris += [(v(i, 1), v(k, 1), v(k, 2)), (v(i, 1), v(k, 2), v(i, 2))]  # base top annulus
        tris += [(v(i, 2), v(k, 2), v(k, 3)), (v(i, 2), v(k, 3), v(i, 3))]  # shaft side
        tris += [(v(i, 3), v(k, 3), v(k, 4)), (v(i, 3), v(k, 4), v(i, 4))]  # chamfer
        tris += [(top_apex, v(i, 4), v(k, 4))]  # shaft top
    return pts, tris


def main() -> int:
    out = Path(args_cli.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rc = 0

    want = 0.5 * math.pi * (SLEEVE_OUTER_R**2 - SLEEVE_INNER_R**2) * SLEEVE_HEIGHT
    for name, sign in zip(SHELL_MESH_NAMES, (+1.0, -1.0), strict=True):
        pts, tris = build_half_shell(args_cli.sectors, sign)
        tris, vol = orient_outward(pts, tris, name)
        check_closed(tris)
        err = abs(vol - want) / want
        flag = "ok" if err < 0.01 else "MISMATCH"
        rc |= flag != "ok"
        usd = out / f"{name}.usd"
        write_usd(usd, pts, tris, SLEEVE_COLOR, rigid=True, root_name="sleeve_half", friction=SHELL_FRICTION)
        if args_cli.keep_obj:
            write_obj(usd.with_suffix(".obj"), pts, tris)
        print(
            f"[shells] {name}: {len(pts)} verts, {len(tris)} tris, volume "
            f"{vol * 1e6:.3f} cm3 vs analytic {want * 1e6:.3f} ({err * 100:.2f} %) "
            f"{flag} -> {usd.name}",
            flush=True,
        )

    for clearance in CLEARANCE_SWEEP:
        r = shaft_radius(clearance)
        pts, tris = build_shaft(2 * args_cli.sectors, r)
        tris, vol = orient_outward(pts, tris, f"shaft {clearance}")
        check_closed(tris)
        cone = (math.pi * SHAFT_CHAMFER / 3) * (r**2 + r * (r - SHAFT_CHAMFER) + (r - SHAFT_CHAMFER) ** 2)
        want_s = (
            math.pi * BASE_RADIUS**2 * BASE_HEIGHT + math.pi * r**2 * (SHAFT_HEIGHT - SHAFT_CHAMFER) + cone
        )
        err = abs(vol - want_s) / want_s
        flag = "ok" if err < 0.02 else "MISMATCH"
        rc |= flag != "ok"
        usd = out / f"{shaft_mesh_name(clearance)}.usd"
        # Static fixture: triangles kept, the scene makes the body kinematic.
        write_usd(
            usd,
            pts,
            tris,
            SHAFT_COLOR,
            rigid=True,
            root_name="shaft",
            friction=SHAFT_FRICTION,
            approximation="none",
        )
        if args_cli.keep_obj:
            write_obj(usd.with_suffix(".obj"), pts, tris)
        print(
            f"[shafts] clearance {clearance * 1e3:.2f} mm -> shaft d {2e3 * r:.2f} mm, "
            f"{len(pts)} verts, volume {vol * 1e6:.2f} cm3 vs {want_s * 1e6:.2f} "
            f"({err * 100:.2f} %) {flag} -> {usd.name}",
            flush=True,
        )

    print("SLEEVE_ASSETS_COMPLETE" if rc == 0 else "SLEEVE_ASSETS_FAILED", flush=True)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)  # skip Kit's own shutdown, which can hang
