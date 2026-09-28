#!/usr/bin/env python3
"""Generate cap_unscrew USD assets: bottle, cap and tray.

python benchmark/run/make_cap_unscrew_assets.py --headless
python benchmark/run/make_cap_unscrew_assets.py --headless --keep_obj
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.assets.meshio import check_closed, orient_outward, steps, write_obj, write_usd
from sim.assets.provision import task_asset_dir
from sim.dims.cap_unscrew import (
    BOTTLE_BODY_H,
    BOTTLE_COLOR,
    BOTTLE_H,
    BOTTLE_R,
    CAP_BORE_DEPTH,
    CAP_BORE_R,
    CAP_COLOR,
    CAP_H,
    CAP_R,
    NECK_R,
    TRAY_COLOR,
    TRAY_FLOOR,
    TRAY_HEIGHT,
    TRAY_INNER_R,
    TRAY_OUTER_R,
)

parser = argparse.ArgumentParser(description="Generate cap_unscrew's bottle, cap and tray.")
parser.add_argument("--out", default=str(task_asset_dir("cap_unscrew")))
parser.add_argument("--sections", type=int, default=128, help="mesh segments around each solid of revolution")
parser.add_argument(
    "--keep_obj", action="store_true", help="also write an OBJ beside each USD, for a mesh viewer"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Kit must be running before `write_usd` can import `pxr`.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Friction authored into the gripped parts: `UsdFileCfg` has no `physics_material`.
GRIP_FRICTION = (1.1, 0.9)
# The tray's friction, so a cap set down in it stays put.
TRAY_FRICTION = (0.6, 0.5)
# Spacing of the tray mesh's vertices on the faces the cap lands on, metres:
# a mesh collider is tested against the cap's SDF at its vertices, so a
# sparsely sampled floor lets the cap sink into it.
CONTACT_STEP = 0.001


def revolve(prof, sections: int):
    """(points, triangles) for the closed solid swept by profile `prof`.

    `prof` is a list of (r, z) that starts and ends on the axis (r = 0); its
    ends become the two apexes and every point between them a ring.
    """
    (_, z0), ring, (_, z1) = prof[0], prof[1:-1], prof[-1]
    m = len(ring)
    pts: list[tuple[float, float, float]] = []
    for i in range(sections):
        t = 2 * math.pi * i / sections
        c, s = math.cos(t), math.sin(t)
        pts += [(r * c, r * s, z) for r, z in ring]
    start = len(pts)
    pts.append((0.0, 0.0, z0))
    end = len(pts)
    pts.append((0.0, 0.0, z1))

    def v(i: int, k: int) -> int:
        return (i % sections) * m + k

    tris: list[tuple[int, int, int]] = []
    for i in range(sections):
        j = i + 1
        tris.append((start, v(j, 0), v(i, 0)))
        for k in range(m - 1):
            tris.append((v(i, k), v(j, k), v(j, k + 1)))
            tris.append((v(i, k), v(j, k + 1), v(i, k + 1)))
        tris.append((end, v(i, m - 1), v(j, m - 1)))
    return pts, tris


def revolved_volume(prof) -> float:
    """Exact volume of the solid swept by `prof`, by Pappus's theorem, m^3."""
    area = moment = 0.0
    for (r0, z0), (r1, z1) in zip(prof, prof[1:] + prof[:1], strict=True):
        cross = r0 * z1 - r1 * z0
        area += cross
        moment += (r0 + r1) * cross
    area /= 2.0
    centroid_r = moment / (6.0 * area)
    return abs(2 * math.pi * centroid_r * area)


def bottle_profile() -> list:
    """Base centre, round the body and neck, to the neck's top centre."""
    return [
        (0.0, 0.0),
        (BOTTLE_R, 0.0),
        (BOTTLE_R, BOTTLE_BODY_H),
        (NECK_R, BOTTLE_BODY_H),
        (NECK_R, BOTTLE_H),
        (0.0, BOTTLE_H),
    ]


def cap_profile() -> list:
    """Roof's underside, down the bore, up the skirt, to the top centre."""
    return [
        (0.0, CAP_BORE_DEPTH),
        (CAP_BORE_R, CAP_BORE_DEPTH),
        (CAP_BORE_R, 0.0),
        (CAP_R, 0.0),
        (CAP_R, CAP_H),
        (0.0, CAP_H),
    ]


def tray_profile() -> list:
    """Base centre, round the rim, down the inner wall and across the floor,
    both sampled every CONTACT_STEP, to the floor's centre."""
    prof = [(0.0, 0.0), (TRAY_OUTER_R, 0.0), (TRAY_OUTER_R, TRAY_HEIGHT)]
    prof += [(TRAY_INNER_R, z) for z in steps(TRAY_HEIGHT, TRAY_FLOOR, CONTACT_STEP)]
    prof += [(r, TRAY_FLOOR) for r in steps(TRAY_INNER_R, 0.0, CONTACT_STEP)[1:]]
    return prof


def main() -> int:
    out = Path(args_cli.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    # (name, profile, colour, root, friction, approximation, sections)
    parts = [
        ("bottle", bottle_profile(), BOTTLE_COLOR, "bottle", GRIP_FRICTION, None, args_cli.sections),
        ("cap", cap_profile(), CAP_COLOR, "cap", GRIP_FRICTION, None, args_cli.sections),
        # Static fixture: triangles kept, so the dish is not filled in.
        ("tray", tray_profile(), TRAY_COLOR, "tray", TRAY_FRICTION, "none", 2 * args_cli.sections),
    ]
    for name, prof, colour, root, friction, approx, sections in parts:
        pts, tris = revolve(prof, sections)
        check_closed(tris)
        tris, vol = orient_outward(pts, tris, name)
        want = revolved_volume(prof)
        err = abs(vol - want) / want
        flag = "ok" if err < 0.01 else "MISMATCH"
        rc |= flag != "ok"
        usd = out / f"{name}.usd"
        write_usd(usd, pts, tris, colour, rigid=True, root_name=root, friction=friction, approximation=approx)
        if args_cli.keep_obj:
            write_obj(usd.with_suffix(".obj"), pts, tris)
        print(
            f"[{name}] {len(pts)} verts, volume {vol * 1e6:.2f} cm3 vs "
            f"{want * 1e6:.2f} ({err * 100:.2f} %) {flag} -> {usd.name}",
            flush=True,
        )

    print("CAP_UNSCREW_ASSETS_COMPLETE" if rc == 0 else "CAP_UNSCREW_ASSETS_FAILED", flush=True)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)  # skip Kit's own shutdown, which can hang
