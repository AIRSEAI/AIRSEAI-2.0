#!/usr/bin/env python3
"""Generate handover's USDs: the two sleeve half-shells and the cups.

Each shell is half of a tube (outer SLEEVE_OUTER_R, inner SLEEVE_INNER_R,
length SLEEVE_LENGTH along z) cut on the yz plane; the scene spawns the pair
a gap apart along x and joins them with a spring. The source cup is a
hollow cylinder with a floor and a chamfered rim; each target cup adds a
key, a flat bar standing on the floor along y, one mesh per bore in
TARGET_BORE_SWEEP. Dimensions come from `sim.dims.handover`; the files land
in `assets/usds/handover/`.

    python benchmark/run/make_handover_assets.py --headless
    python benchmark/run/make_handover_assets.py --headless --keep_obj
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
from sim.dims.handover import (
    CUP_CHAMFER,
    CUP_COLOR,
    CUP_FLOOR,
    CUP_HEIGHT,
    CUP_OUTER_R,
    KEY_CHAMFER,
    KEY_HALF_LENGTH,
    KEY_HEIGHT,
    KEY_THICKNESS,
    SHELL_MESH_NAMES,
    SLEEVE_COLOR,
    SLEEVE_INNER_R,
    SLEEVE_LENGTH,
    SLEEVE_OUTER_R,
    SOURCE_BORE_R,
    TARGET_BORE_SWEEP,
    target_mesh_name,
)

parser = argparse.ArgumentParser(description="Generate handover's sleeve shells and cups.")
parser.add_argument("--out", default=str(task_asset_dir("handover")))
parser.add_argument("--sectors", type=int, default=64,
                    help="mesh segments over each shell's half circle")
parser.add_argument("--keep_obj", action="store_true",
                    help="also write an OBJ beside each USD, for a mesh viewer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Kit must be running before `write_usd` can import `pxr`.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Friction authored into the shells: `UsdFileCfg` has no `physics_material`.
SHELL_FRICTION = (1.1, 0.9)
# The cups' friction: with the sleeve's spring pressing on the source bore,
# this is what a straight pull has to overcome.
CUP_FRICTION = (0.6, 0.5)
# Spacing of the cup mesh's vertices on the faces the sleeve touches, metres.
CONTACT_STEP = 0.001


def build_half_shell(sectors: int, sign: float):
    """(points, triangles) for half a tube along z, centred on the origin.

    `sign` picks the half: +1 keeps x >= 0, -1 mirrors it. Four vertex rings
    per sampled angle: outer bottom, outer top, inner bottom, inner top.
    """
    h = SLEEVE_LENGTH / 2
    ro, ri = SLEEVE_OUTER_R, SLEEVE_INNER_R
    n = sectors + 1
    pts: list[tuple[float, float, float]] = []
    for j in range(n):
        a = -math.pi / 2 + math.pi * j / sectors
        c, s = sign * math.cos(a), math.sin(a)
        pts += [(ro * c, ro * s, -h), (ro * c, ro * s, h),
                (ri * c, ri * s, -h), (ri * c, ri * s, h)]

    def v(j: int, ring: int) -> int:
        return j * 4 + ring

    tris = []
    for j in range(sectors):
        k = j + 1
        tris += [(v(j, 0), v(k, 0), v(k, 1)), (v(j, 0), v(k, 1), v(j, 1))]   # outer wall
        tris += [(v(j, 2), v(j, 3), v(k, 3)), (v(j, 2), v(k, 3), v(k, 2))]   # inner wall
        tris += [(v(j, 1), v(k, 1), v(k, 3)), (v(j, 1), v(k, 3), v(j, 3))]   # top rim
        tris += [(v(j, 0), v(j, 2), v(k, 2)), (v(j, 0), v(k, 2), v(k, 0))]   # bottom rim
    tris += [(v(0, 0), v(0, 1), v(0, 3)), (v(0, 0), v(0, 3), v(0, 2))]      # cut face at j = 0
    s = sectors
    tris += [(v(s, 0), v(s, 3), v(s, 1)), (v(s, 0), v(s, 2), v(s, 3))]      # cut face at j = sectors
    if sign < 0:
        tris = [(a, c, b) for a, b, c in tris]
    return pts, tris


def cup_profile(bore_r: float, height: float) -> list:
    """The cup's cross-section in (r, z), from the outer bottom edge round to
    just short of the axis on the floor.

    The faces the sleeve touches, the bore wall and the floor, are sampled
    every CONTACT_STEP: a mesh collider is tested against the sleeve's SDF at
    its vertices, so a wall with vertices only at its ends lets the sleeve
    sink into its middle.
    """
    prof = [(CUP_OUTER_R, 0.0), (CUP_OUTER_R, height),
            (bore_r + CUP_CHAMFER, height)]
    prof += [(bore_r, z) for z in steps(height - CUP_CHAMFER, CUP_FLOOR, CONTACT_STEP)]
    prof += [(r, CUP_FLOOR) for r in steps(bore_r, 0.0, CONTACT_STEP)[1:-1]]
    return prof


def build_key():
    """(points, triangles) for the key: a bar KEY_THICKNESS thick in x and
    2 KEY_HALF_LENGTH long in y, standing from just inside the floor to
    KEY_HEIGHT above it, its sides and top sampled every CONTACT_STEP."""
    hx, hy = KEY_THICKNESS / 2, KEY_HALF_LENGTH
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    ring: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1], strict=True):
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / CONTACT_STEP))
        ring += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n)]
    m = len(ring)
    z0, z1 = CUP_FLOOR - 0.001, CUP_FLOOR + KEY_HEIGHT
    levels = steps(z0, z1, CONTACT_STEP)
    # The top cap shrinks toward the centre in rings, so a sleeve landing on
    # it meets vertices there too.
    tops = [1.0 - k / 4 for k in range(1, 4)]

    def end_scale(z: float) -> float:
        """How far the ends are drawn in at height z: the top KEY_CHAMFER of
        each end is bevelled at 45 degrees as a lead-in."""
        cut = max(0.0, z - (z1 - KEY_CHAMFER))
        return (hy - cut) / hy

    pts: list[tuple[float, float, float]] = []
    for z in levels:
        pts += [(x, y * end_scale(z), z) for x, y in ring]
    for f in tops:
        pts += [(x * f, y * end_scale(z1) * f, z1) for x, y in ring]
    bottom_apex = len(pts)
    pts.append((0.0, 0.0, z0))
    top_apex = len(pts)
    pts.append((0.0, 0.0, z1))
    rows = len(levels) + len(tops)

    def v(row: int, i: int) -> int:
        return row * m + (i % m)

    tris: list[tuple[int, int, int]] = []
    for i in range(m):
        j = i + 1
        tris.append((bottom_apex, v(0, j), v(0, i)))
        for r in range(rows - 1):
            tris.append((v(r, i), v(r, j), v(r + 1, j)))
            tris.append((v(r, i), v(r + 1, j), v(r + 1, i)))
        tris.append((top_apex, v(rows - 1, i), v(rows - 1, j)))
    return pts, tris


def build_cup(sections: int, bore_r: float, height: float, key: bool):
    """(points, triangles) for a cup `height` tall, centred in xy, base at z = 0,
    with the key standing on its floor when `key`.

    The profile from `cup_profile` revolved about z, closed by a disc at the
    base and one at the floor's centre; the key is a second closed shell.
    """
    prof = cup_profile(bore_r, height)
    m = len(prof)
    pts: list[tuple[float, float, float]] = []
    for i in range(sections):
        t = 2 * math.pi * i / sections
        c, s = math.cos(t), math.sin(t)
        pts += [(r * c, r * s, z) for r, z in prof]
    base_apex = len(pts)
    pts.append((0.0, 0.0, 0.0))
    end_apex = len(pts)
    pts.append((0.0, 0.0, CUP_FLOOR))

    def v(i: int, k: int) -> int:
        return (i % sections) * m + k

    tris: list[tuple[int, int, int]] = []
    for i in range(sections):
        j = i + 1
        tris.append((base_apex, v(j, 0), v(i, 0)))
        for k in range(m - 1):
            tris.append((v(i, k), v(j, k), v(j, k + 1)))
            tris.append((v(i, k), v(j, k + 1), v(i, k + 1)))
        tris.append((end_apex, v(i, m - 1), v(j, m - 1)))
    tris, _vol = orient_outward(pts, tris, "cup body")
    if key:
        kp, kt = build_key()
        kt, _kvol = orient_outward(kp, kt, "key")
        base = len(pts)
        pts += kp
        tris += [(a + base, b + base, c + base) for a, b, c in kt]
    return pts, tris


def cup_volume(bore_r: float, height: float, key: bool) -> float:
    """Analytic volume of a cup, chamfers ignored, m^3."""
    vol = math.pi * CUP_OUTER_R ** 2 * height - math.pi * bore_r ** 2 * (height - CUP_FLOOR)
    if key:
        vol += KEY_THICKNESS * 2 * KEY_HALF_LENGTH * (KEY_HEIGHT + 0.001)
    return vol


def main() -> int:
    out = Path(args_cli.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rc = 0

    want = 0.5 * math.pi * (SLEEVE_OUTER_R ** 2 - SLEEVE_INNER_R ** 2) * SLEEVE_LENGTH
    for name, sign in zip(SHELL_MESH_NAMES, (+1.0, -1.0), strict=True):
        pts, tris = build_half_shell(args_cli.sectors, sign)
        check_closed(tris)
        tris, vol = orient_outward(pts, tris, name)
        err = abs(vol - want) / want
        flag = "ok" if err < 0.01 else "MISMATCH"
        rc |= flag != "ok"
        usd = out / f"{name}.usd"
        write_usd(usd, pts, tris, SLEEVE_COLOR, rigid=True, root_name="sleeve_half",
                  friction=SHELL_FRICTION)
        if args_cli.keep_obj:
            write_obj(usd.with_suffix(".obj"), pts, tris)
        print(f"[shells] {name}: {len(pts)} verts, volume {vol * 1e6:.3f} cm3 vs "
              f"{want * 1e6:.3f} ({err * 100:.2f} %) {flag} -> {usd.name}", flush=True)

    cups = [("cup_source", SOURCE_BORE_R, "source", False)]
    cups += [(target_mesh_name(d), d / 2, "target", True) for d in TARGET_BORE_SWEEP]
    for name, bore_r, role, key in cups:
        colour, height = CUP_COLOR[role], CUP_HEIGHT[role]
        pts, tris = build_cup(4 * args_cli.sectors, bore_r, height, key)
        check_closed(tris)
        tris, vol = orient_outward(pts, tris, name)
        want_c = cup_volume(bore_r, height, key)
        err = abs(vol - want_c) / want_c
        flag = "ok" if err < 0.03 else "MISMATCH"
        rc |= flag != "ok"
        usd = out / f"{name}.usd"
        # Static fixture: triangles kept, so the bore survives; the scene
        # makes the body kinematic.
        write_usd(usd, pts, tris, colour, rigid=True, root_name="cup",
                  friction=CUP_FRICTION, approximation="none")
        if args_cli.keep_obj:
            write_obj(usd.with_suffix(".obj"), pts, tris)
        print(f"[cups] {name}: bore {bore_r * 2e3:.1f} mm, key {key}, {len(pts)} verts, "
              f"volume {vol * 1e6:.2f} cm3 vs {want_c * 1e6:.2f} ({err * 100:.2f} %) {flag} "
              f"-> {usd.name}", flush=True)

    print("HANDOVER_ASSETS_COMPLETE" if rc == 0 else "HANDOVER_ASSETS_FAILED", flush=True)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)       # skip Kit's own shutdown, which can hang
