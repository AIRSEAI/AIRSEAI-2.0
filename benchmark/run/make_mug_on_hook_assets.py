#!/usr/bin/env python3
"""Generate mug_on_hook USD assets: mug, rail and coaster.

python benchmark/run/make_mug_on_hook_assets.py --headless
python benchmark/run/make_mug_on_hook_assets.py --headless --keep_obj
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.assets.meshio import orient_outward, write_obj, write_usd_parts
from sim.assets.provision import task_asset_dir
from sim.dims.mug_on_hook import (
    COASTER_COLOR,
    COASTER_FLOOR,
    COASTER_HEIGHT,
    COASTER_INNER_R,
    COASTER_OUTER_R,
    EYE_FLOOR_Z,
    EYE_HEIGHT,
    EYE_X,
    HANDLE_CORNER_R,
    HANDLE_ROOT_R,
    HANDLE_SECTION,
    MUG_COLOR,
    MUG_FLOOR,
    MUG_HEIGHT,
    MUG_INNER_R,
    MUG_OUTER_R,
    PEG_COLOR,
    PEG_FALL_DEG,
    PEG_LENGTH,
    PEG_R,
    PEG_ROOT_Z,
    RAIL_CAP_H,
    RAIL_CAP_OVERHANG,
    RAIL_COLOR,
    RAIL_HEIGHT,
    RAIL_THICKNESS,
    RAIL_WIDTH,
    STOP_BALL_R,
)

parser = argparse.ArgumentParser(description="Generate mug_on_hook's mug, rail and coaster.")
parser.add_argument("--out", default=str(task_asset_dir("mug_on_hook")))
parser.add_argument("--sections", type=int, default=96, help="mesh segments around each solid of revolution")
parser.add_argument(
    "--eye_x", type=float, default=EYE_X, help="clear width of the handle's eye, metres (the task's knob)"
)
parser.add_argument(
    "--keep_obj", action="store_true", help="also write an OBJ beside each USD, for a mesh viewer"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Kit must be running before `write_usd_parts` can import `pxr`.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Friction authored into the mug: it is the only part a gel touches, and the
# only one whose grip has to hold against its own weight. High, because flat
# gels hold a cylinder along a line and creep at the usual figure.
MUG_FRICTION = (1.5, 1.3)
# The rail's: a polished rod, low enough that a hung mug runs down the fall
# towards the stop ball rather than sticking where it was put. Combined by
# `min`, because the mug is deliberately grippy for the gels and PhysX would
# otherwise average the two and give the rod most of the mug's friction.
RAIL_FRICTION = (0.32, 0.30)
RAIL_FRICTION_COMBINE = "min"
# The coaster's, so a mug set into it stays put.
COASTER_FRICTION = (0.7, 0.6)
# Spacing of the coaster mesh's vertices on the faces the mug stands on,
# metres: a mesh collider is tested against the mug's SDF at its vertices, so
# a sparsely sampled floor lets the mug sink into it.
CONTACT_STEP = 0.0015
# Segments per quarter turn of the handle's rounded corners.
CORNER_STEPS = 6
# Facets around the handle's round section, and steps over the stop ball.
SECTION_SIDES = 12
BALL_STEPS = 7


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


def _steps(a: float, b: float, step: float) -> list[float]:
    """Values from `a` to `b` inclusive, at most `step` apart."""
    n = max(1, math.ceil(abs(b - a) / step))
    return [a + (b - a) * k / n for k in range(n + 1)]


def revolve(prof, sections: int):
    """(points, triangles) for the closed solid swept by profile `prof` about z.

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


def box(lo, hi):
    """(points, triangles) for the axis-aligned box between corners `lo` and `hi`."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    pts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    tris = [
        (0, 2, 1),
        (0, 3, 2),  # bottom
        (4, 5, 6),
        (4, 6, 7),  # top
        (0, 1, 5),
        (0, 5, 4),  # -y
        (1, 2, 6),
        (1, 6, 5),  # +x
        (2, 3, 7),
        (2, 7, 6),  # +y
        (3, 0, 4),
        (3, 4, 7),
    ]  # -x
    return pts, tris


def sweep(path, diameter: float, sides: int = SECTION_SIDES):
    """(points, triangles) for a round section of `diameter` swept along `path`.

    `path` is an open list of (x, z); the section lies across the path's local
    tangent, spanning y and the in-plane normal, and both ends are capped, so
    the result is one closed surface. Corners must turn no tighter than the
    section's radius.
    """
    n = len(path)
    r = diameter / 2.0
    ring = [(math.cos(2 * math.pi * j / sides), math.sin(2 * math.pi * j / sides)) for j in range(sides)]
    pts: list[tuple[float, float, float]] = []
    for i, (px, pz) in enumerate(path):
        ax, az = path[max(i - 1, 0)]
        bx, bz = path[min(i + 1, n - 1)]
        tx, tz = bx - ax, bz - az
        norm = math.hypot(tx, tz) or 1.0
        # In-plane normal: the tangent turned a quarter turn.
        nx, nz = -tz / norm, tx / norm
        for cn, cy in ring:
            pts.append((px + r * cn * nx, r * cy, pz + r * cn * nz))
    tris: list[tuple[int, int, int]] = []
    for i in range(n - 1):
        a, b = sides * i, sides * (i + 1)
        for k in range(sides):
            k2 = (k + 1) % sides
            tris.append((a + k, b + k, b + k2))
            tris.append((a + k, b + k2, a + k2))
    last = sides * (n - 1)
    for k in range(1, sides - 1):
        tris.append((0, k, k + 1))  # start cap
        tris.append((last, last + k + 1, last + k))  # end cap
    return pts, tris


def transform(pts, rot_x_deg: float, offset):
    """`pts` turned about the x axis then shifted by `offset`."""
    a = math.radians(rot_x_deg)
    c, s = math.cos(a), math.sin(a)
    ox, oy, oz = offset
    return [(x + ox, y * c - z * s + oy, y * s + z * c + oz) for x, y, z in pts]


# ----- profiles and paths ----------------------------------------------------
def mug_body_profile() -> list:
    """Base centre, up the outside, over the rim, down the bore, to the floor."""
    return [
        (0.0, 0.0),
        (MUG_OUTER_R, 0.0),
        (MUG_OUTER_R, MUG_HEIGHT),
        (MUG_INNER_R, MUG_HEIGHT),
        (MUG_INNER_R, MUG_FLOOR),
        (0.0, MUG_FLOOR),
    ]


def handle_path(eye_x: float) -> list:
    """Centreline of the handle loop, in the mug's xz plane.

    Out of the wall at the eye's roof, round to the outer face, down, round
    again, and back into the wall at the eye's floor. The two ends sit inside
    the body, so the swept loop closes there rather than meeting the body's
    surface.
    """
    h = HANDLE_SECTION / 2.0
    z_top = EYE_FLOOR_Z + EYE_HEIGHT + h
    z_bot = EYE_FLOOR_Z - h
    x_out = MUG_OUTER_R + eye_x + h
    r = HANDLE_CORNER_R
    path = [(HANDLE_ROOT_R, z_top), (x_out - r, z_top)]
    for k in range(1, CORNER_STEPS + 1):
        a = math.pi / 2 * k / CORNER_STEPS
        path.append((x_out - r + r * math.sin(a), z_top - r + r * math.cos(a)))
    path.append((x_out, z_bot + r))
    for k in range(1, CORNER_STEPS + 1):
        a = math.pi / 2 * k / CORNER_STEPS
        path.append((x_out - r + r * math.cos(a), z_bot + r - r * math.sin(a)))
    path.append((HANDLE_ROOT_R, z_bot))
    return path


def peg_profile() -> list:
    """Root centre, along the rod, round the stop ball, about the peg's own axis.

    One diameter the whole way, then a ball centred on the rod's end. The rod
    runs into the ball where the ball's radius equals the rod's, so there is
    no rim between them -- only the step a hung mug has to climb to pass.
    """
    meet = PEG_LENGTH - math.sqrt(STOP_BALL_R**2 - PEG_R**2)
    prof = [(0.0, 0.0), (PEG_R, 0.0), (PEG_R, meet)]
    a0 = math.asin(PEG_R / STOP_BALL_R)
    for k in range(1, BALL_STEPS):
        a = a0 + (math.pi / 2 - a0) * k / BALL_STEPS
        prof.append((STOP_BALL_R * math.cos(a), PEG_LENGTH + STOP_BALL_R * math.sin(a)))
    prof.append((0.0, PEG_LENGTH + STOP_BALL_R))
    return prof


def coaster_profile() -> list:
    """Base centre, round the rim, down the inner wall and across the floor,
    both sampled every CONTACT_STEP, to the floor's centre."""
    prof = [(0.0, 0.0), (COASTER_OUTER_R, 0.0), (COASTER_OUTER_R, COASTER_HEIGHT)]
    prof += [(COASTER_INNER_R, z) for z in _steps(COASTER_HEIGHT, COASTER_FLOOR, CONTACT_STEP)]
    prof += [(r, COASTER_FLOOR) for r in _steps(COASTER_INNER_R, 0.0, CONTACT_STEP)[1:]]
    return prof


def _finish(name, pts, tris, colour, approx):
    """Check the mesh closes, point its normals out, and report its volume."""
    check_closed(tris)
    tris, vol = orient_outward(pts, tris, name)
    print(f"[{name}] {len(pts)} verts, {len(tris)} tris, volume {vol * 1e6:.2f} cm3", flush=True)
    return (name, pts, tris, colour, approx), vol


def main() -> int:
    out = Path(args_cli.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    eye_x = args_cli.eye_x
    rc = 0

    # --- the mug: body and handle, one rigid body ---------------------------
    prof = mug_body_profile()
    body, vol = _finish("body", *revolve(prof, args_cli.sections), MUG_COLOR, None)
    want = revolved_volume(prof)
    err = abs(vol - want) / want
    if err >= 0.01:
        print(f"[body] volume {vol * 1e6:.2f} vs {want * 1e6:.2f} cm3 MISMATCH", flush=True)
        rc |= 1
    path = handle_path(eye_x)
    handle, _ = _finish("handle", *sweep(path, HANDLE_SECTION), MUG_COLOR, None)
    write_usd_parts(out / "mug.usd", [body, handle], rigid=True, root_name="mug", friction=MUG_FRICTION)

    # --- the rail: board, capping strip, peg --------------------------------
    board, _ = _finish(
        "board",
        *box((-RAIL_WIDTH / 2, -RAIL_THICKNESS, 0.0), (RAIL_WIDTH / 2, 0.0, RAIL_HEIGHT)),
        RAIL_COLOR,
        "none",
    )
    cap, _ = _finish(
        "cap",
        *box(
            (-RAIL_WIDTH / 2 - 0.004, -RAIL_THICKNESS - 0.002, RAIL_HEIGHT),
            (RAIL_WIDTH / 2 + 0.004, RAIL_CAP_OVERHANG, RAIL_HEIGHT + RAIL_CAP_H),
        ),
        RAIL_COLOR,
        "none",
    )
    peg_pts, peg_tris = revolve(peg_profile(), args_cli.sections // 2)
    # The peg's own axis leaves the board along +y, falling by PEG_FALL_DEG.
    peg_pts = transform(peg_pts, -(90.0 + PEG_FALL_DEG), (0.0, 0.0, PEG_ROOT_Z))
    peg, _ = _finish("peg", peg_pts, peg_tris, PEG_COLOR, "none")
    write_usd_parts(
        out / "hook.usd",
        [board, cap, peg],
        rigid=True,
        root_name="hook",
        friction=RAIL_FRICTION,
        friction_combine=RAIL_FRICTION_COMBINE,
    )

    # --- the coaster ---------------------------------------------------------
    prof = coaster_profile()
    coaster, vol = _finish("coaster", *revolve(prof, 2 * args_cli.sections), COASTER_COLOR, "none")
    want = revolved_volume(prof)
    err = abs(vol - want) / want
    if err >= 0.01:
        print(f"[coaster] volume {vol * 1e6:.2f} vs {want * 1e6:.2f} cm3 MISMATCH", flush=True)
        rc |= 1
    write_usd_parts(
        out / "coaster.usd", [coaster], rigid=True, root_name="coaster", friction=COASTER_FRICTION
    )

    if args_cli.keep_obj:
        for name, parts in (("mug", [body, handle]), ("hook", [board, cap, peg]), ("coaster", [coaster])):
            pts: list = []
            tris: list = []
            for _, p, t, _c, _a in parts:
                base = len(pts)
                pts += list(p)
                tris += [(a + base, b + base, c + base) for a, b, c in t]
            write_obj(out / f"{name}.obj", pts, tris)

    print(
        f"[layout] eye {eye_x * 1000:.1f} mm wide; rod {2 * PEG_R * 1000:.1f} mm "
        f"clears by +/-{(eye_x - 2 * PEG_R) / 2 * 1000:.1f} mm, stop ball "
        f"{2 * STOP_BALL_R * 1000:.1f} mm by "
        f"+/-{(eye_x - 2 * STOP_BALL_R) / 2 * 1000:.1f} mm",
        flush=True,
    )
    print("MUG_ON_HOOK_ASSETS_COMPLETE" if rc == 0 else "MUG_ON_HOOK_ASSETS_FAILED", flush=True)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)  # skip Kit's own shutdown, which can hang
