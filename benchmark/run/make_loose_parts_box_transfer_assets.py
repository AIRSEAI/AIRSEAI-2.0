#!/usr/bin/env python3
"""Generate the open box, hex nut and folded shelf USDs."""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

from sim.assets.meshio import write_usd, write_usd_parts
from sim.assets.provision import task_asset_dir
from sim.dims.loose_parts_box_transfer import (
    BOX_COLOR,
    BOX_HEIGHT,
    BOX_LENGTH,
    BOX_WALL,
    BOX_WIDTH,
    NUT_COLOR,
    NUT_FLATS,
    NUT_HEIGHT,
    SHELF_COLOR,
    SHELF_HEIGHT,
    SHELF_LENGTH,
    SHELF_THICKNESS,
    SHELF_WIDTH,
)

parser = argparse.ArgumentParser(description="Generate loose_parts_box_transfer assets.")
parser.add_argument("--out", default=str(task_asset_dir("loose_parts_box_transfer")))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


def cuboid(center, size):
    x, y, z = center
    a, b, c = (v / 2 for v in size)
    points = [(x + sx * a, y + sy * b, z + sz * c) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    faces = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))
    return points, [tri for a, b, c, d in faces for tri in ((a, b, c), (a, c, d))]


def nut_mesh():
    r = NUT_FLATS / math.sqrt(3)
    sections = ((r, 0.0, False), (r, NUT_HEIGHT, False), (0.005, NUT_HEIGHT, True), (0.005, 0.0, True))
    points = []
    sectors = 48
    for radius, z, round_section in sections:
        for i in range(sectors):
            angle = 2 * math.pi * i / sectors
            if round_section:
                x, y = radius * math.cos(angle), radius * math.sin(angle)
            else:
                corner, rem = divmod(i, 8)
                a = corner * math.pi / 3
                b = (corner + 1) * math.pi / 3
                f = rem / 8
                x = radius * ((1 - f) * math.cos(a) + f * math.cos(b))
                y = radius * ((1 - f) * math.sin(a) + f * math.sin(b))
            points.append((x, y, z))
    triangles = []
    for row in range(len(sections)):
        next_row = (row + 1) % len(sections)
        for i in range(sectors):
            j = (i + 1) % sectors
            a, b = row * sectors + i, row * sectors + j
            c, d = next_row * sectors + j, next_row * sectors + i
            triangles.extend(((a, b, c), (a, c, d)))
    return points, triangles


def folded_shelf(out):
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdShade

    t, radius = SHELF_THICKNESS, 0.005
    x, z = (SHELF_LENGTH - t) / 2, SHELF_HEIGHT - t / 2
    path = [(-x, 0.003), (-x, z - radius)]
    path += [
        (-x + radius + radius * math.cos(q), z - radius + radius * math.sin(q))
        for q in np.linspace(math.pi, math.pi / 2, 24)[1:]
    ]
    path += [(x - radius, z)]
    path += [
        (x - radius + radius * math.cos(q), z - radius + radius * math.sin(q))
        for q in np.linspace(math.pi / 2, 0, 24)[1:]
    ]
    path += [(x, 0.003)]
    path = np.asarray(path)
    tangents = np.gradient(path, axis=0)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    sections = []
    for point, normal in zip(path, normals, strict=True):
        sections.append(
            [
                (float((point + s * normal)[0]), y, float((point + s * normal)[1]))
                for y in (-SHELF_WIDTH / 2, SHELF_WIDTH / 2)
                for s in (-t / 2, t / 2)
            ]
        )
    faces = ((0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3), (0, 1, 3, 2), (4, 6, 7, 5))
    triangles = [tri for a, b, c, d in faces for tri in ((a, b, c), (a, c, d))]
    parts = [
        (f"fold_{i}", a + b, triangles, SHELF_COLOR, "convexHull")
        for i, (a, b) in enumerate(zip(sections[:-1], sections[1:], strict=True))
    ]
    for side in (-1, 1):
        points, tris = cuboid((side * x, 0, 0.002), (0.015, 0.155, 0.004))
        parts.append((f"foot_{side + 1}", points, tris, (0.025, 0.028, 0.03), "convexHull"))
    target = out / "folded_shelf.usd"
    write_usd_parts(target, parts, root_name="folded_shelf", friction=(0.6, 0.5))
    stage = Usd.Stage.Open(str(target))
    for name, _, _, color, _ in parts:
        material = UsdShade.Material.Define(stage, f"/folded_shelf/materials/{name}")
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.15 if name.startswith("fold") else 0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            0.35 if name.startswith("fold") else 0.8
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"/folded_shelf/{name}")).Bind(material)
    stage.Save()


def main():
    from pathlib import Path

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    length, width, height, wall = BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT, BOX_WALL
    shell = [
        ("floor", (0, 0, wall / 2), (length, width, wall)),
        ("near_wall", (0, -width / 2 + wall / 2, height / 2), (length, wall, height)),
        ("far_wall", (0, width / 2 - wall / 2, height / 2), (length, wall, height)),
        ("left_wall", (-length / 2 + wall / 2, 0, height / 2), (wall, width, height)),
        ("right_wall", (length / 2 - wall / 2, 0, height / 2), (wall, width, height)),
    ]
    parts = []
    for name, pos, size in shell:
        points, triangles = cuboid(pos, size)
        parts.append((name, points, triangles, BOX_COLOR, "convexHull"))
    write_usd_parts(out / "parts_box.usd", parts, rigid=True, root_name="parts_box", friction=(3.0, 2.5))
    points, triangles = nut_mesh()
    write_usd(
        out / "hex_nut.usd",
        points,
        triangles,
        NUT_COLOR,
        rigid=True,
        root_name="hex_nut",
        friction=(0.6, 0.5),
        approximation="convexDecomposition",
    )
    folded_shelf(out)
    print("OBJECTS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
