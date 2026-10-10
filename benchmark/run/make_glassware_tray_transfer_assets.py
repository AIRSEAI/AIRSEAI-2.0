#!/usr/bin/env python3
"""Generate a wooden tray with cutout handles and transparent glass cups.

python benchmark/run/make_glassware_tray_transfer_assets.py --headless
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from sim.assets.meshio import check_closed, orient_outward, write_usd_parts
from sim.assets.provision import task_asset_dir
from sim.dims.glassware_tray_transfer import (
    CORNER_RADIUS,
    CUP_HEIGHT,
    CUP_RADIUS,
    CUP_WALL,
    HANDLE_HEIGHT,
    HANDLE_HOLE_HEIGHT,
    HANDLE_HOLE_WIDTH,
    HANDLE_HOLE_Z,
    HANDLE_THICKNESS,
    RIM_HEIGHT,
    RIM_THICKNESS,
    TRAY_COLOR,
    TRAY_LENGTH,
    TRAY_THICKNESS,
    TRAY_WIDTH,
)


def rounded_outline(width, height, radius, center=(0.0, 0.0)):
    points = []
    for x, y, start in (
        (width / 2 - radius, height / 2 - radius, 0),
        (-width / 2 + radius, height / 2 - radius, 90),
        (-width / 2 + radius, -height / 2 + radius, 180),
        (width / 2 - radius, -height / 2 + radius, 270),
    ):
        for step in range(9):
            angle = math.radians(start + step * 90 / 8)
            points.append(
                (x + radius * math.cos(angle) + center[0], y + radius * math.sin(angle) + center[1])
            )
    return points


def extrude(name, outline, low, high, inner=None, vertical=False):
    n = len(outline)
    loops = [outline] if inner is None else [outline, inner]
    points = [(x, y, z) for loop in loops for z in (low, high) for x, y in loop]
    faces = []
    for base in range(0, len(points), 2 * n):
        for i in range(n):
            j = (i + 1) % n
            pair = [(base + i, base + j, base + n + j), (base + i, base + n + j, base + n + i)]
            faces.extend(pair if base == 0 else [(a, c, b) for a, b, c in pair])
    if inner is None:
        for i in range(1, n - 1):
            faces.extend([(0, i + 1, i), (n, n + i, n + i + 1)])
    else:
        for i in range(n):
            j = (i + 1) % n
            faces.extend(
                [
                    (i, 2 * n + j, j),
                    (i, 2 * n + i, 2 * n + j),
                    (n + i, n + j, 3 * n + j),
                    (n + i, 3 * n + j, 3 * n + i),
                ]
            )
    if vertical:
        points = [(x, z, y) for x, y, z in points]
    check_closed(faces)
    faces, _ = orient_outward(points, faces, name)
    return name, points, faces, TRAY_COLOR, None


def tray_parts():
    outer = rounded_outline(TRAY_LENGTH, TRAY_WIDTH, CORNER_RADIUS)
    inner = rounded_outline(
        TRAY_LENGTH - 2 * RIM_THICKNESS, TRAY_WIDTH - 2 * RIM_THICKNESS, CORNER_RADIUS - RIM_THICKNESS
    )
    body = [
        extrude("deck", outer, 0.0, TRAY_THICKNESS),
        extrude("rim", outer, TRAY_THICKNESS, TRAY_THICKNESS + RIM_HEIGHT, inner),
    ]
    panel = rounded_outline(
        TRAY_LENGTH - 2 * CORNER_RADIUS, HANDLE_HEIGHT, 0.010, center=(0.0, HANDLE_HEIGHT / 2)
    )
    hole = rounded_outline(HANDLE_HOLE_WIDTH, HANDLE_HOLE_HEIGHT, 0.008, center=(0.0, HANDLE_HOLE_Z))
    points, faces = [], []
    for name, sign in (("right_handle", -1), ("left_handle", 1)):
        y = sign * (TRAY_WIDTH / 2 - HANDLE_THICKNESS / 2)
        _, handle_points, handle_faces, _, _ = extrude(
            name, panel, y - HANDLE_THICKNESS / 2, y + HANDLE_THICKNESS / 2, hole, vertical=True
        )
        offset = len(points)
        points.extend(handle_points)
        faces.extend(tuple(index + offset for index in triangle) for triangle in handle_faces)
    check_closed(faces)
    faces, _ = orient_outward(points, faces, "handles")
    return [("handles", points, faces, TRAY_COLOR, None), *body]


def add_wood_material(path):
    import numpy as np
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    y, x = np.mgrid[0:512, 0:512] / 512
    phase = y * 95 + 1.5 * np.sin(x * 7) + 0.5 * np.sin(x * 19 + y * 4)
    grain = 0.055 * np.sin(phase * 2 * np.pi) + 0.018 * np.sin(phase * 6 * np.pi)
    grain += np.random.default_rng(4).normal(0, 0.008, x.shape)
    colour = np.array([0.62, 0.43, 0.25])[None, None, :] + grain[..., None]
    Image.fromarray(np.uint8(np.clip(colour, 0, 1) * 255)).save(path.parent / "wood.png")
    stage = Usd.Stage.Open(str(path))
    material = UsdShade.Material.Define(stage, "/tray/wood")
    surface = UsdShade.Shader.Define(stage, "/tray/wood/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    surface.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12))
    texture = UsdShade.Shader.Define(stage, "/tray/wood/texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("wood.png"))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    reader = UsdShade.Shader.Define(stage, "/tray/wood/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        vertical = "handle" in prim.GetName()
        uv = [
            Gf.Vec2f(float(p[0]) / TRAY_LENGTH + 0.5, float(p[2] if vertical else p[1]) / TRAY_WIDTH + 0.5)
            for p in points
        ]
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(uv)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    stage.Save()


def make_glass_cup(path):
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/glass_cup")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim())
    collider = UsdGeom.Cube.Define(stage, "/glass_cup/collider")
    collider.CreateSizeAttr(1.0)
    collider.AddTranslateOp().Set(Gf.Vec3d(0, 0, CUP_HEIGHT / 2))
    side = math.sqrt(2) * CUP_RADIUS
    collider.AddScaleOp().Set(Gf.Vec3d(side, side, CUP_HEIGHT))
    collider.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    material = UsdShade.Material.Define(stage, "/glass_cup/material")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.65)
    physics.CreateDynamicFrictionAttr(0.5)
    physics.CreateRestitutionAttr(0.03)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("min")
    binding = UsdShade.MaterialBindingAPI.Apply(collider.GetPrim())
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    shader = UsdShade.Shader.Define(stage, "/glass_cup/material/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.68, 0.84, 0.91))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.06)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    glass_shader = UsdShade.Shader.Define(stage, "/glass_cup/material/glass_mdl")
    glass_shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    glass_shader.SetSourceAsset("OmniGlass.mdl", "mdl")
    glass_shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    glass_shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.82, 0.88))
    glass_shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.47)
    glass_shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(50.0)
    glass_shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.03)
    material.CreateSurfaceOutput("mdl").ConnectToSource(glass_shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(glass_shader.ConnectableAPI(), "out")
    segments = 32
    rings = (
        (CUP_RADIUS, 0.0),
        (CUP_RADIUS, CUP_HEIGHT),
        (CUP_RADIUS - CUP_WALL, CUP_HEIGHT),
        (CUP_RADIUS - CUP_WALL, CUP_WALL),
    )
    points = [
        Gf.Vec3f(
            radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments), z
        )
        for radius, z in rings
        for i in range(segments)
    ]
    points.append(Gf.Vec3f(0, 0, CUP_WALL))
    faces = []
    for ring in range(3):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = ring * segments + i, ring * segments + j
            c, d = (ring + 1) * segments + i, (ring + 1) * segments + j
            faces.extend(((a, b, d), (a, d, c)))
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((3 * segments + i, 3 * segments + j, 4 * segments))
    surface = UsdGeom.Mesh.Define(stage, "/glass_cup/glass")
    surface.CreatePointsAttr(points)
    surface.CreateFaceVertexCountsAttr([3] * len(faces))
    surface.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
    surface.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
    edge_material = UsdShade.Material.Define(stage, "/glass_cup/edge_material")
    edge_shader = UsdShade.Shader.Define(stage, "/glass_cup/edge_material/surface")
    edge_shader.CreateIdAttr("UsdPreviewSurface")
    edge_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.80, 0.94, 0.98))
    edge_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.08)
    edge_material.CreateSurfaceOutput().ConnectToSource(edge_shader.ConnectableAPI(), "surface")
    for name, z in (("lip", CUP_HEIGHT), ("base_edge", CUP_WALL)):
        ring_points = [
            Gf.Vec3f(
                radius * math.cos(2 * math.pi * i / segments),
                radius * math.sin(2 * math.pi * i / segments),
                z,
            )
            for radius in (CUP_RADIUS + 0.0005, CUP_RADIUS - CUP_WALL)
            for i in range(segments)
        ]
        ring_faces = []
        for i in range(segments):
            j = (i + 1) % segments
            ring_faces.extend(((i, j, segments + j), (i, segments + j, segments + i)))
        edge = UsdGeom.Mesh.Define(stage, f"/glass_cup/{name}")
        edge.CreatePointsAttr(ring_points)
        edge.CreateFaceVertexCountsAttr([3] * len(ring_faces))
        edge.CreateFaceVertexIndicesAttr([v for face in ring_faces for v in face])
        edge.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        UsdShade.MaterialBindingAPI.Apply(edge.GetPrim()).Bind(edge_material)
    for name, center in (("left_reflection", 35), ("right_reflection", -105)):
        angles = [math.radians(center - 13 + 26 * i / 6) for i in range(7)]
        side_points = [
            Gf.Vec3f((CUP_RADIUS + 0.0005) * math.cos(angle), (CUP_RADIUS + 0.0005) * math.sin(angle), z)
            for z in (CUP_WALL, CUP_HEIGHT - CUP_WALL)
            for angle in angles
        ]
        side_faces = []
        for i in range(6):
            side_faces.extend(((i, i + 1, i + 8), (i, i + 8, i + 7)))
        reflection = UsdGeom.Mesh.Define(stage, f"/glass_cup/{name}")
        reflection.CreatePointsAttr(side_points)
        reflection.CreateFaceVertexCountsAttr([3] * len(side_faces))
        reflection.CreateFaceVertexIndicesAttr([v for face in side_faces for v in face])
        reflection.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        UsdShade.MaterialBindingAPI.Apply(reflection.GetPrim()).Bind(edge_material)
    stage.Save()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Generate glassware_tray_transfer assets.")
    parser.add_argument("--out", default=str(task_asset_dir("glassware_tray_transfer")))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    launcher = AppLauncher(args)
    try:
        path = Path(args.out).expanduser() / "tray.usd"
        write_usd_parts(path, tray_parts(), rigid=True, root_name="tray", friction=(0.9, 0.7))
        add_wood_material(path)
        make_glass_cup(path.parent / "glass_cup.usd")
        print(f"[glassware_tray_transfer] wrote {path}", flush=True)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
