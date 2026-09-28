"""Author a triangle mesh as a USD, with its winding checked first.

`pxr` is imported inside `write_usd`, so this module imports without Kit;
call `write_usd` only after `AppLauncher` has started the app.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

__all__ = [
    "check_closed",
    "orient_outward",
    "signed_volume",
    "steps",
    "write_obj",
    "write_usd",
    "write_usd_parts",
]


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


def steps(a: float, b: float, step: float) -> list[float]:
    """Values from `a` to `b` inclusive, at most `step` apart."""
    n = max(1, math.ceil(abs(b - a) / step))
    return [a + (b - a) * k / n for k in range(n + 1)]


def signed_volume(pts, tris) -> float:
    """Volume enclosed by the mesh, signed by the triangle winding.

    Positive means the normals point outward, which the renderer and PhysX's
    SDF cooker both require.
    """
    v = 0.0
    for a, b, c in tris:
        ax, ay, az = pts[a]
        bx, by, bz = pts[b]
        cx, cy, cz = pts[c]
        v += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return v / 6.0


def orient_outward(pts, tris, label: str):
    """Flip every triangle if the mesh came out inside-out.

    Returns `(tris, volume)` -- the corrected triangle list and the unsigned
    volume, so a caller can compare the latter against the analytic value it
    was aiming for. Callers must use the returned list; this does not mutate.
    """
    vol = signed_volume(pts, tris)
    if vol < 0:
        tris = [(a, c, b) for a, b, c in tris]
        print(f"[mesh] {label}: winding was inside-out (volume {vol * 1e6:+.1f} cm3), flipped", flush=True)
    return tris, abs(vol)


def write_usd(
    path: Path,
    pts,
    tris,
    colour,
    rigid: bool = False,
    root_name: str = "mesh_root",
    friction: tuple[float, float] | None = None,
    approximation: str | None = None,
) -> None:
    """Author one mesh as `/<root_name>/mesh` in a new USD at `path`.

    `rigid`: apply `RigidBodyAPI` to the root. Required on anything a tactile
    pad senses and on anything the scene re-poses; `UsdFileCfg` does not add
    the schema itself.

    `friction`: `(static, dynamic)`, authored as a physics material bound to
    the mesh. `UsdFileCfg` takes no `physics_material`, so this is the only
    way to give a USD object friction.

    `approximation`: the collider approximation token to author, e.g. "none"
    to keep the triangles (a hole or a bore survives only this way). Leave
    unset for sensed objects; `sim.tactile.apply_sdf_collision` sets "sdf"
    at runtime.
    """
    write_usd_parts(
        path,
        [("mesh", pts, tris, colour, approximation)],
        rigid=rigid,
        root_name=root_name,
        friction=friction,
    )


def write_usd_parts(
    path: Path,
    parts,
    rigid: bool = False,
    root_name: str = "mesh_root",
    friction: tuple[float, float] | None = None,
    friction_combine: str | None = None,
) -> None:
    """Author several meshes under one `/<root_name>` in a new USD at `path`.

    `parts` is a sequence of `(name, pts, tris, colour, approximation)`, each
    authored as `/<root_name>/<name>`. They become one rigid body with one
    collider per mesh, which is how a shape that is not a single closed
    surface reaches PhysX: a union of two watertight meshes is not itself
    watertight, and the SDF cooker needs each mesh closed.

    `friction_combine`: how PhysX mixes this material's friction with the
    other body's -- "min", "max", "multiply" or "average". PhysX averages by
    default, so a deliberately slippery surface is useless against a grippy
    object unless this says otherwise. Arguments otherwise as `write_usd`.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())

    mat = None
    if friction is not None:
        mat = UsdShade.Material.Define(stage, f"/{root_name}/physics_material")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(friction[0]))
        api.CreateDynamicFrictionAttr(float(friction[1]))
        api.CreateRestitutionAttr(0.0)
        if friction_combine is not None:
            from pxr import PhysxSchema

            PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr(friction_combine)

    for name, pts, tris, colour, approximation in parts:
        mesh = UsdGeom.Mesh.Define(stage, f"/{root_name}/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        mesh.CreateFaceVertexCountsAttr([3] * len(tris))
        mesh.CreateFaceVertexIndicesAttr([i for tri in tris for i in tri])
        lo = [min(p[k] for p in pts) for k in range(3)]
        hi = [max(p[k] for p in pts) for k in range(3)]
        mesh.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
        # Flat shading, so the render matches the collider.
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])

        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mca = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        if approximation is not None:
            mca.CreateApproximationAttr(getattr(UsdPhysics.Tokens, approximation))
        if mat is not None:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim())
    stage.Save()


def write_obj(path: Path, pts, tris) -> None:
    """A plain OBJ beside the USD, for looking at outside the simulator."""
    with open(path, "w") as f:
        for x, y, z in pts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in tris:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")
