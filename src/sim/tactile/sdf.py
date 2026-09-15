"""Post-spawn collider setup for tactile scenes: SDF, triangle mesh and CCD.

TacSL needs the sensed object to be a mesh whose collision approximation is
`"sdf"`; Isaac Lab's spawner configs cannot set it, so these helpers author
the schema after spawning and before `sim.reset()`, when PhysX cooks it.
"""

from __future__ import annotations

__all__ = ["apply_ccd", "apply_sdf_collision", "apply_triangle_mesh_collision"]


def apply_sdf_collision(prim_path: str, sdf_resolution: int = 256) -> list[str]:
    """Give an object's collision meshes an SDF, which the force field needs.

    Must run after the prims exist but before `sim.reset()`, because PhysX cooks
    the SDF when the collision shape is created. Returns the prim paths that
    ended up with approximation == "sdf" so callers can fail loudly.

    `sdf_resolution` 256 is the PhysX default; higher costs cooking time and
    VRAM.
    """
    import omni.usd
    from isaaclab.sim.schemas import SDFMeshPropertiesCfg, define_mesh_collision_properties
    from pxr import UsdGeom, UsdPhysics

    cfg = SDFMeshPropertiesCfg(
        sdf_resolution=sdf_resolution, sdf_margin=0.01,
        sdf_narrow_band_thickness=0.01, sdf_subgrid_resolution=6,
    )
    stage = omni.usd.get_context().get_stage()

    def walk(prim):
        yield prim
        for child in prim.GetChildren():
            yield from walk(child)

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"no prim at {prim_path}; SDF colliders cannot be "
                         "applied to an object that was never spawned")

    applied = []
    for prim in walk(root):
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        path = prim.GetPath().pathString
        define_mesh_collision_properties(prim_path=path, cfg=cfg)
        if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf":
            applied.append(path)
    return applied


def apply_triangle_mesh_collision(prim_path: str) -> list[str]:
    """Keep an object's collision meshes as triangle meshes ("none").

    For static objects with a hole or bore: a convex approximation fills the
    hole in. A triangle mesh is a legal collider only for a static body.
    Returns the prims that ended up with approximation "none".
    """
    import omni.usd
    from isaaclab.sim.schemas import (
        TriangleMeshPropertiesCfg,
        define_mesh_collision_properties,
    )
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"no prim at {prim_path}")

    def walk(prim):
        yield prim
        for child in prim.GetChildren():
            yield from walk(child)

    applied = []
    for prim in walk(root):
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        path = prim.GetPath().pathString
        define_mesh_collision_properties(prim_path=path,
                                         cfg=TriangleMeshPropertiesCfg())
        if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "none":
            applied.append(path)
    return applied


def apply_ccd(prim_path: str, scene_path: str = "/physicsScene") -> None:
    """Enable continuous collision detection on a rigid body, post-spawn.

    Writes the PhysX schema attributes directly; both the body's
    ``ccdEnabled`` and the scene's ``enableCCD`` are required. Needed for
    falling objects, which can cross a thin contact offset in one step and
    tunnel through a triangle-mesh collider.
    """
    import omni.usd
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"no prim at {prim_path}")
    api = PhysxSchema.PhysxRigidBodyAPI(prim)
    if not api:
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    api.CreateEnableCCDAttr(True)

    scene_prim = stage.GetPrimAtPath(scene_path)
    if scene_prim.IsValid():
        sapi = PhysxSchema.PhysxSceneAPI(scene_prim)
        if not sapi:
            sapi = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        sapi.CreateEnableCCDAttr(True)
