#!/usr/bin/env python3
"""Build the simulation robot USD: tactile fingers, wrist and head cameras.

The stock finger geometry of the OpenArm USD is deactivated and a complete
GelSight R1.5 finger module (housing, gel, camera, with its vendor prim names
intact) is welded to each finger link; the finger links and prismatic joints
stay. The stock USD is not modified: the result is written to a separate file.

    python -m sim.assets.build_robot_asset --robot <stock.usd> --gel <gelsight.usd> --out <out.usd>
"""

from __future__ import annotations

import argparse

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# (finger links, tcp link) for both grippers; unused sensors go unconfigured.
DEFAULT_GRIPPERS = (
    (("openarm_right_left_finger", "openarm_right_right_finger"),
     "openarm_right_ee_tcp"),
    (("openarm_left_left_finger", "openarm_left_right_finger"),
     "openarm_left_ee_tcp"),
)

# Vendor geometry, measured off the asset: the gel centre sits this far from
# the housing origin along the housing's +z.
GEL_FROM_HOUSING = 0.06776

# Housing inner face offset from the jaw origin along the closing axis, m
# (about where the stock jaw face sits).
FACE_STANDOFF = 0.0055

# Wrist camera (RealSense D405), in world axes at the USD rest pose relative
# to the hand origin; rebased into each hand's frame so it holds for both arms.
D405_SIZE = (0.042, 0.042, 0.023)            # w, h, depth (depth = view axis)
D405_OFFSET = np.array([0.055, 0.0, 0.058])  # hand origin -> body centre
D405_TARGET = np.array([0.0, 0.0, -0.093])   # hand origin -> look-at point

# Head camera (the real robot's `base` stream): on the front face of the torso
# column just under its top, looking down at the tabletop workspace.
HEAD_OFFSET = np.array([0.105, 0.0, 0.745])  # body_link origin -> body centre
# Look-at point straight ahead (y = 0): the camera mounts square to the torso.
HEAD_TARGET = np.array([0.30, 0.0, 0.12])


def _world(cache: UsdGeom.XformCache, prim: Usd.Prim):
    m = np.array(cache.GetLocalToWorldTransform(prim), dtype=float)
    return m[:3, :3].T.copy(), m[3, :3].copy()


def _housing_inner_face(gel_stage: Usd.Stage) -> float:
    """How far the housing protrudes past its origin toward the sensed side.

    The housing origin is not on its inner face, so the protrusion is
    measured from the mesh points.
    """
    root = gel_stage.GetDefaultPrim().GetPath().pathString
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    lo = 1e9
    for prim in Usd.PrimRange(gel_stage.GetPrimAtPath(f"{root}/gelsight_finger")):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        R_m, t_m = _world(cache, prim)
        pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        lo = min(lo, float((pts @ R_m.T + t_m)[:, 1].min()))
    # The gel window sits on the housing's local -y face, so the protrusion
    # toward the sensed side is |min y|.
    return -lo


def _gf_matrix(R: np.ndarray, t: np.ndarray) -> Gf.Matrix4d:
    m = np.eye(4)
    m[:3, :3] = R.T                      # Gf is row-vector convention
    m[3, :3] = t
    return Gf.Matrix4d(*[float(v) for v in m.flatten()])


def _canonical(prim, t, R=None, scale=(1.0, 1.0, 1.0)):
    """Author the canonical [translate, orient, scale] op triple.

    Isaac Lab's XformPrimView (what TiledCamera wraps prims with) rejects
    matrix/partial op orders: every prim on a camera path must carry exactly
    this triple; a matrix TransformOp hangs the renderer without an error.
    """
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in t]))
    q = (_gf_matrix(R, np.zeros(3)).ExtractRotationQuat()
         if R is not None else Gf.Quatd(1.0, Gf.Vec3d(0, 0, 0)))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(q)
    x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))


def _weld_camera(stage: Usd.Stage, link: Usd.Prim, name: str,
                 cache: UsdGeom.XformCache, offset: np.ndarray,
                 target: np.ndarray, size, focal_mm: float) -> None:
    """Camera body + lens prim as children of a link, world-pose specified.

    Children of a rigid-body link ride it without changing the articulation
    topology. `offset`/`target` are relative to the link origin, in world
    axes at the USD rest pose. The camera prim sits 1 mm in front of the
    body's lens face; placed inside the body it renders black.
    """
    R_h, t_h = _world(cache, link)
    cam_w = t_h + offset
    tgt_w = t_h + target
    fwd = tgt_w - cam_w
    fwd = fwd / np.linalg.norm(fwd)
    z = -fwd                                     # USD cameras look along -Z
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R_cw = np.stack([x, y, z], axis=1)

    # World pose -> link-local (rigid: R_h orthonormal).
    R_local = R_h.T @ R_cw
    t_local = R_h.T @ (cam_w - t_h)

    mount = stage.DefinePrim(f"{link.GetPath()}/{name}", "Xform")
    _canonical(mount, t_local, R_local)

    body = UsdGeom.Cube.Define(stage, f"{mount.GetPath()}/body")
    body.GetSizeAttr().Set(1.0)
    _canonical(body.GetPrim(), (0.0, 0.0, 0.0), scale=size)
    body.GetDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.15, 0.18)])

    cam = UsdGeom.Camera.Define(stage, f"{mount.GetPath()}/camera")
    # In front of the -Z face (depth/2) plus 1 mm clearance.
    _canonical(cam.GetPrim(), (0.0, 0.0, -(size[2] / 2 + 0.001)))
    cam.GetFocalLengthAttr().Set(focal_mm)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 20.0))

    print(f"  {name}[{link.GetName()}]: body at world ({cam_w[0]:+.4f} "
          f"{cam_w[1]:+.4f} {cam_w[2]:+.4f}), looking at ({tgt_w[0]:+.4f} "
          f"{tgt_w[1]:+.4f} {tgt_w[2]:+.4f})")


def _weld_wrist_camera(stage: Usd.Stage, root: str, side: str,
                       cache: UsdGeom.XformCache) -> None:
    """RealSense D405 on the hand link, at D405_OFFSET."""
    hand = stage.GetPrimAtPath(f"{root}/openarm_{side}_hand")
    if not hand.IsValid():
        raise SystemExit(f"hand link not found: {root}/openarm_{side}_hand")
    _weld_camera(stage, hand, "wrist_camera", cache, D405_OFFSET, D405_TARGET,
                 D405_SIZE, focal_mm=12.0)   # ~D405 87deg HFOV on 18.7mm


def _weld_head_camera(stage: Usd.Stage, root: str,
                      cache: UsdGeom.XformCache) -> None:
    """The `base` stream's camera, on the torso column's head."""
    body = stage.GetPrimAtPath(f"{root}/openarm_body_link")
    if not body.IsValid():
        raise SystemExit(f"torso link not found: {root}/openarm_body_link")
    _weld_camera(stage, body, "head_camera", cache, HEAD_OFFSET, HEAD_TARGET,
                 D405_SIZE, focal_mm=14.0)


def graft(robot_usd: str, gel_usd: str, out_usd: str,
          grippers=DEFAULT_GRIPPERS,
          standoff: float = FACE_STANDOFF, gel_z_offset: float = 0.0,
          keep_stock_fingers: bool = False, robot_root: str = "") -> str:
    """Write a robot USD whose fingers are GelSight tactile fingers.

    `out_usd` must sit beside `robot_usd` (relative payload references).
    `gel_z_offset` shifts the gel centre along world z relative to the tcp
    height, for tuning off the preview renders.
    """
    stage = Usd.Stage.Open(robot_usd)
    root = robot_root or stage.GetDefaultPrim().GetPath().pathString
    gel_stage = Usd.Stage.Open(gel_usd)
    gel_root = gel_stage.GetDefaultPrim().GetPath().pathString
    inner_face = _housing_inner_face(gel_stage)
    print(f"  housing protrudes {inner_face*1e3:.1f} mm past its origin on the gel side")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    for jaws, tcp in grippers:
      tcp_prim = stage.GetPrimAtPath(f"{root}/{tcp}")
      if not tcp_prim.IsValid():
          raise SystemExit(f"tcp prim not found: {root}/{tcp}")
      _, tcp_pos = _world(cache, tcp_prim)

      jaw_world = {}
      for jaw in jaws:
          prim = stage.GetPrimAtPath(f"{root}/{jaw}")
          if not prim.IsValid():
              raise SystemExit(f"jaw prim not found: {root}/{jaw}")
          jaw_world[jaw] = _world(cache, prim)

      for i, jaw in enumerate(jaws):
          R_jaw, p_jaw = jaw_world[jaw]
          _, p_partner = jaw_world[jaws[1 - i]]
          u = p_partner - p_jaw
          u = u / np.linalg.norm(u)        # closing direction, toward partner

          # Retire the stock finger's geometry; its link and prismatic joint stay
          # so the gripper still opens and closes.
          if not keep_stock_fingers:
              for sub in ("visuals", "collisions"):
                  p = stage.GetPrimAtPath(f"{root}/{jaw}/{sub}")
                  if p.IsValid():
                      p.SetActive(False)

          # The whole vendor finger as one assembly. Its internal joints come
          # with it and keep gel and camera in the vendor arrangement.
          pad_name = f"{jaw}_tactile"
          dst = f"{root}/{pad_name}"
          prim = stage.DefinePrim(dst, "Xform")
          prim.GetReferences().AddReference(assetPath=gel_usd, primPath=gel_root)

          # The vendor welds its finger to the world; ours is welded to the jaw.
          wj = stage.GetPrimAtPath(f"{dst}/root_joint")
          if wj.IsValid():
              wj.SetActive(False)
          # The housing keeps its collider: with the stock geometry retired it
          # is the gripper's contact with the world. The two housings overlap
          # at full close; the scene disables self-collisions on the articulation.

          # Housing orientation: long axis (+z) pointing DOWN so the module
          # hangs like a finger, gel face (local -y, see _housing_inner_face)
          # facing the partner jaw. RotX(180): z->-z, y->-y sends local -y to
          # +y; RotY(180): z->-z keeps y, so local -y stays -y.
          if u[1] > 0:      # partner at +y: local -y must become +y
              R_wh = np.diag([1.0, -1.0, -1.0])       # RotX(180)
          else:             # partner at -y: local -y stays -y
              R_wh = np.diag([-1.0, 1.0, -1.0])       # RotY(180)

          # Mount flush: the housing's inner face (which carries the gel window)
          # lands `standoff` from the jaw origin toward the partner; the housing
          # origin is not on that face. Both R choices map local +y onto the
          # `toward partner` direction, so the shift is simply `inner_face` back
          # along u. Height comes from the tcp.
          face_pt = p_jaw + u * standoff
          t_wh = face_pt - u * inner_face
          gel_z = tcp_pos[2] + gel_z_offset
          t_wh[2] = gel_z + GEL_FROM_HOUSING          # R maps +z to -z
          g_world = t_wh + R_wh @ np.array([0.0, 0.0, GEL_FROM_HOUSING])

          # Authored xform and joint must agree; Isaac Lab may use either one.
          xf = UsdGeom.Xformable(prim)
          xf.ClearXformOpOrder()
          xf.AddTransformOp().Set(_gf_matrix(R_wh, t_wh))

          X_jaw = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{root}/{jaw}"))
          X0 = _gf_matrix(R_wh, t_wh) * X_jaw.GetInverse()
          q0, t0 = X0.ExtractRotationQuat(), X0.ExtractTranslation()
          j = UsdPhysics.FixedJoint.Define(stage, f"{root}/joints/{pad_name}_mount")
          j.GetBody0Rel().SetTargets([Sdf.Path(f"{root}/{jaw}")])
          j.GetBody1Rel().SetTargets([Sdf.Path(f"{dst}/gelsight_finger")])
          j.GetLocalPos0Attr().Set(Gf.Vec3f(t0))
          j.GetLocalRot0Attr().Set(Gf.Quatf(float(q0.GetReal()),
                                            Gf.Vec3f(q0.GetImaginary())))
          j.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
          j.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

          print(f"  {pad_name}: housing at ({t_wh[0]:+.4f} {t_wh[1]:+.4f} {t_wh[2]:+.4f}), "
                f"gel centre ({g_world[0]:+.4f} {g_world[1]:+.4f} {g_world[2]:+.4f}), "
                f"dome -> {jaws[1 - i]}")

    for side in ("right", "left"):
        _weld_wrist_camera(stage, root, side, cache)
    _weld_head_camera(stage, root, cache)

    stage.GetRootLayer().Export(out_usd)
    n_fingers = sum(len(jaws) for jaws, _ in grippers)
    print(f"replaced {n_fingers} fingers with GelSight modules -> {out_usd}")
    return out_usd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Replace OpenArm fingers with GelSight tactile fingers.")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--gel", required=True)
    ap.add_argument("--out", required=True, help="must sit beside --robot")
    ap.add_argument("--standoff", type=float, default=FACE_STANDOFF)
    ap.add_argument("--gel_z_offset", type=float, default=0.0)
    ap.add_argument("--keep_stock_fingers", action="store_true")
    a = ap.parse_args()
    graft(a.robot, a.gel, a.out, DEFAULT_GRIPPERS, a.standoff,
          a.gel_z_offset, a.keep_stock_fingers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
