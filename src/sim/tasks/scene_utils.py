"""Helpers shared by the task experts.

Camera and wrist orientation quaternions, tactile pad retargeting, and the D6
joint that holds a split object's two shells together.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["align_frame_quat", "jaw_tilt_deg", "join_split_shells", "look_at",
           "retarget_pads", "top_down_grasp_quat"]


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a camera at `eye` aimed at `target`.

    OpenGL convention, which is what `TiledCameraCfg.OffsetCfg` wants when its
    `convention` is "opengl": -z forward, +y up.
    """
    eye, target, up = map(np.asarray, (eye, target, up))
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:                 # camera looking straight down
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    w = max(math.sqrt(max(1.0 + R.trace(), 0.0)) / 2.0, 1e-8)
    return (w, (R[2, 1] - R[1, 2]) / (4 * w),
            (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))


def retarget_pads(sensors: dict[str, Any], prim: str,
                  env_pattern: str = "/World/envs/env_.*") -> str:
    """Point every pad in `sensors` at `<env>/<prim>` and rebuild its views.

    TacSL resolves `contact_object_prim_path_expr` once, in
    `VisuoTactileSensor._create_physx_views()`, so a task whose object changes
    per episode must rebuild the views. This calls that private method again,
    so a TacSL release can break it.

    Call it after the new object is written to the simulation and before the
    episode's first step. Returns the prim name.
    """
    for sensor in sensors.values():
        sensor.cfg.contact_object_prim_path_expr = f"{env_pattern}/{prim}"
        sensor._create_physx_views()
    return prim


def _mat_from_quat(q) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_from_mat(m: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from a rotation matrix, via the largest-diagonal branch.

    The naive `w = sqrt(1 + trace)/2` formula loses all precision when the
    trace approaches -1, which is exactly the half-turn region this is used in:
    levelling a badly tilted wrist can want a large rotation.
    """
    t = m.trace()
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def _orthonormal_frame(primary, secondary) -> np.ndarray:
    """Columns [p, s', p x s'] with `s'` the part of `s` perpendicular to `p`."""
    p = np.asarray(primary, dtype=float)
    p = p / np.linalg.norm(p)
    s = np.asarray(secondary, dtype=float)
    s = s - np.dot(s, p) * p
    n = np.linalg.norm(s)
    if n < 1e-9:
        raise ValueError("primary and secondary directions are parallel")
    s = s / n
    return np.stack([p, s, np.cross(p, s)], axis=1)


def top_down_grasp_quat(tcp_quat_w, jaw_axis_w, approach_w,
                        jaw_target=(0.0, 1.0, 0.0),
                        approach_target=(0.0, 0.0, -1.0)) -> np.ndarray:
    """Wrist orientation that levels the jaws and points the blade down.

    Position-only IK leaves the jaw axis out of horizontal, which rolling
    joint7 cannot fix; the pads then touch at different heights and closing
    tips a standing object.

    The measured jaw axis and approach direction are expressed in the TCP
    frame and matched against their targets::

        R_des = B A^T,  A = frame(approach_local, jaw_local),
                        B = frame(approach_target, jaw_target)

    Arguments are world-frame vectors. The defaults ask for a vertical
    approach with the jaws opening along y. Returns (w, x, y, z) for pose IK.
    """
    R = _mat_from_quat(tcp_quat_w)
    jaw_l = R.T @ np.asarray(jaw_axis_w, dtype=float)
    app_l = R.T @ np.asarray(approach_w, dtype=float)
    A = _orthonormal_frame(app_l, jaw_l)
    B = _orthonormal_frame(approach_target, jaw_target)
    return _quat_from_mat(B @ A.T)


def align_frame_quat(tcp_quat_w, axis_a_w, axis_b_w,
                     target_a, target_b) -> np.ndarray:
    """Wrist orientation carrying two measured world axes onto two targets.

    The construction of `top_down_grasp_quat` with both axes chosen by the
    caller, e.g. a held object's long axis and the jaw axis. Do not use
    `gel_midpoint - TCP`: it is zero, since the TCP is the gel midpoint.

    Arguments are world-frame vectors. `axis_a` is matched exactly, `axis_b`
    only in its component perpendicular to `axis_a`. Both are lines, so each
    is flipped toward its target first to avoid a needless half turn.
    Returns (w, x, y, z) for pose IK.
    """
    a = np.asarray(axis_a_w, dtype=float)
    b = np.asarray(axis_b_w, dtype=float)
    ta = np.asarray(target_a, dtype=float)
    tb = np.asarray(target_b, dtype=float)
    a = a * (1.0 if float(a @ ta) >= 0.0 else -1.0)
    b = b * (1.0 if float(b @ tb) >= 0.0 else -1.0)
    R = _mat_from_quat(tcp_quat_w)
    A = _orthonormal_frame(R.T @ a, R.T @ b)
    B = _orthonormal_frame(ta, tb)
    return _quat_from_mat(B @ A.T)


def jaw_tilt_deg(jaw_axis_w) -> float:
    """How far the jaw axis is from horizontal, degrees.

    The number to assert on before closing: a levelled grasp reads a few
    degrees, and the uncorrected pose on this arm reads tens of degrees.
    """
    v = np.asarray(jaw_axis_w, dtype=float)
    v = v / np.linalg.norm(v)
    return float(math.degrees(math.asin(abs(v[2]))))


def join_split_shells(half_a: str, half_b: str, *, stiffness: float,
                      damping: float, break_force: float, travel: float,
                      gap: float, axis: str = "Y",
                      name: str = "split_joint") -> str:
    """Author (or re-author) the D6 joint that holds a split object together.

    A hollow part modelled as two rigid half-shells gets its compliance from
    this joint, since the tactile pads sense rigid SDF colliders only. The
    joint locks five degrees of freedom, drives the sixth with a linear spring
    of `stiffness` (N/m, along the gripper's closing axis), limits it to
    `travel`, and carries `physics:breakForce` (`inf` for a part that only
    deforms). The shells are collision-filtered against each other.

    A D6 joint, because a drive on a PhysX prismatic joint discards its
    stiffness. `breakForce` is ignored on articulation joints, so the shells
    must stay free rigid bodies; `travel` must stay wide enough that the
    spring, not the limit, carries the load. Re-authored every reset: a broken
    joint cannot be repaired in place.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    env = half_a.rsplit("/", 1)[0]
    path = f"{env}/{name}"
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

    free = {"X": "transX", "Y": "transY", "Z": "transZ"}[axis]
    joint = UsdPhysics.Joint.Define(stage, path)          # generic == D6
    prim = joint.GetPrim()
    joint.CreateBody0Rel().SetTargets([Sdf.Path(half_a)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(half_b)])
    # The shells spawn at +gap/2 (a) and -gap/2 (b); offset each joint frame
    # half a gap back so both coincide on the part's axis at rest.
    off = float(gap) / 2.0
    ax = {"X": 0, "Y": 1, "Z": 2}[axis]
    p0 = [0.0, 0.0, 0.0]
    p1 = [0.0, 0.0, 0.0]
    p0[ax], p1[ax] = -off, +off
    joint.CreateLocalPos0Attr(Gf.Vec3f(*p0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*p1))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))

    # A generic joint with no LimitAPI is free in all six degrees of freedom,
    # so every axis has to be named. `low > high` is USD's spelling of
    # "locked"; the free axis gets real limits.
    for token in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, token)
        if token == free:
            # Symmetric, so the part is squeezable whichever sign the
            # closing direction has.
            limit.CreateLowAttr(-float(travel))
            limit.CreateHighAttr(float(travel))
        else:
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)

    joint.CreateBreakForceAttr(float(break_force))
    # Torque is not the failure being modelled; a finite value would let a
    # twisted grasp break the joint.
    joint.CreateBreakTorqueAttr(float("inf"))

    # Force mode, not acceleration mode: an acceleration drive scales the
    # spring by the pair's reduced mass, coupling stiffness to weight when
    # the drawn classes differ in mass. In force mode `stiffness` is N/m.
    drive = UsdPhysics.DriveAPI.Apply(prim, free)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateMaxForceAttr(float("inf"))

    # Filter the pair from both sides; a single-sided filter is not reliable.
    for own, other in ((half_a, half_b), (half_b, half_a)):
        prim = stage.GetPrimAtPath(own)
        if not prim.IsValid():
            continue
        filt = (UsdPhysics.FilteredPairsAPI(prim)
                if prim.HasAPI(UsdPhysics.FilteredPairsAPI)
                else UsdPhysics.FilteredPairsAPI.Apply(prim))
        rel = filt.CreateFilteredPairsRel()
        if Sdf.Path(other) not in (rel.GetTargets() or []):
            rel.AddTarget(Sdf.Path(other))
    return path
