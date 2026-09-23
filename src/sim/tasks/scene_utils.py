"""Helpers shared by the task experts.

Camera and wrist orientation quaternions, tactile pad retargeting, the D6
joint that holds a split object's two shells together, and collision between
the two arms.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["align_frame_quat", "enable_cross_arm_collision", "jaw_tilt_deg",
           "join_spin", "join_split_shells", "look_at", "oblique_grasp_quat",
           "retarget_pads", "set_joint_enabled", "top_down_grasp_quat"]


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


def oblique_grasp_quat(tcp_quat_w, jaw_axis_w, approach_w, pitch_deg: float,
                       yaw_deg: float | None = None) -> np.ndarray:
    """Wrist orientation for a grasp from the front and above, the way a hand
    reaches: jaws level, blade pitched `pitch_deg` below horizontal.

    `yaw_deg` is the blade's heading in the world xy plane; None keeps the
    heading of the measured approach, so the hand keeps pointing the way the
    arm reached. 90 degrees of pitch is the straight-down grasp of
    `top_down_grasp_quat` without its fixed jaw direction.

    Arguments are world-frame vectors. Returns (w, x, y, z) for pose IK.
    """
    a = np.asarray(approach_w, dtype=float)
    a = a / np.linalg.norm(a)
    if yaw_deg is None:
        h = a[:2]
        if np.linalg.norm(h) < 1e-6:           # pointing straight down: use the jaw
            j = np.asarray(jaw_axis_w, dtype=float)
            h = np.array([j[1], -j[0]])
        psi = math.atan2(h[1], h[0])
    else:
        psi = math.radians(yaw_deg)
    th = math.radians(pitch_deg)
    blade = (math.cos(th) * math.cos(psi), math.cos(th) * math.sin(psi), -math.sin(th))
    jaw = (-math.sin(psi), math.cos(psi), 0.0)
    return align_frame_quat(tcp_quat_w, approach_w, jaw_axis_w, blade, jaw)


def jaw_tilt_deg(jaw_axis_w) -> float:
    """How far the jaw axis is from horizontal, degrees.

    The number to assert on before closing; a levelled grasp reads a few
    degrees.
    """
    v = np.asarray(jaw_axis_w, dtype=float)
    v = v / np.linalg.norm(v)
    return float(math.degrees(math.asin(abs(v[2]))))


def join_split_shells(half_a: str, half_b: str, *, stiffness: float,
                      damping: float, break_force: float, travel: float,
                      gap: float, axis: str = "Y",
                      name: str = "split_joint", rest: float = 0.0,
                      limits: tuple[float, float] | None = None) -> str:
    """Author (or re-author) the D6 joint that holds a split object together.

    A hollow part modelled as two rigid half-shells gets its compliance from
    this joint, since the tactile pads sense rigid SDF colliders only. The
    joint locks five degrees of freedom, drives the sixth with a linear spring
    of `stiffness` (N/m, along the gripper's closing axis), limits it to
    `travel`, and carries `physics:breakForce` (`inf` for a part that only
    deforms). The shells are collision-filtered against each other.

    `rest` is where the spring is relaxed, as a change in separation from the
    spawn `gap`: zero for a part at rest as spawned, positive for one that
    spawns compressed and opens when let go. `limits` replaces the symmetric
    `travel` with an explicit (low, high) change in separation.

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
            # Symmetric by default, so the part is squeezable whichever sign
            # the closing direction has.
            low, high = limits if limits is not None else (-float(travel), float(travel))
            limit.CreateLowAttr(float(low))
            limit.CreateHighAttr(float(high))
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
    drive.CreateTargetPositionAttr(float(rest))
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


def join_spin(body0: str, body1: str, *, local_pos0, local_pos1=(0.0, 0.0, 0.0),
              damping: float, limits_deg: tuple[float, float] | None = None,
              name: str = "spin_joint") -> str:
    """Author (or re-author) a D6 joint that leaves `body1` only a turn about
    `body0`'s local z, and return its path.

    The joint frames sit at `local_pos0` in `body0` and `local_pos1` in
    `body1`, both unrotated, so the two bodies keep the relative pose they
    are spawned in. The free turn is resisted by a damper of `damping`
    (N m s/rad) and nothing else; `limits_deg` bounds it, None leaves it
    free. `body0` empty joins `body1` to the world. Author it before
    `sim.reset()`: a joint that first appears at a reset kicks its bodies.

    A generic joint with no LimitAPI is free in all six degrees of freedom,
    so every axis is named; `low > high` is USD's spelling of "locked".
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    env = body1.rsplit("/", 1)[0]
    path = f"{env}/{name}"
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    joint = UsdPhysics.Joint.Define(stage, path)          # generic == D6
    prim = joint.GetPrim()
    if body0:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in local_pos0)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in local_pos1)))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    # A jointed pair does not collide by default, and that filter outlives
    # the joint: a body freed with `set_joint_enabled` would fall through the
    # one it was joined to.
    joint.CreateCollisionEnabledAttr(True)
    # Locked axes first, then the one that turns.
    for token in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        if token == "rotZ" and limits_deg is None:
            continue
        limit = UsdPhysics.LimitAPI.Apply(prim, token)
        lo, hi = limits_deg if token == "rotZ" else (1.0, -1.0)
        limit.CreateLowAttr(float(lo))
        limit.CreateHighAttr(float(hi))
    # A pure damper: a velocity drive to rest, force mode so `damping` is in
    # N m s/rad whatever the bodies weigh.
    drive = UsdPhysics.DriveAPI.Apply(prim, "rotZ")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(damping))
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(float("inf"))
    return path


def set_joint_enabled(path: str, enabled: bool) -> None:
    """Switch an authored joint on or off while the simulation runs."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    UsdPhysics.Joint(stage.GetPrimAtPath(path)).GetJointEnabledAttr().Set(bool(enabled))


def enable_cross_arm_collision(robot_path: str, sides=("left", "right")) -> int:
    """Let one arm's links collide with the other's, and nothing else within
    the robot. Returns the number of link pairs left colliding.

    Both arms are one articulation, whose self-collision is all or nothing:
    turned on as is, a gripper's own gels, which overlap with the jaws shut,
    push each other apart. So every pair of links on the same arm, and every
    pair involving a link on neither arm, is filtered out. Must run before
    `sim.reset()`.
    """
    import omni.usd
    from pxr import PhysxSchema, Sdf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_path)
    if not root.IsValid():
        raise ValueError(f"no robot prim at {robot_path}")

    def walk(prim):
        yield prim
        for child in prim.GetChildren():
            yield from walk(child)

    links = [p for p in walk(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    art = [p for p in walk(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not art:
        raise ValueError(f"no articulation root under {robot_path}")

    def side_of(prim) -> str | None:
        # By the top-level prim under the robot: the tactile fingers are
        # grafted bodies (`openarm_<side>_..._tactile/gelsight_finger`) whose
        # own names carry no side.
        top = prim.GetPath().MakeRelativePath(root.GetPath()).pathString.split("/")[0]
        return next((s for s in sides if top.startswith(f"openarm_{s}_")), None)

    kept = 0
    for i, a in enumerate(links):
        filt = (UsdPhysics.FilteredPairsAPI(a) if a.HasAPI(UsdPhysics.FilteredPairsAPI)
                else UsdPhysics.FilteredPairsAPI.Apply(a))
        rel = filt.CreateFilteredPairsRel()
        targets = set(rel.GetTargets() or [])
        for j, b in enumerate(links):
            if i == j:
                continue
            sa, sb = side_of(a), side_of(b)
            if sa is None or sb is None or sa == sb:
                if b.GetPath() not in targets:
                    rel.AddTarget(Sdf.Path(b.GetPath()))
            elif i < j:
                kept += 1
    for prim in art:
        api = (PhysxSchema.PhysxArticulationAPI(prim)
               if prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
               else PhysxSchema.PhysxArticulationAPI.Apply(prim))
        api.CreateEnabledSelfCollisionsAttr(True)
    return kept
