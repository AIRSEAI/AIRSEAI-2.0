"""One-time Z-up overview geometry, independent of robot or physics state."""

import hashlib
import json
import os
import re
import stat

import numpy as np


TABLE_OVERVIEW_EYE = (1.0, -0.8, 0.95)
TABLE_OVERVIEW_TARGET = (0.26, -0.085, 0.25)
TABLE_FRONT_EYE = (1.35, -0.085, 0.65)
TABLE_FRONT_TARGET = (0.26, -0.085, 0.65)


def table_overview_camera_pose(Gf):
    """Return camera-to-world pose using SDK107's XRTest camera convention."""
    return Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*TABLE_OVERVIEW_EYE), Gf.Vec3d(*TABLE_OVERVIEW_TARGET),
        Gf.Vec3d(0.0, 0.0, 1.0),
    ).GetInverse()


def table_front_camera_pose(Gf):
    """Level initial view above the table; the user can look down naturally."""
    return Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*TABLE_FRONT_EYE), Gf.Vec3d(*TABLE_FRONT_TARGET),
        Gf.Vec3d(0.0, 0.0, 1.0),
    ).GetInverse()


def placed_table_front_camera_pose(Gf, world_down_m, world_yaw_deg, world_toward_viewer_m=0.):
    """Equivalent scene placement by camera only: C_new = C_base * inverse(W)."""
    angle = np.deg2rad(world_yaw_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    # Row-vector W = Rz(yaw) * T(toward, 0, -down), about the robot base origin.
    world = Gf.Matrix4d(
        c, s, 0., 0., -s, c, 0., 0., 0., 0., 1., 0., world_toward_viewer_m, 0., -world_down_m, 1.,
    )
    return table_front_camera_pose(Gf) * world.GetInverse()


def load_view_placement(path):
    """Read a bounded regular JSON file; return validated request and stable hash.

    Reject symlinks/reparse points and verify the opened file is the same regular
    file inspected before opening. No file content can execute code or set a pivot.
    """
    before = os.lstat(path)
    if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
        raise ValueError("placement file must be regular and not a symlink")
    if before.st_size > 4096:
        raise ValueError("placement file exceeds 4096 bytes")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or opened.st_size > 4096
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
            raise ValueError("placement file changed while opening")
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("placement file exceeds 4096 bytes")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate placement key")
            result[key] = value
        return result

    try:
        request = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("placement JSON is invalid") from error
    keys = {"schema_version", "request_id", "world_down_m", "world_yaw_deg", "robot_pivot"}
    if not isinstance(request, dict) or set(request) not in (keys, keys | {"world_toward_viewer_m"}):
        raise ValueError("placement keys must match schema_version 1")
    request.setdefault("world_toward_viewer_m", 0.)
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("placement schema_version must be 1")
    request_id = request["request_id"]
    if not isinstance(request_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", request_id) is None:
        raise ValueError("placement request_id is invalid")
    for key, lower, upper in (("world_down_m", -1., 2.), ("world_yaw_deg", -360., 360.),
                              ("world_toward_viewer_m", -.75, .8)):
        value = request[key]
        if type(value) not in (int, float) or not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"placement {key} is invalid")
        request[key] = float(value)
    pivot = request["robot_pivot"]
    if (not isinstance(pivot, list) or len(pivot) != 3
            or any(type(value) not in (int, float) or value != 0 for value in pivot)):
        raise ValueError("placement robot_pivot must be [0,0,0]")
    request["robot_pivot"] = [0., 0., 0.]
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return request, digest


def fixed_horizontal_controller_frame(core, head_pose, camera_pose, Gf):
    """Map physical Y-up hand poses into a fixed horizontal Z-up task frame.

    Calibrate once from physical head yaw; display pitch never tilts hand motion.
    A vertical physical gaze has no heading and must wait for another head pose.
    """
    forward = head_pose.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    if not np.isfinite([*forward]).all() or np.hypot(forward[0], forward[2]) < 1e-6:
        return None
    head_yaw = core.reorient_transform_matrix_up_right(head_pose, y_up=True)
    head_yaw.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 0.0))
    camera_forward = camera_pose.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    horizontal = Gf.Vec3d(camera_forward[0], camera_forward[1], 0.0)
    if not np.isfinite([*horizontal]).all() or np.hypot(horizontal[0], horizontal[1]) < 1e-6:
        return None
    camera_heading = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(0.0, 0.0, 0.0), horizontal, Gf.Vec3d(0.0, 0.0, 1.0),
    ).GetInverse()
    # Gf uses row vectors: physical hand pose * this fixed change of basis.
    return head_yaw.GetInverse() * camera_heading
