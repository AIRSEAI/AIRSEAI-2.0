# SPDX-License-Identifier: Apache-2.0
"""YAML -> USD camera/contact transforms and native RTX LUT lenses.

No image postprocessing, hardware IO, global environment variables, or changes
of the original Harness. Call apply_environment before creating image sensors.
"""

import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

CAMERA_PATHS = {
    "base": "openarm_body_link/head_camera/camera",
    "left_wrist": "openarm_left_hand/wrist_camera/camera",
    "right_wrist": "openarm_right_hand/wrist_camera/camera",
}
# Explicit registration for AIRSEAI's supplied OpenArm asset layout only.
LINK_ALIASES = {"openarm_body_link0": "openarm_body_link"}


def intrinsics(item):
    src = item["intrinsics"]
    out = {}
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        value = src[key + "_px"]
        if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"Invalid camera {key}")
        if key in ("width", "height"):
            if int(value) != value or value < 4:
                raise ValueError(f"Invalid image size {key}")
            value = int(value)
        else:
            value = float(value)
        out[key] = value
    if out["fx"] <= 0 or out["fy"] <= 0:
        raise ValueError("Focal length must be positive")
    model = out["model"] = src["distortion_model"]
    counts = {
        "none": (0, 5),
        "realsense_inverse_brown_conrady": (5,),
    }
    if out["model"] not in counts:
        raise ValueError(f"Unsupported distortion model: {model}")
    d = np.asarray(src["distortion_coeffs"], dtype=float)
    if d.ndim != 1 or d.size not in counts[out["model"]] or not np.isfinite(d).all():
        raise ValueError("Invalid distortion coefficients")
    if out["model"] == "none" and np.any(d):
        raise ValueError("Nonzero coefficients with model=none")
    out["coeffs"] = d.tolist()
    out["image_domain"] = src.get("image_domain", "raw")
    if out["image_domain"] not in ("raw", "ideal"):
        raise ValueError("image_domain must be raw or ideal")
    return out


def matrix(k):
    return np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]])


def pixel_rays(pixels, k):
    """RealSense SDK 2.56.5 pixel-to-ray convention; return optical x/z, y/z."""
    points = np.asarray(pixels, dtype=np.float64)
    if points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite (...,2) pixel coordinates")
    p = (points - [k["cx"], k["cy"]]) / [k["fx"], k["fy"]]
    if k["image_domain"] == "ideal" or k["model"] == "none" or not np.any(k["coeffs"]):
        return p
    if k["model"] == "realsense_inverse_brown_conrady":
        # SDK uses float32 and 10 iterations, including tangential terms at
        # radial-scaled xq/yq. Float64 here avoids map accumulation roundoff.
        x0, y0 = p[..., 0], p[..., 1]
        x, y = x0.copy(), y0.copy()
        k1, k2, p1, p2, k3 = k["coeffs"]
        for _ in range(10):
            r2 = x * x + y * y
            radial = 1 + ((k3 * r2 + k2) * r2 + k1) * r2
            if np.any(radial <= 0) or not np.isfinite(radial).all():
                raise ValueError("Invalid inverse Brown radial mapping")
            xq, yq = x * radial, y * radial
            dx = 2 * p1 * xq * yq + p2 * (r2 + 2 * xq * xq)
            dy = 2 * p2 * xq * yq + p1 * (r2 + 2 * yq * yq)
            x, y = (x0 - dx) / radial, (y0 - dy) / radial
        rays = np.stack([x, y], axis=-1)
        projected = project_realsense(rays, k)
        if not np.isfinite(rays).all() or np.max(np.linalg.norm(projected - points, axis=-1)) > 0.01:
            raise ValueError("Inverse Brown failed pixel round-trip tolerance 0.01 px")
        return rays
    raise ValueError("Unsupported lens model")


def project_realsense(rays, k):
    """Inverse of the SDK deprojection, also used for the exit LUT."""
    x, y = np.asarray(rays)[..., 0], np.asarray(rays)[..., 1]
    k1, k2, p1, p2, k3 = k["coeffs"]
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xr, yr = x * f, y * f
    xd = xr + 2 * p1 * xr * yr + p2 * (r2 + 2 * xr * xr)
    yd = yr + 2 * p2 * xr * yr + p1 * (r2 + 2 * yr * yr)
    return np.stack([xd * k["fx"] + k["cx"], yd * k["fy"] + k["cy"]], axis=-1)


def transform(value):
    t = np.asarray(value, dtype=float)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Expected a rigid 4x4 T_parent_from_child (translation in metres)")
    return t


def _fields(value, allowed, where, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"Invalid fields at {where}; allowed={allowed}, required={required}")


def load_environment(path):
    """Read one explicit YAML file; reject unknown/duplicate fields."""
    import yaml

    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()

    # Reject duplicate keys: a typo must not silently replace a calibration.
    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node)
        return result

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    data = yaml.load(raw, Loader=Loader)
    _fields(data, ("cameras", "fingertips"), "root")
    cameras, fingertips = data.get("cameras", {}), data.get("fingertips", {})
    _fields(cameras, CAMERA_PATHS, "cameras")
    if not isinstance(fingertips, dict) or not (cameras or fingertips):
        raise ValueError("Expected at least one camera or fingertip")
    for name, item in cameras.items():
        _fields(item, ("intrinsics", "extrinsics"), name, ("intrinsics", "extrinsics"))
        fields = (
            "width_px",
            "height_px",
            "fx_px",
            "fy_px",
            "cx_px",
            "cy_px",
            "distortion_model",
            "distortion_coeffs",
        )
        _fields(item["intrinsics"], (*fields, "image_domain"), name + ".intrinsics", fields)
        item["intrinsics"].setdefault("image_domain", "raw")
        intrinsics(item)
    for name, frame in [*((name, item["extrinsics"]) for name, item in cameras.items()), *fingertips.items()]:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"Invalid frame identifier: {name!r}")
        _fields(frame, ("parent_frame", "T_parent_from_child"), name, ("parent_frame", "T_parent_from_child"))
        if not isinstance(frame["parent_frame"], str) or not frame["parent_frame"].isidentifier():
            raise ValueError(f"Invalid parent link for {name}")
        transform(frame["T_parent_from_child"])
    return dict(
        cameras=cameras, fingertips=fingertips, source=str(path), sha256=hashlib.sha256(raw).hexdigest()
    )


def _link(stage, root, name):
    from pxr import UsdGeom

    exact = root + "/" + name
    path = exact if stage.GetPrimAtPath(exact) else root + "/" + LINK_ALIASES.get(name, name)
    if not stage.GetPrimAtPath(path):
        raise ValueError(f"Missing parent link: {exact}")
    transform(np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T)
    return path


def _set_transform(prim, t):
    from pxr import Gf, UsdGeom

    # Preserve the canonical TRS order required by Isaac's camera views.
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    m = Gf.Matrix4d(*t.T.reshape(-1).tolist())  # USD uses row vectors.
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(m.ExtractTranslation())
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(m.ExtractRotationQuat())
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1))


def apply_stage(stage, roots, spec):
    """Runtime-stage overrides only; source assets are never saved or modified."""
    from pxr import UsdGeom

    if not spec:
        return []
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("Environment calibration requires a metre-based stage")
    changes = []
    for root in roots:
        for name, item in spec["cameras"].items():
            path = root + "/" + CAMERA_PATHS[name]
            camera = stage.GetPrimAtPath(path)
            if not camera or not camera.IsA(UsdGeom.Camera):
                raise ValueError(f"Missing camera: {path}")
            for attr in ("omni:lensdistortion:model", "cameraProjectionType"):
                value = camera.GetAttribute(attr).Get()
                if value and value not in ("none", "pinhole"):
                    raise ValueError(f"Disable native lens distortion first: {path}: {attr}={value}")
            if UsdGeom.Camera(camera).GetProjectionAttr().Get() != "perspective":
                raise ValueError(f"Expected a pinhole perspective camera: {path}")
            frame = item["extrinsics"]
            parent = _link(stage, root, frame["parent_frame"])
            mount = camera.GetParent()
            if str(mount.GetParent().GetPath()) != parent:
                raise ValueError(f"{name} must be mounted under {parent} to follow the declared parent")
            if mount.IsInstanceProxy() or mount.IsInstance():
                raise ValueError(f"Camera mount must be editable: {mount.GetPath()}")
            optical = transform(frame["T_parent_from_child"])
            camera_local = np.array(UsdGeom.Xformable(camera).GetLocalTransformation()).T
            transform(camera_local)  # No scaled optical frames.
            if UsdGeom.Xformable(camera).GetResetXformStack():
                raise ValueError(f"Camera must inherit its mount transform: {path}")
            # Move the whole housing, retaining the lens's offset inside it.
            _set_transform(mount, optical @ np.diag([1.0, -1.0, -1.0, 1.0]) @ np.linalg.inv(camera_local))
            changes.append(dict(kind="camera", name=name, path=path, parent=parent))
        for name, frame in spec["fingertips"].items():
            parent = _link(stage, root, frame["parent_frame"])
            path = parent + "/calibrated_contact_" + name
            if stage.GetPrimAtPath(parent).IsInstance():
                raise ValueError(f"Contact parent must be editable: {parent}")
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            _set_transform(prim, transform(frame["T_parent_from_child"]))
            changes.append(dict(kind="contact_frame", name=name, path=path, parent=parent))
    return changes


def direction_table(k, scale=2):
    """Float32 RGB: texel-centred NDC -> normalized USD-camera direction.

    Pixel centres in the calibration are integers; NDC pixel centres are
    (pixel + 0.5)/size. K, including the principal point, is baked only here.
    """
    width, height = k["width"], k["height"]
    yy, xx = np.indices((height * scale, width * scale), dtype=np.float64)
    pixels = np.stack([(xx + 0.5) / scale - 0.5, (yy + 0.5) / scale - 0.5], axis=-1)
    xy = pixel_rays(pixels, k)
    directions = np.stack([xy[..., 0], -xy[..., 1], -np.ones_like(xx)], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.min(-directions[..., 2] / np.abs(directions).sum(axis=-1)) <= 0.11:
        raise ValueError("Field of view exceeds this forward-facing LUT's calibrated domain")
    return directions.astype(np.float32)


def exit_table(k, size=2048):
    """Octahedrally encoded USD-view direction -> NDC for Isaac Sim 5.1 RTX.

    Backward/grazing rays are outside this front-facing calibrated camera.
    Finite off-screen coordinates prevent NaNs in texture filtering.
    """
    coord = (np.arange(size, dtype=np.float64) + 0.5) * (2.0 / size) - 1.0
    x, y = np.meshgrid(coord, coord)
    z = 1.0 - np.abs(x) - np.abs(y)
    back = z < 0
    unfolded_x = np.where(back, (1.0 - np.abs(y)) * np.where(x >= 0, 1, -1), x)
    unfolded_y = np.where(back, (1.0 - np.abs(x)) * np.where(y >= 0, 1, -1), y)
    # This legacy generalizedProjection shader encodes view -Z directly.
    # The newer LutAPI documentation uses a Z flip; that yields zero flow here.
    valid = z < -0.1
    rays = np.stack([unfolded_x / np.maximum(-z, 0.1), -unfolded_y / np.maximum(-z, 0.1)], axis=-1)
    if k["image_domain"] == "raw" and k["model"] == "realsense_inverse_brown_conrady":
        pixels = project_realsense(rays, k)
    else:
        pixels = rays * [k["fx"], k["fy"]] + [k["cx"], k["cy"]]
    uv = (pixels + 0.5) / [k["width"], k["height"]]
    uv = np.clip(uv, -1e4, 1e4)
    uv[~valid] = 2.0
    return uv.astype(np.float32)


def _write_exr(path, data):
    import OpenEXR

    channels = {key: np.ascontiguousarray(data[..., i]) for i, key in enumerate("RGB"[: data.shape[-1]])}
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION}, channels) as image:
        image.write(str(path))


def apply_lut(camera, k, directory):
    """Author native renderer attributes; no post-render image transformations."""
    from pxr import Gf, Sdf, UsdGeom

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(k, sort_keys=True).encode()).hexdigest()[:16]
    enter_path = directory / f"{digest}-enter.exr"
    exit_path = directory / f"{digest}-exit.exr"
    _write_exr(enter_path, direction_table(k))
    _write_exr(exit_path, exit_table(k))
    # Isaac Sim 5.1's RTX delegate implements the native generalized projection
    # texture path. The exposed Lut API schema is not consumed by this delegate.
    attrs = {
        "cameraProjectionType": (Sdf.ValueTypeNames.Token, "generalizedProjection"),
        "generalizedProjectionDirectionTexturePath": (Sdf.ValueTypeNames.Asset, str(enter_path)),
        "generalizedProjectionNDCTexturePath": (Sdf.ValueTypeNames.Asset, str(exit_path)),
        "fthetaWidth": (Sdf.ValueTypeNames.Float, float(k["width"])),
        "fthetaHeight": (Sdf.ValueTypeNames.Float, float(k["height"])),
        "fthetaCx": (Sdf.ValueTypeNames.Float, k["width"] / 2),
        "fthetaCy": (Sdf.ValueTypeNames.Float, k["height"] / 2),
        "fthetaMaxFov": (Sdf.ValueTypeNames.Float, 180.0),
    }
    for name, (kind, value) in attrs.items():
        camera.CreateAttribute(name, kind, custom=False).Set(value)
    # Conventional USD lens values are metadata/culling hints, not a second K.
    usd_camera = UsdGeom.Camera(camera)
    usd_camera.CreateFocalLengthAttr(10.0)
    usd_camera.CreateHorizontalApertureAttr(10.0 * k["width"] / k["fx"])
    usd_camera.CreateVerticalApertureAttr(10.0 * k["height"] / k["fy"])
    usd_camera.CreateHorizontalApertureOffsetAttr(0.0)
    usd_camera.CreateVerticalApertureOffsetAttr(0.0)
    usd_camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000))
    return dict(
        backend="isaac_rtx_generalizedProjection_lut",
        postprocessing=False,
        intrinsics=k,
        ray_model="librealsense-2.56.5-inverse-brown-10iter",
        direction_texture=str(enter_path),
        exit_texture=str(exit_path),
        direction_sha256=hashlib.sha256(enter_path.read_bytes()).hexdigest(),
        exit_sha256=hashlib.sha256(exit_path.read_bytes()).hexdigest(),
        direction_size=[k["width"] * 2, k["height"] * 2],
        exit_size=[2048, 2048],
    )


def apply_environment(stage, robot_root, spec, lut_directory):
    """One explicit call before sensor creation/reset; writes only stage overrides."""
    changes = apply_stage(stage, [robot_root], spec)
    cameras = {}
    for name, item in spec["cameras"].items():
        path = robot_root + "/" + CAMERA_PATHS[name]
        cameras[name] = apply_lut(stage.GetPrimAtPath(path), intrinsics(item), lut_directory)
    return dict(
        yaml_sha256=spec["sha256"],
        configuration=spec,
        stage_changes=changes,
        cameras=cameras,
        fingertip_scope="reference_frames_only",
        hardware_access=False,
    )
