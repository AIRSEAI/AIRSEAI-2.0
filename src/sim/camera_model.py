# SPDX-License-Identifier: Apache-2.0
"""RGB pinhole-to-camera remapping, independent of Isaac and hardware.

RealSense inverse Brown follows librealsense 2.56.5's deprojection algorithm,
validated against pyrealsense2 2.56.5.9235. Do not substitute the older wiki's
single polynomial or cv2.undistortPoints with the same coefficient values.
"""

import hashlib
from numbers import Real

import numpy as np

IMPLEMENTATION = "librealsense-2.56.5-inverse-brown-deproject-10iter-v1"


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
        "opencv_brown_conrady": (5,),
        "opencv_fisheye": (4,),
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
    """Destination pixel -> normalized ideal ray, before source K.

    Inverse sampling is the direction cv2.remap needs, so there are no holes
    from forward splatting. Returns x/z,y/z, not a depth-scaled 3D point.
    """
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
    import cv2

    flat = points.reshape(-1, 1, 2)
    if k["model"] == "opencv_fisheye":
        rays = cv2.fisheye.undistortPoints(flat, matrix(k), np.array(k["coeffs"]))
    else:
        rays = cv2.undistortPointsIter(
            flat,
            matrix(k),
            np.array(k["coeffs"]),
            None,
            None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
        )
    if not np.isfinite(rays).all():
        raise ValueError("Invalid undistorted rays")
    return rays.reshape(points.shape)


def project_realsense(rays, k):
    """SDK-compatible projection used only to check map invertibility."""
    x, y = np.asarray(rays)[..., 0], np.asarray(rays)[..., 1]
    k1, k2, p1, p2, k3 = k["coeffs"]
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xr, yr = x * f, y * f
    xd = xr + 2 * p1 * xr * yr + p2 * (r2 + 2 * xr * xr)
    yd = yr + 2 * p2 * xr * yr + p1 * (r2 + 2 * yr * yr)
    return np.stack([xd * k["fx"] + k["cx"], yd * k["fy"] + k["cy"]], axis=-1)


def image_rays(k):
    yy, xx = np.indices((k["height"], k["width"]), dtype=np.float64)
    return pixel_rays(np.stack([xx, yy], axis=-1), k)


def covering_pinhole(k):
    """Centered square-pixel source K covering all target rays plus a margin."""
    rays = image_rays(k)
    extents = np.max(np.abs(rays), axis=(0, 1))
    f = min((k["width"] - 5) / (2 * extents[0]), (k["height"] - 5) / (2 * extents[1]))
    return np.array([[f, 0.0, k["width"] / 2], [0.0, f, k["height"] / 2], [0.0, 0.0, 1.0]])


def build_remap(k, source_k, source_size=None):
    source_k = np.asarray(source_k, dtype=np.float64)
    if (
        source_k.shape != (3, 3)
        or not np.isfinite(source_k).all()
        or source_k[0, 0] <= 0
        or source_k[1, 1] <= 0
        or not np.allclose(source_k[2], [0, 0, 1])
        or abs(source_k[0, 1]) > 1e-10
        or abs(source_k[1, 0]) > 1e-10
    ):
        raise ValueError("Invalid source pinhole K")
    width, height = source_size or (k["width"], k["height"])
    rays = image_rays(k)
    mx = (rays[..., 0] * source_k[0, 0] + source_k[0, 2]).astype(np.float32)
    my = (rays[..., 1] * source_k[1, 1] + source_k[1, 2]).astype(np.float32)
    valid = (mx >= 0) & (mx <= width - 1) & (my >= 0) & (my <= height - 1)
    receipt = {
        "image_domain": k["image_domain"],
        "distortion_model": k["model"],
        "distortion_coeffs": k["coeffs"],
        "target_K": matrix(k).tolist(),
        "source_K": source_k.tolist(),
        "output_size": [k["width"], k["height"]],
        "source_size": [width, height],
        "implementation": IMPLEMENTATION,
        "distortion_applied": bool(k["image_domain"] == "raw" and np.any(k["coeffs"])),
        "backend": "cpu_opencv_inverse_sampling_rgb",
        "native_lens_distortion": False,
        "valid_fraction": float(valid.mean()),
        "border": "constant_black_invalid_mask",
        "interpolation": "bilinear",
        "map_sha256": hashlib.sha256(mx.tobytes() + my.tobytes()).hexdigest(),
    }
    return (mx, my), valid, receipt


def remap_rgb(image, maps, source_size=None):
    import cv2

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("RGB/RGBA image expected; depth requires a separate contract")
    if source_size is not None and (image.shape[1], image.shape[0]) != tuple(source_size):
        raise ValueError("Source image resolution changed after map construction")
    return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
