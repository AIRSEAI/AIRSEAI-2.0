# SPDX-License-Identifier: Apache-2.0
"""Optional camera calibration and finger contact frames. No hardware access.

Load once before constructing sensors, author local USD mounts before reset,
then build RGB maps from the renderer's actual K. Imports Isaac only in bind().
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sim.camera_model import build_remap, covering_pinhole, intrinsics, remap_rgb

CAMERA_PATHS = {
    "base": "openarm_body_link/head_camera/camera",
    "left_wrist": "openarm_left_hand/wrist_camera/camera",
    "right_wrist": "openarm_right_hand/wrist_camera/camera",
}
# Explicit registration for AIRSEAI's supplied OpenArm asset layout only.
LINK_ALIASES = {"openarm_body_link0": "openarm_body_link"}


def add_environment_argument(parser):
    parser.add_argument(
        "--environment-yaml", default=None, help="Camera/contact YAML; defaults to AIRSEAI_ENVIRONMENT_YAML"
    )


def select_environment(args):
    """Resolve a CLI selection once; existing subprocess runners inherit it."""
    if args.environment_yaml is not None:
        os.environ["AIRSEAI_ENVIRONMENT_YAML"] = (
            str(Path(args.environment_yaml).expanduser().resolve()) if args.environment_yaml else ""
        )
    load_environment()  # Fail before Kit launches on malformed input.


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


def load_environment(path=None):
    """None uses AIRSEAI_ENVIRONMENT_YAML; an empty string explicitly disables it."""
    path = os.environ.get("AIRSEAI_ENVIRONMENT_YAML", "") if path is None else path
    if not path:
        return None
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
        # This adapter is for the minimal YAML, whose default is the real raw image.
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


def camera_size(spec, name):
    if not spec or name not in spec["cameras"]:
        return 640, 480
    k = intrinsics(spec["cameras"][name])
    return k["width"], k["height"]


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


def prepare_manager(env, env_ids, spec):
    """Isaac prestartup event: author every clone before physics/sensor initialization."""
    import omni.usd

    env.environment_spec = spec
    env.environment_changes = apply_stage(
        omni.usd.get_context().get_stage(),
        [f"/World/envs/env_{i}/Robot" for i in range(env.num_envs)],
        spec,
    )


class Calibration:
    def __init__(self, spec, changes=()):
        self.spec, self.maps = spec, {}
        self.receipt = dict(
            source=spec["source"],
            yaml_sha256=spec["sha256"],
            configuration=spec,
            stage_changes=list(changes),
            cameras={},
            contact_scope="reference_frames_only",
        )

    def bind(self, cameras):
        """Called after sensors initialize; one source K/map per camera and slot."""
        import torch

        for name, item in self.spec["cameras"].items():
            camera = cameras[name]
            k = intrinsics(item)
            source = covering_pinhole(k)
            current = camera.data.intrinsic_matrices
            camera.set_intrinsic_matrices(
                torch.as_tensor(source, dtype=current.dtype, device=current.device)
                .unsqueeze(0)
                .repeat(current.shape[0], 1, 1)
            )
            actual = camera.data.intrinsic_matrices.detach().cpu().numpy()
            self.maps[name], self.receipt["cameras"][name] = [], []
            for source_k in actual:
                maps, valid, record = build_remap(k, source_k)
                if not valid.all():
                    raise ValueError(f"Renderer FOV does not cover {name}; cannot fill missing pixels")
                self.maps[name].append(maps)
                self.receipt["cameras"][name].append(record)
            camera.update(0.0, force_recompute=True)
        return self

    def rgb(self, name, image, slot=0):
        if name not in self.maps:
            return image
        size = camera_size(self.spec, name)
        return remap_rgb(image, self.maps[name][slot], size)

    def write_receipt(self, path):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def contact_transform(robot, spec, name, relative_to="openarm_body_link0"):
    """Live body-state FK, so Fabric need not mirror moving links into USD."""
    from isaaclab.utils.math import matrix_from_quat

    def body(link):
        names = list(robot.body_names)
        link = link if link in names else LINK_ALIASES.get(link, link)
        i = names.index(link)
        t = np.eye(4)
        t[:3, :3] = matrix_from_quat(robot.data.body_quat_w[0, i]).detach().cpu().numpy()
        t[:3, 3] = robot.data.body_pos_w[0, i].detach().cpu().numpy()
        return t

    frame = spec["fingertips"][name]
    return (
        np.linalg.inv(body(relative_to))
        @ body(frame["parent_frame"])
        @ transform(frame["T_parent_from_child"])
    )


def save_camera_images(cameras, calibration, directory, slot=0):
    """Save the calibrated output, not the pinhole viewport, for runtime inspection."""
    from PIL import Image

    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for name in calibration.spec["cameras"]:
        raw = cameras[name].data.output["rgb"][slot, ..., :3].detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(raw).save(directory / f"{name}-pinhole.png")
        Image.fromarray(calibration.rgb(name, raw, slot)).save(directory / f"{name}-output.png")
    calibration.write_receipt(directory / "environment-receipt.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate environment YAML offline; no Isaac/hardware")
    parser.add_argument("yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = load_environment(args.yaml)
    calibration = Calibration(spec)
    for name, item in spec["cameras"].items():
        k = intrinsics(item)
        _, valid, record = build_remap(k, covering_pinhole(k))
        if not valid.all():
            raise ValueError(f"Insufficient pinhole coverage: {name}")
        calibration.receipt["cameras"][name] = [record]
    calibration.receipt["validation"] = "offline_only; source K is planned, not Isaac readback"
    calibration.write_receipt(args.out)
    print(f"Validated {len(spec['cameras'])} cameras, {len(spec['fingertips'])} contact frames")
