"""CPU tests; USD and RealSense oracle cases require usd-core / pyrealsense2."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sim.camera_model import build_remap, covering_pinhole, intrinsics, matrix, pixel_rays, remap_rgb
from sim.environment import (
    CAMERA_PATHS,
    Calibration,
    _set_transform,
    apply_stage,
    camera_size,
    contact_transform,
    load_environment,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config/environment.example.yaml"


def write_config(tmp_path, data):
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def config():
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def test_original_minimal_format_defaults_to_raw(tmp_path):
    data = config()
    for camera in data["cameras"].values():
        camera["intrinsics"].pop("image_domain")
    spec = load_environment(write_config(tmp_path, data))
    assert all(c["intrinsics"]["image_domain"] == "raw" for c in spec["cameras"].values())
    assert camera_size(spec, "right_wrist") == (640, 480)


def test_opt_in(monkeypatch):
    monkeypatch.delenv("AIRSEAI_ENVIRONMENT_YAML", raising=False)
    assert load_environment() is None
    monkeypatch.setenv("AIRSEAI_ENVIRONMENT_YAML", str(EXAMPLE))
    assert load_environment()["sha256"]
    assert load_environment("") is None


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(camera={}),
        lambda d: d["cameras"]["base"]["intrinsics"].update(width_px=0),
        lambda d: d["cameras"]["base"]["intrinsics"].update(fx_px=float("nan")),
        lambda d: d["cameras"]["base"]["intrinsics"].update(distortion_model="unknown"),
        lambda d: d["cameras"]["base"]["intrinsics"].update(distortion_coeffs=[0, 1]),
        lambda d: d["cameras"]["base"]["intrinsics"].update(image_domain="raww"),
        lambda d: d["cameras"]["base"]["extrinsics"].update(parent_frame="../bad"),
        lambda d: d["cameras"]["base"]["extrinsics"]["T_parent_from_child"][0].__setitem__(0, 2),
    ],
)
def test_reject_bad_config(tmp_path, change):
    data = config()
    change(data)
    with pytest.raises(ValueError):
        load_environment(write_config(tmp_path, data))


def test_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("cameras: {}\ncameras: {}\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_environment(path)


@pytest.mark.parametrize("name", CAMERA_PATHS)
def test_full_output_coverage_and_sdk_oracle(name):
    rs = pytest.importorskip("pyrealsense2")
    k = intrinsics(load_environment(EXAMPLE)["cameras"][name])
    maps, valid, receipt = build_remap(k, covering_pinhole(k))
    assert valid.all()
    assert receipt["valid_fraction"] == 1
    sdk = rs.intrinsics()
    sdk.width, sdk.height = k["width"], k["height"]
    sdk.fx, sdk.fy, sdk.ppx, sdk.ppy = k["fx"], k["fy"], k["cx"], k["cy"]
    sdk.model, sdk.coeffs = rs.distortion.inverse_brown_conrady, k["coeffs"]
    yy, xx = np.mgrid[0:480:17, 0:640:19]
    pixels = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(float)
    expected = np.array([rs.rs2_deproject_pixel_to_point(sdk, p.tolist(), 1.0)[:2] for p in pixels])
    error_px = np.linalg.norm((pixel_rays(pixels, k) - expected) * [k["fx"], k["fy"]], axis=-1)
    assert error_px.max() < 0.0001
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[::24, :] = 255
    result = remap_rgb(image, maps, (640, 480))
    assert result.shape == image.shape


def test_zero_distortion_is_identity():
    k = intrinsics(load_environment(EXAMPLE)["cameras"]["base"])
    maps, valid, _ = build_remap(k, matrix(k))
    image = np.random.default_rng(8).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert valid.all()
    np.testing.assert_array_equal(remap_rgb(image, maps), image)
    with pytest.raises(ValueError, match="resolution"):
        remap_rgb(image[:32], maps, (640, 480))


@pytest.mark.parametrize(
    "model, coeffs",
    [
        ("opencv_brown_conrady", [-0.05, 0.01, 0.001, -0.002, 0.002]),
        ("opencv_fisheye", [0.01, -0.005, 0.001, -0.001]),
        ("none", []),
    ],
)
def test_other_declared_models(model, coeffs):
    import cv2

    item = config()["cameras"]["right_wrist"]
    item["intrinsics"].update(distortion_model=model, distortion_coeffs=coeffs)
    k = intrinsics(item)
    pixels = np.array([[0.0, 0.0], [639, 479], [300, 240]])
    rays = pixel_rays(pixels, k)
    if model == "opencv_fisheye":
        back = cv2.fisheye.distortPoints(rays.reshape(-1, 1, 2), matrix(k), np.array(coeffs))
    else:
        back, _ = cv2.projectPoints(
            np.c_[rays, np.ones(3)], np.zeros(3), np.zeros(3), matrix(k), np.array(coeffs) if coeffs else None
        )
    np.testing.assert_allclose(back.reshape(-1, 2), pixels, atol=1e-6)


def test_ideal_domain_and_nondefault_resolution(tmp_path):
    data = config()
    data["cameras"]["right_wrist"]["intrinsics"].update(
        width_px=320,
        height_px=240,
        fx_px=196,
        fy_px=196,
        cx_px=159,
        cy_px=115,
        image_domain="ideal",
    )
    spec = load_environment(write_config(tmp_path, data))
    assert camera_size(spec, "right_wrist") == (320, 240)
    k = intrinsics(spec["cameras"]["right_wrist"])
    points = np.array([[0.0, 0.0], [319, 239]])
    np.testing.assert_allclose(pixel_rays(points, k), (points - [159, 115]) / 196)
    _, valid, receipt = build_remap(k, covering_pinhole(k))
    assert valid.all() and not receipt["distortion_applied"]


def usd_scene():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    roots = [f"/World/envs/env_{i}/Robot" for i in range(2)]
    for i, root in enumerate(roots):
        r = UsdGeom.Xform.Define(stage, root).GetPrim()
        placement = np.eye(4)
        placement[:3, 3] = [3 * i, -2 * i, 0.1]
        placement[:2, :2] = [[0, -1], [1, 0]]
        _set_transform(r, placement)
        for link in (
            "openarm_body_link",
            "openarm_left_hand",
            "openarm_right_hand",
            "openarm_right_right_finger",
            "openarm_right_left_finger",
            "openarm_left_left_finger",
            "openarm_left_right_finger",
        ):
            UsdGeom.Xform.Define(stage, root + "/" + link)
        for path in CAMERA_PATHS.values():
            mount_path = root + "/" + path.rsplit("/", 1)[0]
            _set_transform(UsdGeom.Xform.Define(stage, mount_path).GetPrim(), np.eye(4))
            camera = UsdGeom.Camera.Define(stage, root + "/" + path)
            local = np.eye(4)
            local[2, 3] = -0.0125
            _set_transform(camera.GetPrim(), local)
    return stage, roots


def test_usd_optical_axes_housing_and_contact_frames_follow_parents():
    stage, roots = usd_scene()
    from pxr import UsdGeom

    spec = load_environment(EXAMPLE)
    changes = apply_stage(stage, roots, spec)
    assert len(changes) == 14
    # Re-applying the same calibration must be idempotent.
    apply_stage(stage, roots, spec)
    for root in roots:
        for shift in (0.0, 0.8):  # Moving a link and resetting it needs no re-application.
            hand = stage.GetPrimAtPath(root + "/openarm_right_hand")
            pose = np.eye(4)
            pose[:3, 3] = [shift, -0.3, 0.5]
            pose[:2, :2] = [[0, -1], [1, 0]]
            _set_transform(hand, pose)
            cache = UsdGeom.XformCache()
            for name, item in spec["cameras"].items():
                cam = stage.GetPrimAtPath(root + "/" + CAMERA_PATHS[name])
                parent = cam.GetParent().GetParent()
                world_cam = np.array(cache.GetLocalToWorldTransform(cam)).T
                world_parent = np.array(cache.GetLocalToWorldTransform(parent)).T
                optical = np.linalg.inv(world_parent) @ world_cam @ np.diag([1, -1, -1, 1])
                np.testing.assert_allclose(optical, item["extrinsics"]["T_parent_from_child"], atol=1e-9)
            for name, item in spec["fingertips"].items():
                parent = stage.GetPrimAtPath(root + "/" + item["parent_frame"])
                child = stage.GetPrimAtPath(str(parent.GetPath()) + "/calibrated_contact_" + name)
                actual = np.array(
                    cache.GetLocalToWorldTransform(child)
                    * cache.GetLocalToWorldTransform(parent).GetInverse()
                ).T
                np.testing.assert_allclose(actual, item["T_parent_from_child"], atol=1e-9)


@pytest.mark.parametrize("issue", ["units", "native_lens", "wrong_parent", "missing_camera"])
def test_usd_invalid_asset_rejected(issue):
    stage, roots = usd_scene()
    from pxr import Sdf, UsdGeom

    spec = load_environment(EXAMPLE)
    path = roots[0] + "/" + CAMERA_PATHS["base"]
    if issue == "units":
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    elif issue == "native_lens":
        stage.GetPrimAtPath(path).CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set(
            "opencvPinhole"
        )
    elif issue == "wrong_parent":
        spec["cameras"]["base"]["extrinsics"]["parent_frame"] = "openarm_right_hand"
    else:
        stage.RemovePrim(path)
    with pytest.raises(ValueError):
        apply_stage(stage, roots, spec)


class Tensor:
    """Tiny CPU tensor double for adapter wiring, not a substitute for a Kit test."""

    def __init__(self, a):
        self.a = np.asarray(a)
        self.dtype, self.device, self.shape = self.a.dtype, "cpu", self.a.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def unsqueeze(self, n):
        return Tensor(np.expand_dims(self.a, n))

    def repeat(self, *shape):
        return Tensor(np.tile(self.a, shape))

    def permute(self, *dims):
        return Tensor(self.a.transpose(dims))

    def float(self):
        return Tensor(self.a.astype(np.float32))

    def to(self, device):
        return self

    def __getitem__(self, idx):
        return Tensor(self.a[idx])

    def __truediv__(self, val):
        return Tensor(self.a / val)


def torch_double(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            as_tensor=lambda a, dtype=None, device=None: Tensor(np.asarray(a, dtype=dtype)),
            Tensor=Tensor,
        ),
    )


def test_renderer_readback_recording_policy_and_slots_use_same_mapping(monkeypatch):
    torch_double(monkeypatch)
    spec = load_environment(EXAMPLE)
    spec["cameras"] = {"right_wrist": spec["cameras"]["right_wrist"]}
    k = intrinsics(spec["cameras"]["right_wrist"])
    actual = np.tile(covering_pinhole(k), (2, 1, 1))
    # Model renderer rounding/centering, differently for each clone.
    actual[:, 0, 2] += [0.5, 0.25]
    camera = SimpleNamespace(data=SimpleNamespace(intrinsic_matrices=Tensor(actual)))
    camera.set_intrinsic_matrices = lambda value: None
    camera.update = lambda *a, **kw: None
    calibration = Calibration(spec).bind({"right_wrist": camera})
    assert len(calibration.maps["right_wrist"]) == 2
    np.testing.assert_allclose(calibration.receipt["cameras"]["right_wrist"][1]["source_K"], actual[1])
    # Execute actual Harness methods without importing unavailable Isaac/torch.
    tree = ast.parse((ROOT / "src/sim/harness.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Harness")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("cam_rgb", "image_batch")]
    namespace = {"np": np, "torch": sys.modules["torch"]}
    tree = ast.Module(
        body=[ast.ClassDef(name="Harness", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), "<Harness methods>", "exec"), namespace)
    for slot in (0, 1):
        h = namespace["Harness"]()
        h.calibration, h.slot, h.device = calibration, slot, "cpu"
        image = np.random.default_rng(slot).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        h.cams = {"right_wrist": SimpleNamespace(data=SimpleNamespace(output={"rgb": Tensor(image[None])}))}
        expected = calibration.rgb("right_wrist", image, slot)
        np.testing.assert_array_equal(h.cam_rgb("right_wrist"), expected)
        obs = h.image_batch("right_wrist").numpy()[0].transpose(1, 2, 0)
        np.testing.assert_array_equal(np.rint(obs * 255).astype(np.uint8), expected)
        h.calibration = None
        np.testing.assert_array_equal(h.cam_rgb("right_wrist"), image)


def test_contact_uses_live_fk_not_stale_usd(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "isaaclab.utils.math", SimpleNamespace(matrix_from_quat=lambda q: Tensor(np.eye(3)))
    )
    spec = load_environment(EXAMPLE)
    robot = SimpleNamespace(
        body_names=["openarm_body_link", "openarm_right_right_finger"],
        data=SimpleNamespace(
            body_quat_w=Tensor(np.array([[[1, 0, 0, 0]] * 2])),
            body_pos_w=Tensor([[[3, 4, 1], [3.1, 4.2, 1.3]]]),
        ),
    )
    local = np.array(spec["fingertips"]["right_hand_right_finger"]["T_parent_from_child"])
    actual = contact_transform(robot, spec, "right_hand_right_finger")
    np.testing.assert_allclose(actual[:3, 3], local[:3, 3] + [0.1, 0.2, 0.3])
    robot.data.body_pos_w.a[0, 1, 0] += 0.02
    shifted = contact_transform(robot, spec, "right_hand_right_finger")
    assert shifted[0, 3] - actual[0, 3] == pytest.approx(0.02)
