"""Offline tests; real SDK and USD, no mocked renderer claims."""

import copy
from pathlib import Path

import numpy as np
import pytest

from sim.yaml_scene import (
    _write_exr,
    apply_stage,
    direction_table,
    exit_table,
    intrinsics,
    load_environment,
    pixel_rays,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "config/environment.example.yaml"


def sample(table, uv):
    """Independent bilinear texture sampling at normalized texel centres."""
    xy = np.asarray(uv) * [table.shape[1], table.shape[0]] - 0.5
    ij = np.floor(xy).astype(int)
    fraction = xy - ij
    x, y = ij.T
    dx, dy = fraction.T
    x1 = np.clip(x + 1, 0, table.shape[1] - 1)
    y1 = np.clip(y + 1, 0, table.shape[0] - 1)
    x = np.clip(x, 0, table.shape[1] - 1)
    y = np.clip(y, 0, table.shape[0] - 1)
    return (
        table[y, x] * ((1 - dx) * (1 - dy))[:, None]
        + table[y, x1] * (dx * (1 - dy))[:, None]
        + table[y1, x] * ((1 - dx) * dy)[:, None]
        + table[y1, x1] * (dx * dy)[:, None]
    )


@pytest.mark.parametrize("name", ["base", "left_wrist", "right_wrist"])
def test_both_lut_directions_against_independent_sdk(name):
    rs = pytest.importorskip("pyrealsense2")
    k = intrinsics(load_environment(EXAMPLE)["cameras"][name])
    sdk = rs.intrinsics()
    sdk.width, sdk.height = k["width"], k["height"]
    sdk.fx, sdk.fy, sdk.ppx, sdk.ppy = k["fx"], k["fy"], k["cx"], k["cy"]
    sdk.coeffs = k["coeffs"]
    sdk.model = rs.distortion.inverse_brown_conrady
    pixels = np.array([[x, y] for y in np.linspace(0, 479, 21) for x in np.linspace(0, 639, 29)])
    rays = np.array([rs.rs2_deproject_pixel_to_point(sdk, p.tolist(), 1.0) for p in pixels])
    direct = pixel_rays(pixels, k)
    assert np.max(np.abs((direct - rays[:, :2]) * [k["fx"], k["fy"]])) < 0.00015
    entry = sample(direction_table(k), (pixels + 0.5) / [k["width"], k["height"]])
    actual = np.column_stack([entry[:, 0] / -entry[:, 2], entry[:, 1] / entry[:, 2]])
    assert np.max(np.abs((actual - rays[:, :2]) * [k["fx"], k["fy"]])) < 0.001
    # Isaac Sim 5.1 generalizedProjection encodes USD view -Z directly.
    view = rays * [1, -1, -1]
    q = view / np.abs(view).sum(axis=1, keepdims=True)
    encoded = np.column_stack(
        [
            (1 - np.abs(q[:, 1])) * np.where(q[:, 0] >= 0, 1, -1),
            (1 - np.abs(q[:, 0])) * np.where(q[:, 1] >= 0, 1, -1),
        ]
    )
    oct_uv = (encoded + 1) / 2
    projected = sample(exit_table(k), oct_uv) * [k["width"], k["height"]] - 0.5
    error = np.linalg.norm(projected - pixels, axis=1)
    assert error.max() < 0.35  # Half-texel clamp at legacy octahedral border seams.
    interior = ((oct_uv > 0.5 / 2048) & (oct_uv < 1 - 0.5 / 2048)).all(axis=1)
    assert error[interior].max() < 0.01


def test_float32_exr_channel_round_trip(tmp_path):
    exr = pytest.importorskip("OpenEXR")
    data = np.random.default_rng(0).normal(size=(12, 16, 3)).astype(np.float32)
    path = tmp_path / "direction.exr"
    _write_exr(path, data)
    with exr.File(str(path)) as image:
        np.testing.assert_array_equal(image.channels()["RGB"].pixels, data)


def test_reject_rays_outside_exit_table_domain():
    k = intrinsics(load_environment(EXAMPLE)["cameras"]["base"])
    k.update(fx=1.0, fy=1.0, model="none", coeffs=[])
    with pytest.raises(ValueError, match="Field of view"):
        direction_table(k)


@pytest.mark.parametrize(
    "text",
    [
        "cameras: {}\ncameras: {}\n",
        "cameras: {}\nfingertips: {}\n",
        "unknown: 1\n",
    ],
)
def test_reject_bad_yaml(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_environment(path)


def test_official_four_frames_follow_finger_parents():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    spec = copy.deepcopy(load_environment(EXAMPLE))
    spec["cameras"] = {}
    assert set(spec["fingertips"]) == {
        f"{hand}_hand_{finger}_finger" for hand in ("right", "left") for finger in ("right", "left")
    }
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/Robot")
    for frame in spec["fingertips"].values():
        UsdGeom.Xform.Define(stage, "/Robot/" + frame["parent_frame"])
    changes = apply_stage(stage, ["/Robot"], spec)
    assert len(changes) == 4
    for change in changes:
        frame = spec["fingertips"][change["name"]]
        expected = np.array(frame["T_parent_from_child"])
        assert expected[2, 3] == pytest.approx(0.078, abs=1e-6)
        contact = stage.GetPrimAtPath(change["path"])
        before = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(contact)).T
        np.testing.assert_allclose(before, expected, atol=1e-12)
        UsdGeom.Xformable(stage.GetPrimAtPath(change["parent"])).AddTranslateOp().Set(Gf.Vec3d(0.02, 0, 0))
        after = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(contact)).T
        np.testing.assert_allclose(after[:3, 3] - before[:3, 3], [0.02, 0, 0], atol=1e-12)


def test_startup_selection_and_fail_closed(tmp_path, monkeypatch):
    import yaml

    from sim.yaml_scene import load_startup_environment

    monkeypatch.delenv("AIRSEAI_ENVIRONMENT_YAML", raising=False)
    assert load_startup_environment()["sha256"] == load_environment(EXAMPLE)["sha256"]
    monkeypatch.chdir(tmp_path)
    assert load_startup_environment()["source"] == str(EXAMPLE)
    monkeypatch.setenv("AIRSEAI_ENVIRONMENT_YAML", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        load_startup_environment()
    assert load_startup_environment(EXAMPLE)["source"] == str(EXAMPLE)
    monkeypatch.setenv("AIRSEAI_ENVIRONMENT_YAML", "")
    with pytest.raises(ValueError, match="empty"):
        load_startup_environment()
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    del data["cameras"]["base"]
    partial = tmp_path / "partial.yaml"
    partial.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="all three"):
        load_startup_environment(partial)


def test_startup_requires_official_fingers(tmp_path):
    import yaml

    from sim.yaml_scene import load_startup_environment

    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    name = next(iter(data["fingertips"]))
    data["fingertips"][name]["parent_frame"] = "openarm_body_link"
    target = tmp_path / "wrong-parent.yaml"
    target.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="official finger"):
        load_startup_environment(target)
    del data["fingertips"][name]
    target.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="all four"):
        load_startup_environment(target)


def test_shared_startup_applies_every_usd_clone_and_shares_lut(tmp_path, monkeypatch):
    pytest.importorskip("pxr")
    pytest.importorskip("OpenEXR")
    import json
    from types import SimpleNamespace

    from pxr import Gf, Usd, UsdGeom

    from sim.yaml_scene import (
        CAMERA_PATHS,
        OFFICIAL_FINGERS,
        apply_startup_environment,
        load_startup_environment,
    )

    spec = load_startup_environment(EXAMPLE)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    root = "/World/envs/env_0/Robot/"
    for path in CAMERA_PATHS.values():
        UsdGeom.Xform.Define(stage, root + path.split("/")[0])
        UsdGeom.Xform.Define(stage, root + path.rsplit("/", 1)[0])
        UsdGeom.Camera.Define(stage, root + path)
    for parent in OFFICIAL_FINGERS.values():
        UsdGeom.Xform.Define(stage, root + parent)
    clone = UsdGeom.Xform.Define(stage, "/World/envs/env_1")
    clone.GetPrim().GetReferences().AddInternalReference("/World/envs/env_0")
    clone.AddTranslateOp().Set(Gf.Vec3d(2, 0, 0))
    scene = SimpleNamespace(
        stage=stage, num_envs=2, env_prim_paths=["/World/envs/env_0", "/World/envs/env_1"]
    )
    monkeypatch.setenv("AIRSEAI_ENVIRONMENT_OUTPUT", str(tmp_path))
    receipt = apply_startup_environment(scene, spec)
    assert len(receipt["stage_changes"]) == 14
    assert len(list(Path(receipt["receipt_path"]).parent.glob("lut/*.exr"))) == 6
    assert json.loads(Path(receipt["receipt_path"]).read_text())["yaml_sha256"] == spec["sha256"]
    for name in CAMERA_PATHS:
        left, right = [receipt["cameras"][r][name] for r in receipt["robot_roots"]]
        assert left["direction_texture"] == right["direction_texture"]
    for item in receipt["stage_changes"]:
        if item["kind"] == "camera":
            assert (
                stage.GetPrimAtPath(item["path"]).GetAttribute("cameraProjectionType").Get()
                == "generalizedProjection"
            )
        else:
            actual = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(item["path"])).GetLocalTransformation()).T
            np.testing.assert_allclose(
                actual, spec["fingertips"][item["name"]]["T_parent_from_child"], atol=1e-6
            )
    # Missing USD links must abort and never publish a successful receipt.
    spec["cameras"]["base"]["extrinsics"]["parent_frame"] = "missing_link"
    fresh = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(fresh, 1.0)
    scene.stage = fresh
    with pytest.raises(ValueError, match="Missing camera"):
        apply_startup_environment(scene, spec)
    assert len(list(tmp_path.glob("*/environment-receipt.json"))) == 1
