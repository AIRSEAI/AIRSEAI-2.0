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
