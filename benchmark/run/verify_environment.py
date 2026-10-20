#!/usr/bin/env python3
"""Render official fingers and calibrated RGB, checking frames against live FK."""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from sim.environment import add_environment_argument, select_environment
from sim.runtime import install_error_handler

install_error_handler()
p = argparse.ArgumentParser()
p.add_argument("--robot_usd", required=True)
p.add_argument("--out", required=True)
p.add_argument("--reference-frames", help="Prior render-validation.json for fixed hand close-up cameras")
add_environment_argument(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
select_environment(a)
a.enable_cameras = True
app = AppLauncher(a).app

import isaaclab.sim as sim_utils
import numpy as np
import omni.usd
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from PIL import Image
from pxr import Gf, UsdGeom

from sim.environment import save_camera_images
from sim.envs.bimanual_tactile import make_preview_cfg
from sim.harness import Harness, look_at
from sim.tactile import apply_sdf_collision

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cfg = make_preview_cfg()
for key in cfg.robot.init_state.joint_pos:
    if "finger_joint" in key:
        cfg.robot.init_state.joint_pos[key] = 0.018
h = Harness(
    cfg,
    device=a.device,
    grid=(36, 48),
    robot_usd=a.robot_usd,
    environment_yaml=a.environment_yaml,
    scene_cam=((1.1, -1.4, 1.05), (0, 0, 0.5)),
)
stage = omni.usd.get_context().get_stage()
assert apply_sdf_collision("/World/envs/env_0/manipuland", sdf_resolution=256)
markers = []
for item in h.calibration.receipt["stage_changes"]:
    if item["kind"] != "contact_frame":
        continue
    root = item["path"]
    for axis, color in zip("XYZ", ((1.0, 0.08, 0.04), (0.05, 1.0, 0.08), (0.05, 0.25, 1.0)), strict=True):
        path = root + "/axis_" + axis
        line = UsdGeom.Cylinder.Define(stage, path)
        line.CreateAxisAttr(axis)
        line.CreateRadiusAttr(0.0008)
        line.CreateHeightAttr(0.023)
        line.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        position = [0.0, 0.0, 0.0]
        position["XYZ".index(axis)] = 0.0115
        UsdGeom.Xformable(line).AddTranslateOp().Set(Gf.Vec3d(*position))
        markers.append(path)
    sphere = UsdGeom.Sphere.Define(stage, root + "/origin")
    sphere.CreateRadiusAttr(0.0015)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 0.15)])
    markers.append(str(sphere.GetPath()))

review_cams = {"overview": h.scene_cam}
if a.reference_frames:
    reference = json.loads(Path(a.reference_frames).read_text())["frames"]["home"]["T_world_from_contact"]
    for side in ("right", "left"):
        pair = [np.array(t) for name, t in reference.items() if name.startswith(side + "_")]
        centre = (pair[0][:3, 3] + pair[1][:3, 3]) / 2
        eye = centre + pair[0][:3, 0] * 0.18 - pair[0][:3, 1] * 0.10
        review_cams[side] = TiledCamera(
            TiledCameraCfg(
                prim_path="/World/review_" + side,
                offset=TiledCameraCfg.OffsetCfg(
                    pos=eye.tolist(), rot=look_at(eye, centre), convention="opengl"
                ),
                width=960,
                height=720,
                update_latest_camera_pose=True,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(),
            )
        )
h.start()
h.initial_render()
for _ in range(60):
    h.robot.set_joint_position_target(h.robot.data.default_joint_pos)
    h.robot.write_data_to_sim()
    h.step()


def render_review(name, camera_key):
    sensor = review_cams[camera_key]
    for _ in range(8):
        h.step()
        sensor.update(1 / 60, force_recompute=True)
    Image.fromarray(sensor.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)).save(
        out / (name + ".png")
    )
    camera = sensor.data
    pose = np.eye(4)
    pose[:3, :3] = matrix_from_quat(camera.quat_w_ros[0]).cpu().numpy()
    pose[:3, 3] = camera.pos_w[0].cpu().numpy()
    contacts = {
        key: (body_matrix(base_i) @ h.contact_transform(key)).tolist()
        for key in h.environment_spec["fingertips"]
    }
    (out / (name + ".camera.json")).write_text(
        json.dumps(
            dict(
                K=camera.intrinsic_matrices[0].cpu().tolist(),
                T_world_from_optical=pose.tolist(),
                contacts=contacts,
            ),
            indent=2,
        )
    )


records = {}
from isaaclab.utils.math import matrix_from_quat

names = list(h.robot.body_names)
base_i = names.index("openarm_body_link")


def body_matrix(index):
    t = np.eye(4)
    t[:3, :3] = matrix_from_quat(h.robot.data.body_quat_w[0, index]).cpu().numpy()
    t[:3, 3] = h.robot.data.body_pos_w[0, index].cpu().numpy()
    return t


for pose in ("home", "moved"):
    if pose == "moved":
        target = h.robot.data.default_joint_pos.clone()
        for side in ("right", "left"):
            target[0, list(h.robot.joint_names).index(f"openarm_{side}_joint1")] += 0.12
        for i, name in enumerate(h.robot.joint_names):
            if "finger_joint" in name:
                target[0, i] = 0.025
        for _ in range(90):
            h.robot.set_joint_position_target(target)
            h.robot.write_data_to_sim()
            h.step()
    world_base = body_matrix(base_i)
    frames = {name: world_base @ h.contact_transform(name) for name in h.environment_spec["fingertips"]}
    records[pose] = {
        "T_world_from_contact": {k: v.tolist() for k, v in frames.items()},
        "joint_names": list(h.robot.joint_names),
        "joint_pos": h.robot.data.joint_pos[0].cpu().tolist(),
    }
    for key in review_cams:
        render_review(pose + "-" + key + "-axes", key)
    # Keep diagnostics out of the actual policy/recording images.
    for path in markers:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    for _ in range(8):
        h.sim.render()
        for cam in h.cams.values():
            cam.update(1 / 60, force_recompute=True)
    folder = out / pose
    save_camera_images(h.cams, h.calibration, folder)
    for key in h.cams:
        tensor = h.image_batch(key)[0].permute(1, 2, 0).cpu().numpy()
        assert np.array_equal(np.rint(tensor * 255).astype(np.uint8), h.cam_rgb(key))
    for path in markers:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeVisible()
    errors = {}
    for key, item in h.environment_spec["cameras"].items():
        parent_name = item["extrinsics"]["parent_frame"].replace("openarm_body_link0", "openarm_body_link")
        expected = body_matrix(names.index(parent_name)) @ np.array(item["extrinsics"]["T_parent_from_child"])
        cam = h.cams[key]
        actual = np.eye(4)
        actual[:3, :3] = matrix_from_quat(cam.data.quat_w_ros[0]).cpu().numpy()
        actual[:3, 3] = cam.data.pos_w[0].cpu().numpy()
        errors[key] = float(np.max(np.abs(actual - expected)))
        (out / (pose + "-pose-check.json")).write_text(
            json.dumps(
                dict(camera=key, expected=expected.tolist(), actual=actual.tolist(), errors=errors), indent=2
            )
        )
        assert errors[key] < 1e-4, (key, errors[key])
    records[pose]["camera_pose_max_abs_error"] = errors

report = dict(
    isaac_kit_executed=True,
    hardware_access=False,
    asset=a.robot_usd,
    image_batch_equals_recording=True,
    frames=records,
    calibration=h.calibration.receipt,
)
(out / "render-validation.json").write_text(json.dumps(report, indent=2) + "\n")
print("OFFICIAL_FOUR_FINGERS_RENDER_OK", flush=True)
# Kit sensor teardown can hang after successful standalone capture.
os._exit(0)
