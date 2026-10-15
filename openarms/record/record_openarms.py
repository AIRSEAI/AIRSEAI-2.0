# SPDX-License-Identifier: Apache-2.0
"""Record OpenArms leader/follower demonstrations with isolated tactile acquisition."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import select
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from sensors.tactile_features import dataset_features as tactile_features
from sensors.tactile_process import TactileProcess

MOTORS = tuple([f"joint_{i}" for i in range(1, 8)] + ["gripper"])
SIDES = ("right", "left")
JOINT_KEYS = tuple(f"{side}_{motor}.pos" for side in SIDES for motor in MOTORS)
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def resolve_path(value, base):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in value:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


def vector(value, name, *, nonnegative=False):
    array = np.asarray(value, dtype=float)
    if array.shape != (8,) or not np.isfinite(array).all() or (nonnegative and (array < 0).any()):
        raise ValueError(f"{name} requires eight finite {'nonnegative ' if nonnegative else ''}values")
    return array


def load_config(path, *, teleop_only=False):
    path = Path(path).resolve()
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    if teleop_only:
        cfg["RECORD"] = False
    for key in ("RECORD", "USE_TACTILE", "USE_DEPTH"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be true or false")
    for key in ("NUM_EPISODES", "FPS"):
        positive(cfg[key], key)
        if not isinstance(cfg[key], int):
            raise ValueError(f"{key} must be an integer")
    for key in ("EPISODE_TIME_SEC", "RESET_TIME_SEC", "CAMERA_MAX_AGE_S"):
        positive(cfg[key], key)
    positive(cfg["FRICTION_SCALE"], "FRICTION_SCALE", zero=True)
    if not str(cfg["TASK_DESCRIPTION"]).strip():
        raise ValueError("TASK_DESCRIPTION must not be empty")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", cfg["DATASET_REPO_ID"]):
        raise ValueError("DATASET_REPO_ID must be namespace/name")
    can = cfg["CAN"]
    ports = [can[role][side] for role in ("leader", "follower") for side in SIDES]
    if any(not isinstance(port, str) or not port for port in ports) or len(set(ports)) != 4:
        raise ValueError("CAN requires four distinct, nonempty ports")
    if can["interface"] not in ("socketcan", "slcan", "auto") or not isinstance(can["use_fd"], bool):
        raise ValueError("Invalid CAN interface or use_fd")
    for key in ("bitrate", "data_bitrate", "state_timeout_s"):
        positive(can[key], f"CAN.{key}")
    control, comp, home, rec = (cfg[key] for key in ("CONTROL", "COMPENSATION", "HOMING", "RECORDING"))
    for section, keys in (
        (control, ("calibrate",)),
        (comp, ("gravity", "friction")),
        (rec, ("disk_buffer",)),
        (cfg["TACTILE"], ("previews",)),
    ):
        if any(not isinstance(section[key], bool) for key in keys):
            raise ValueError(f"{keys} must be true or false")
    for key in ("hz", "max_relative_target_deg", "max_tick_s"):
        positive(control[key], f"CONTROL.{key}")
    if control["hz"] < cfg["FPS"]:
        raise ValueError("CONTROL.hz must be at least FPS")
    for key in ("position_kp", "position_kd"):
        vector(control[key], key, nonnegative=True)
    if max(control["position_kp"]) > 500 or max(control["position_kd"]) > 5:
        raise ValueError("MIT gains must satisfy kp <= 500 and kd <= 5")
    for key in ("friction_fc", "friction_k", "friction_fv", "friction_fo", "damping_kd", "torque_limit_nm"):
        vector(comp[key], key, nonnegative=key != "friction_fo")
    positive(comp["friction_velocity_scale"], "friction_velocity_scale", zero=True)
    positive(home["initial_seconds"], "HOMING.initial_seconds")
    if set(control["motor_config"]) != set(MOTORS):
        raise ValueError("motor_config must contain joint_1 through joint_7 and gripper")
    sends, receives = [], []
    for motor, item in control["motor_config"].items():
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"Invalid motor configuration: {motor}")
        send, receive, kind = item
        if any(type(value) is not int or not 0 < value <= 0x7FF for value in (send, receive)):
            raise ValueError(f"Invalid motor CAN IDs: {motor}")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"Invalid motor type: {motor}")
        sends.append(send)
        receives.append(receive)
    if len(set(sends)) != 8 or len(set(receives)) != 8:
        raise ValueError("Motor CAN IDs must be unique within each arm")
    for side in SIDES:
        positions = vector(home[f"{side}_deg"], f"HOMING.{side}_deg")
        limits = control["joint_limits"][side]
        if set(limits) != set(MOTORS):
            raise ValueError(f"Missing joint limits for {side}")
        for motor, position in zip(MOTORS, positions, strict=True):
            bounds = np.asarray(limits[motor], dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid limits for {side}_{motor}")
            if not bounds[0] <= position <= bounds[1]:
                raise ValueError(f"Home position outside limits: {side}_{motor}")
    for key in ("queue_size", "writer_timeout_s", "max_frame_gap_s"):
        positive(rec[key], f"RECORDING.{key}")
    if not isinstance(rec["queue_size"], int) or rec["max_frame_gap_s"] < 1 / cfg["FPS"]:
        raise ValueError("Invalid recording queue size or frame gap")
    if rec["gripper_unit"] not in ("deg", "mm"):
        raise ValueError("gripper_unit must be deg or mm")
    if not np.isfinite([rec["gripper_scale"], rec["gripper_offset"]]).all() or rec["gripper_scale"] == 0:
        raise ValueError("Invalid gripper conversion")
    for key in ("DATASET_PATH", "LOG_FILE_PATH"):
        cfg[key] = str(resolve_path(cfg[key], path.parent))
    control["calibration_dir"] = str(resolve_path(control["calibration_dir"], path.parent))
    comp["urdf_path"] = str(resolve_path(comp["urdf_path"], path.parent))
    if comp["gravity"] and not Path(comp["urdf_path"]).is_file():
        raise ValueError(f"Missing compensation URDF: {comp['urdf_path']}")
    if cfg["USE_TACTILE"]:
        names = cfg["TACTILE_SENSORS"]
        if not names or any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
            raise ValueError("TACTILE_SENSORS needs valid sensor names")
        serials = list(names.values())
        if any(not isinstance(s, str) or not s.strip() for s in serials) or len(set(serials)) != len(serials):
            raise ValueError("Set distinct TACTILE_SENSORS serial numbers")
        for key in ("fps", "sensor_fps", "height", "width", "ready_timeout_s", "max_age_s"):
            positive(cfg["TACTILE"][key], f"TACTILE.{key}")
        for key in ("height", "width"):
            if type(cfg["TACTILE"][key]) is not int or cfg["TACTILE"][key] % 2:
                raise ValueError(f"TACTILE.{key} must be an even integer")
    for name, camera in cfg["CAMERAS"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not str(camera["serial"]).strip():
            raise ValueError(f"Set camera serial number: {name}")
        for key in ("fps", "width", "height"):
            positive(camera[key], f"CAMERAS.{name}.{key}")
            if type(camera[key]) is not int:
                raise ValueError(f"CAMERAS.{name}.{key} must be an integer")
        if camera["width"] % 2 or camera["height"] % 2:
            raise ValueError(f"Camera image dimensions must be even: {name}")
    serials = [str(c["serial"]) for c in cfg["CAMERAS"].values()]
    if len(serials) != len(set(serials)):
        raise ValueError("Camera serial numbers must be distinct")
    return cfg


def make_devices(cfg):
    import lerobot
    from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
    from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
    from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader, BiOpenArmLeaderConfig
    from lerobot.teleoperators.openarm_leader import OpenArmLeaderConfigBase

    if lerobot.__version__ != "0.5.1":
        raise RuntimeError(f"Expected AIRSEAI LeRobot 0.5.1, found {lerobot.__version__}")
    can, control = cfg["CAN"], cfg["CONTROL"]
    common = dict(
        can_interface=can["interface"],
        use_can_fd=can["use_fd"],
        can_bitrate=can["bitrate"],
        can_data_bitrate=can["data_bitrate"],
        motor_config={key: tuple(value) for key, value in control["motor_config"].items()},
    )
    leaders, followers = {}, {}
    for side in SIDES:
        leaders[f"{side}_arm_config"] = OpenArmLeaderConfigBase(
            port=can["leader"][side],
            manual_control=False,
            **common,
        )
        followers[f"{side}_arm_config"] = OpenArmFollowerConfigBase(
            port=can["follower"][side],
            side=None,
            joint_limits=control["joint_limits"][side],
            max_relative_target=control["max_relative_target_deg"],
            disable_torque_on_disconnect=True,
            position_kp=control["position_kp"],
            position_kd=control["position_kd"],
            cameras={},
            **common,
        )
    calibration = Path(control["calibration_dir"])
    leader = BiOpenArmLeader(
        BiOpenArmLeaderConfig(
            id=control["leader_id"],
            calibration_dir=calibration,
            **leaders,
        )
    )
    follower = BiOpenArmFollower(
        BiOpenArmFollowerConfig(
            id=control["follower_id"],
            calibration_dir=calibration,
            **followers,
        )
    )
    return leader, follower


class BusHealth:
    """Reject stale cached motor states after missed CAN responses."""

    def __init__(self, devices, timeout):
        self.timeout = timeout
        self.entries = []
        for device in devices:
            for side in SIDES:
                bus = getattr(device, f"{side}_arm").bus
                seen = {}
                original = bus._process_response

                def receive(motor, message, bus=bus, seen=seen, original=original):
                    previous = bus._last_known_states.get(motor)
                    original(motor, message)
                    if bus._last_known_states.get(motor) is not previous:
                        seen[motor] = time.monotonic()

                bus._process_response = receive
                self.entries.append((bus, seen))

    def check(self):
        now = time.monotonic()
        for bus, seen in self.entries:
            for motor in bus.motors:
                if now - seen.get(motor, -float("inf")) > self.timeout:
                    raise TimeoutError(f"Stale CAN feedback on {bus.port}: {motor}")


def positions(values):
    result = {key: float(values[key]) for key in JOINT_KEYS}
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Nonfinite joint position")
    return result


def follower_positions(follower):
    result = {}
    for side in SIDES:
        bus = getattr(follower, f"{side}_arm").bus
        result.update(
            {f"{side}_{motor}.pos": value for motor, value in bus.sync_read("Present_Position").items()}
        )
    return positions(result)


class Compensation:
    def __init__(self, cfg):
        self.cfg, self.scale = cfg["COMPENSATION"], cfg["FRICTION_SCALE"]
        self.model = None
        self.joints = {}
        if self.cfg["gravity"]:
            import pinocchio as pin

            self.pin = pin
            self.model = pin.buildModelFromUrdf(self.cfg["urdf_path"])
            self.data = self.model.createData()
            for side in SIDES:
                for index in range(1, 8):
                    name = self.cfg["joint_name_template"].format(side=side, index=index)
                    if not self.model.existJointName(name):
                        raise ValueError(f"Missing URDF joint: {name}")
                    joint = self.model.joints[self.model.getJointId(name)]
                    if joint.nq != 1 or joint.nv != 1:
                        raise ValueError(f"Expected one-axis URDF joint: {name}")
                    self.joints[f"{side}_joint_{index}"] = (joint.idx_q, joint.idx_v)

    def torques(self, state):
        gravity = {}
        if self.model is not None:
            q = self.pin.neutral(self.model)
            for name, (iq, _) in self.joints.items():
                q[iq] = np.deg2rad(state[f"{name}.pos"])
            g = self.pin.computeGeneralizedGravity(self.model, self.data, q)
            gravity = {name: float(g[iv]) for name, (_, iv) in self.joints.items()}
        result = {}
        for side in SIDES:
            for i, motor in enumerate(MOTORS):
                name = f"{side}_{motor}"
                friction = 0.0
                if self.cfg["friction"]:
                    velocity = np.deg2rad(float(state[f"{name}.vel"]))
                    friction = self.scale * (
                        self.cfg["friction_fc"][i]
                        * np.tanh(self.cfg["friction_velocity_scale"] * self.cfg["friction_k"][i] * velocity)
                        + self.cfg["friction_fv"][i] * velocity
                        + self.cfg["friction_fo"][i]
                    )
                torque = gravity.get(name, 0.0) + friction
                if not np.isfinite(torque):
                    raise ValueError(f"Nonfinite compensation torque: {name}")
                limit = self.cfg["torque_limit_nm"][i]
                result[name] = float(np.clip(torque, -limit, limit))
        return result

    def apply(self, leader, state):
        torque = self.torques(state)
        for side in SIDES:
            commands = {
                motor: (
                    0.0,
                    self.cfg["damping_kd"][i],
                    state[f"{side}_{motor}.pos"],
                    0.0,
                    torque[f"{side}_{motor}"],
                )
                for i, motor in enumerate(MOTORS)
            }
            getattr(leader, f"{side}_arm").bus._mit_control_batch(commands)


def hold_leader(leader, target, cfg):
    for side in SIDES:
        commands = {
            motor: (
                cfg["CONTROL"]["position_kp"][i],
                cfg["CONTROL"]["position_kd"][i],
                target[f"{side}_{motor}.pos"],
                0.0,
                0.0,
            )
            for i, motor in enumerate(MOTORS)
        }
        getattr(leader, f"{side}_arm").bus._mit_control_batch(commands)


def interpolate(start, end, fraction):
    fraction = float(np.clip(fraction, 0, 1))
    return {key: start[key] + fraction * (end[key] - start[key]) for key in JOINT_KEYS}


def home_target(cfg):
    return {
        f"{side}_{motor}.pos": value
        for side in SIDES
        for motor, value in zip(MOTORS, cfg["HOMING"][f"{side}_deg"], strict=True)
    }


def make_cameras(cfg):
    if not cfg["CAMERAS"]:
        return {}
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    return {
        name: RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=str(item["serial"]),
                width=item["width"],
                height=item["height"],
                fps=item["fps"],
                use_depth=cfg["USE_DEPTH"],
            )
        )
        for name, item in cfg["CAMERAS"].items()
    }


def camera_frames(cameras, cfg):
    frame = {}
    for name, camera in cameras.items():
        if camera.thread is None or not camera.thread.is_alive():
            raise RuntimeError(f"Camera stopped: {name}")
        if camera.frame_lock.acquire(False):
            try:
                camera._record_snapshot = (
                    camera.latest_timestamp,
                    camera.latest_color_frame,
                    camera.latest_depth_frame,
                )
            finally:
                camera.frame_lock.release()
        stamp, rgb, depth = getattr(camera, "_record_snapshot", (None, None, None))
        if stamp is None or rgb is None or time.perf_counter() - stamp > cfg["CAMERA_MAX_AGE_S"]:
            raise TimeoutError(f"Camera has no fresh frame: {name}")
        frame[f"observation.images.{name}"] = rgb.copy()
        if cfg["USE_DEPTH"]:
            if depth is None:
                raise RuntimeError(f"Camera has no depth: {name}")
            frame[f"observation.camera_depth.{name}"] = (
                depth.astype(np.float32)[..., None] * camera._record_depth_scale * 1000
            )
    return frame


def dataset_features(cfg):
    from tactile_codec import QuantProfile

    features = {
        key: {"dtype": "float32", "shape": (16,), "names": list(JOINT_KEYS)}
        for key in ("observation.state", "action")
    }
    features["observation.capture_time_s"] = {"dtype": "float32", "shape": (1,)}
    for name, item in cfg["CAMERAS"].items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (item["height"], item["width"], 3),
        }
        if cfg["USE_DEPTH"]:
            features[f"observation.camera_depth.{name}"] = {
                "dtype": "data",
                "shape": (item["height"], item["width"], 1),
                "profile": QuantProfile("realsense_mm", 1, 0),
            }
    if cfg["USE_TACTILE"]:
        tactile = cfg["TACTILE"]
        features.update(
            tactile_features(cfg["TACTILE_SENSORS"], tactile["previews"], tactile["height"], tactile["width"])
        )
    return features


def joint_vector(state, cfg):
    result = np.array([state[key] for key in JOINT_KEYS], dtype=np.float32)
    rec = cfg["RECORDING"]
    result[[7, 15]] = result[[7, 15]] * rec["gripper_scale"] + rec["gripper_offset"]
    return result


def _writer_worker(root, cfg, messages, replies):
    from lerobot_tactile_ext.writer import LeRobotTactileWriter

    writer = None
    try:
        writer = LeRobotTactileWriter(
            root,
            cfg["FPS"],
            dataset_features(cfg),
            cfg["TASK_DESCRIPTION"],
            robot_type="bi_openarm_follower",
            disk_buffer=cfg["RECORDING"]["disk_buffer"],
        )
        Path(root, "recording_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        replies.send(("ready", None))
        count = 0
        while True:
            operation, payload = messages.get()
            if operation == "frame":
                writer.add_frame(payload)
                count += 1
            elif operation == "save":
                if count == 0:
                    raise ValueError("Cannot save an empty episode")
                stats = writer.save_episode()
                count = 0
                replies.send(("saved", stats))
            elif operation == "discard":
                writer.discard_episode()
                count = 0
                replies.send(("discarded", None))
            elif operation == "stop":
                writer.discard_episode()
                summary = writer.finalize()
                replies.send(("closed", summary))
                break
            else:
                raise ValueError(f"Unknown writer operation: {operation}")
    except BaseException:
        error = traceback.format_exc()
        Path(root).mkdir(parents=True, exist_ok=True)
        Path(root, "recording_error.txt").write_text(error)
        replies.send(("error", error))
        if writer is not None:
            writer.abort()
    finally:
        replies.close()


class RecordingProcess:
    def __init__(self, root, cfg):
        ctx = mp.get_context("spawn")
        self.messages = ctx.Queue(maxsize=cfg["RECORDING"]["queue_size"])
        self.replies, child = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=_writer_worker, args=(str(root), cfg, self.messages, child), name="openarms-writer"
        )
        self.timeout = cfg["RECORDING"]["writer_timeout_s"]
        self.closed = False
        self.failed = False
        try:
            self.process.start()
            child.close()
            self.wait_for("ready")
        except BaseException:
            child.close()
            self.close(force=True)
            raise

    def poll(self):
        if self.replies.poll():
            try:
                operation, payload = self.replies.recv()
            except EOFError as exc:
                self.failed = True
                raise RuntimeError("Writer closed without a result") from exc
            if operation == "error":
                self.failed = True
                raise RuntimeError(f"Dataset writer failed:\n{payload}")
            return operation, payload
        if not self.process.is_alive():
            self.failed = True
            raise RuntimeError(f"Dataset writer exited: {self.process.exitcode}")
        return None

    def put(self, operation, payload=None):
        if not self.process.is_alive():
            raise RuntimeError("Dataset writer is not running")
        try:
            self.messages.put_nowait((operation, payload))
        except queue.Full as exc:
            raise RuntimeError("Recording queue full; episode cannot be recorded without frame loss") from exc

    def wait_for(self, expected):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = self.poll()
            if result is not None:
                if result[0] != expected:
                    raise RuntimeError(f"Expected writer {expected}, received {result[0]}")
                return result[1]
            time.sleep(0.02)
        raise TimeoutError(f"Dataset writer timed out waiting for {expected}")

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        force = force or self.failed
        try:
            if not force and self.process.is_alive():
                self.messages.put(("stop", None), timeout=5)
                # A save/discard acknowledgment can precede the final result.
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    result = self.poll()
                    if result and result[0] == "closed":
                        break
                    time.sleep(0.02)
                else:
                    raise TimeoutError("Dataset finalization timed out")
        finally:
            if self.process.pid is not None:
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)
            self.messages.cancel_join_thread()
            self.messages.close()
            self.replies.close()


def read_command():
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        return "q" if line == "" else line.strip().lower()
    return ""


def control_loop(
    cfg,
    leader,
    follower,
    compensation,
    health,
    cameras,
    tactile,
    writer,
    command_reader=read_command,
    metrics=None,
):
    period, frame_period = 1 / cfg["CONTROL"]["hz"], 1 / cfg["FPS"]
    target = home_target(cfg)
    leader_start = positions(leader.get_action())
    follower_start = follower_positions(follower)
    health.check()
    hold_l, hold_f = leader_start, follower_start
    phase, phase_start, home_duration = "homing", time.monotonic(), cfg["HOMING"]["initial_seconds"]
    saved, frames, pending = 0, 0, None
    next_frame = last_frame = episode_start = 0.0
    logging.info("Commands + Enter: r=start, s=save, d=discard, q=stop. Homing...")
    while True:
        tick = time.monotonic()
        command = command_reader()
        if command == "q":
            return
        if writer is not None:
            reply = writer.poll()
            if reply:
                if reply[0] not in ("saved", "discarded") or pending is None:
                    raise RuntimeError(f"Unexpected writer reply: {reply[0]}")
                saved += reply[0] == "saved"
                pending = None
                if saved >= cfg["NUM_EPISODES"]:
                    return
                leader_start, follower_start = hold_l, hold_f
                phase, phase_start, home_duration = "homing", tick, cfg["RESET_TIME_SEC"]
            if pending is not None and tick - pending > cfg["RECORDING"]["writer_timeout_s"]:
                raise TimeoutError("Dataset writer did not finish the episode")
        if tactile is not None:
            tactile.check_health()
        state = leader.get_action()
        leader_pos = positions(state)
        health.check()
        if phase == "ready" and command == "r":
            phase, episode_start, next_frame, frames = "recording", tick, tick, 0
            logging.info("Teleoperation started%s", " with recording" if writer else "")
        if phase == "recording":
            compensation.apply(leader, state)
            sent = positions(follower.send_action(leader_pos))
            actual = follower_positions(follower)
            health.check()
            hold_l, hold_f = leader_pos, sent
            if writer is not None and tick >= next_frame:
                if frames and tick - last_frame > cfg["RECORDING"]["max_frame_gap_s"]:
                    raise TimeoutError("Recording frame gap exceeded the configured limit")
                frame = {
                    "observation.state": joint_vector(actual, cfg),
                    "action": joint_vector(sent, cfg),
                    "observation.capture_time_s": np.array([tick - episode_start], dtype=np.float32),
                }
                frame.update(camera_frames(cameras, cfg))
                if tactile is not None:
                    frame.update(tactile.get_latest())
                writer.put("frame", frame)
                frames += 1
                last_frame = tick
                next_frame += (int((tick - next_frame) / frame_period) + 1) * frame_period
            if command in ("s", "d") or (
                writer is not None and tick - episode_start >= cfg["EPISODE_TIME_SEC"]
            ):
                phase = "confirm"
                logging.info("Holding position. s=save, d=discard, q=stop (then Enter)")
        else:
            if phase == "homing":
                fraction = (tick - phase_start) / home_duration
                hold_l = interpolate(leader_start, target, fraction)
                hold_f = interpolate(follower_start, target, fraction)
                if fraction >= 1:
                    phase = "ready"
                    logging.info("Ready: r + Enter starts teleoperation")
            hold_leader(leader, hold_l, cfg)
            follower.send_action(hold_f)
            health.check()
        if phase == "confirm" and command in ("s", "d"):
            if writer is not None:
                writer.put("save" if command == "s" and frames else "discard")
                phase, pending = "saving", tick
            else:
                leader_start, follower_start = hold_l, hold_f
                phase, phase_start, home_duration = "homing", tick, cfg["RESET_TIME_SEC"]
        duration = time.monotonic() - tick
        if metrics is not None:
            metrics.append(duration)
        if duration > cfg["CONTROL"]["max_tick_s"]:
            raise TimeoutError(f"Control tick exceeded limit: {duration:.3f}s")
        if duration > period:
            logging.debug("Control overrun: %.1f ms", duration * 1000)
        time.sleep(max(0, period - duration))


def disconnect_devices(devices):
    for device in devices:
        if device is None:
            continue
        for side in SIDES:
            bus = getattr(device, f"{side}_arm").bus
            if bus.is_connected:
                try:
                    bus.disconnect(disable_torque=True)
                except Exception:
                    logging.exception("Could not disconnect %s arm", side)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Recording YAML path")
    parser.add_argument("--teleop-only", action="store_true", help="Control arms without recording")
    parser.add_argument(
        "--check-config", action="store_true", help="Validate settings without opening devices"
    )
    args = parser.parse_args()
    cfg = load_config(args.config, teleop_only=args.teleop_only)
    if args.check_config:
        print("Configuration valid")
        return
    log_path = Path(cfg["LOG_FILE_PATH"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    leader = follower = tactile = writer = None
    cameras = {}
    root = None
    metrics = []
    try:
        compensation = Compensation(cfg)
        leader, follower = make_devices(cfg)
        health = BusHealth([leader, follower], cfg["CAN"]["state_timeout_s"])
        if cfg["USE_TACTILE"]:
            options = dict(cfg["TACTILE"])
            timeout = options.pop("ready_timeout_s")
            tactile = TactileProcess(cfg["TACTILE_SENSORS"], **options)
            tactile.start(timeout)
        cameras = make_cameras(cfg)
        for camera in cameras.values():
            camera.connect()
            if cfg["USE_DEPTH"]:
                camera._record_depth_scale = (
                    camera.rs_profile.get_device().first_depth_sensor().get_depth_scale()
                )
        camera_frames(cameras, cfg)
        if cfg["RECORD"]:
            name = cfg["DATASET_REPO_ID"].replace("/", "__")
            root = Path(cfg["DATASET_PATH"]) / f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
            root.mkdir(parents=True, exist_ok=False)
            writer = RecordingProcess(root, cfg)
            logging.info("Dataset output: %s", root)
        follower.connect(calibrate=cfg["CONTROL"]["calibrate"])
        leader.connect(calibrate=cfg["CONTROL"]["calibrate"])
        control_loop(cfg, leader, follower, compensation, health, cameras, tactile, writer, metrics=metrics)
    except KeyboardInterrupt:
        logging.info("Recording interrupted; pending episode will be discarded")
    except BaseException as exc:
        if root is not None:
            (root / "run_error.json").write_text(json.dumps({"error": str(exc)}, indent=2))
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        disconnect_devices([leader, follower])
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception:
                logging.exception("Camera cleanup failed")
        if tactile is not None:
            try:
                tactile.stop()
            except Exception:
                logging.exception("Tactile cleanup failed")
        if writer is not None:
            try:
                writer.close()
            except Exception:
                if not active_error:
                    raise
                logging.exception("Dataset cleanup failed")
        if metrics:
            summary = {
                "ticks": len(metrics),
                "overruns": int(np.count_nonzero(np.asarray(metrics) > 1 / cfg["CONTROL"]["hz"])),
                "tick_ms_p50_p95_p99_max": [
                    float(value) * 1000 for value in (*np.percentile(metrics, [50, 95, 99]), max(metrics))
                ],
            }
            logging.info("Control timing: %s", summary)
            if root is not None:
                (root / "control_timing.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
