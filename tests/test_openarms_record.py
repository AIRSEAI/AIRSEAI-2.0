import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sensors.tactile_features import frame_from_sensor
from sensors.tactile_process import TactileProcess

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openarms/record"))
record = importlib.import_module("record_openarms")


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    cfg = yaml.safe_load(record.DEFAULT_CONFIG.read_text())
    cfg["COMPENSATION"]["urdf_path"] = str(ROOT / "openarms/urdf/openarm_bimanual_dynamics.urdf")
    cfg["CAMERAS"] = {}
    cfg["USE_TACTILE"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return record.load_config(path)


def test_config_paths_and_duplicate_can(config, tmp_path):
    assert Path(config["DATASET_PATH"]).is_absolute()
    config["CAN"]["leader"]["left"] = config["CAN"]["follower"]["left"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="distinct"):
        record.load_config(path)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda c: c["HOMING"]["left_deg"].__setitem__(0, 200), "outside limits"),
        (lambda c: c["CONTROL"].__setitem__("hz", 1), "at least FPS"),
        (lambda c: c.__setitem__("USE_TACTILE", "false"), "true or false"),
        (lambda c: c.__setitem__("FPS", float("nan")), "finite"),
    ],
)
def test_invalid_config(config, tmp_path, change, match):
    change(config)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        record.load_config(path)


def sample(height=4, width=6, fid=1):
    return dict(
        img=np.full((height, width, 3), 80, np.uint8),
        depth=np.full((height, width), 1.2, np.float32),
        deformation=np.full((height, width, 2), -0.7, np.float32),
        shear=np.full((height, width, 2), 0.3, np.float32),
        force=np.arange(6, dtype=np.float32),
        timestamp=time.monotonic(),
        frame_id=fid,
    )


class FakeSensors:
    def __init__(self, configs, **kwargs):
        self.names = configs
        self.count = 0

    def start(self):
        pass

    def get_latest_data(self):
        self.count += 1
        return {name: sample(fid=self.count) for name in self.names}

    def stop(self):
        pass


class FailedSensors(FakeSensors):
    def start(self):
        raise RuntimeError("sensor initialization failed")


class SilentSensors(FakeSensors):
    def get_latest_data(self):
        return {}


def test_tactile_float_fields():
    frame = frame_from_sensor("finger", sample(), height=4, width=6)
    assert frame["observation.depth.finger_tactile"].dtype == np.float32
    assert frame["observation.depth.finger_tactile"][0, 0, 0] == pytest.approx(1.2)
    assert frame["observation.shear.finger_tactile"].shape == (4, 6, 2)


def test_spawn_shared_memory_and_cleanup():
    from multiprocessing import shared_memory

    tactile = TactileProcess({"finger": "fake"}, fps=40, height=4, width=6, manager_factory=FakeSensors)
    try:
        tactile.start(10)
        shm_name = tactile.shm.name
        assert tactile.process.pid is not None
        first = tactile.get_latest()
        time.sleep(0.05)
        second = tactile.get_latest()
        assert second["observation.tactile_status.finger"][2] >= first["observation.tactile_status.finger"][2]
        assert second["observation.depth.finger_tactile"][0, 0, 0] == pytest.approx(1.2)
        tactile.last["_status"][0, 0] = time.monotonic() - 10
        for lock in tactile.locks:
            lock.acquire()
        try:
            with pytest.raises(TimeoutError, match="Stale"):
                tactile.get_latest()
        finally:
            for lock in tactile.locks:
                lock.release()
    finally:
        tactile.stop()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm_name)
    tactile.stop()


@pytest.mark.parametrize("factory,error", [(FailedSensors, RuntimeError), (SilentSensors, TimeoutError)])
def test_tactile_start_failure_cleans_resources(factory, error):
    tactile = TactileProcess({"finger": "fake"}, height=4, width=6, manager_factory=factory)
    with pytest.raises(error):
        tactile.start(1)
    assert tactile.shm is None
    assert tactile.process is None


def test_compensation_formula_and_torque_clamp(config):
    config["COMPENSATION"]["gravity"] = False
    config["COMPENSATION"]["torque_limit_nm"] = [0.1] * 8
    compensation = record.Compensation(config)
    state = {f"{side}_{motor}.vel": 45 for side in record.SIDES for motor in record.MOTORS}
    torques = compensation.torques(state)
    assert len(torques) == 16
    assert all(abs(value) <= 0.1 for value in torques.values())
    state["right_joint_1.vel"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        compensation.torques(state)


def test_named_urdf_gravity(config):
    if importlib.util.find_spec("pinocchio") is None:
        pytest.skip("Pinocchio is not installed")
    compensation = record.Compensation(config)
    state = record.home_target(config)
    state.update({f"{side}_{motor}.vel": 0 for side in record.SIDES for motor in record.MOTORS})
    torques = compensation.torques(state)
    assert len(compensation.joints) == 14
    assert np.isfinite(list(torques.values())).all()


def test_gripper_conversion_and_order(config):
    state = {key: i for i, key in enumerate(record.JOINT_KEYS)}
    config["RECORDING"].update(gripper_scale=-0.5, gripper_offset=2, gripper_unit="mm")
    values = record.joint_vector(state, config)
    assert values[0] == 0
    assert values[7] == -1.5
    assert values[8] == 8
    assert values[15] == -5.5


def test_writer_roundtrip_and_discard(config, tmp_path):
    import json

    from lerobot_tactile_ext import decode_data_frames
    from tactile_codec import PROFILES, dequantize

    pytest.importorskip("av")
    pytest.importorskip("pyarrow")

    config["USE_TACTILE"] = True
    config["TACTILE_SENSORS"] = {"finger": "fake"}
    config["TACTILE"].update(height=4, width=6)
    output = tmp_path / "dataset"
    writer = record.RecordingProcess(output, config)
    frame = frame_from_sensor("finger", sample(), height=4, width=6)
    frame.update(
        {
            "observation.state": np.arange(16, dtype=np.float32),
            "action": np.arange(16, dtype=np.float32) + 1,
            "observation.capture_time_s": np.array([0], np.float32),
            "observation.tactile_status.finger": np.array([1, 0.01, 1], np.float32),
        }
    )
    try:
        writer.put("frame", frame)
        writer.put("discard")
        writer.wait_for("discarded")
        writer.put("frame", frame)
        writer.put("save")
        writer.wait_for("saved")
    finally:
        writer.close()
    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_frames"] == 1
    assert info["total_episodes"] == 1
    depth_key = "observation.depth.finger_tactile"
    decoded = decode_data_frames(output / "videos" / depth_key / "chunk-000/file-000.mkv", 1)
    assert decoded.flat[0] == 31200
    assert dequantize(decoded, PROFILES["daimon_v1_depth"]).flat[0] == pytest.approx(1.2)


def test_writer_invalid_frame_propagates(config, tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("pyarrow")
    writer = record.RecordingProcess(tmp_path / "bad", config)
    try:
        writer.put("frame", {})
        with pytest.raises(RuntimeError, match="missing feature"):
            writer.wait_for("saved")
    finally:
        writer.close(force=True)


class FakeBus:
    def __init__(self):
        self.motors = record.MOTORS
        self.port = "fake"
        self.is_connected = True
        self._last_known_states = {}
        self.commands = []

    def _process_response(self, motor, message):
        self._last_known_states[motor] = dict(message)

    def _mit_control_batch(self, commands):
        self.commands.append(commands)

    def sync_read(self, name):
        return dict.fromkeys(self.motors, 0.0)

    def disconnect(self, disable_torque):
        assert disable_torque
        self.is_connected = False


class FakeRobot:
    def __init__(self):
        self.right_arm = SimpleNamespace(bus=FakeBus())
        self.left_arm = SimpleNamespace(bus=FakeBus())

    def get_action(self):
        return {
            f"{side}_{motor}.{field}": 1.0
            for side in record.SIDES
            for motor in record.MOTORS
            for field in ("pos", "vel")
        }

    def send_action(self, action):
        return {key: min(value, 0.5) for key, value in action.items()}


def test_bus_health_detects_missing_responses():
    robot = FakeRobot()
    health = record.BusHealth([robot], 0.1)
    with pytest.raises(TimeoutError):
        health.check()
    for side in record.SIDES:
        bus = getattr(robot, f"{side}_arm").bus
        for motor in record.MOTORS:
            bus._process_response(motor, {"position": 1})
    health.check()
    health.entries[0][1]["joint_1"] -= 1
    with pytest.raises(TimeoutError):
        health.check()


def test_disconnect_partial_connection():
    robot = FakeRobot()
    robot.right_arm.bus.is_connected = False
    record.disconnect_devices([None, robot])
    assert not robot.left_arm.bus.is_connected


def test_control_records_sent_action_and_saves(config, monkeypatch):
    config["COMPENSATION"]["gravity"] = False
    config["HOMING"]["initial_seconds"] = 0.001
    config["NUM_EPISODES"] = 1
    config["CONTROL"]["max_tick_s"] = 10
    clock = [10.0]
    monkeypatch.setattr(record.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(record.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 0.05))
    commands = iter(["", "", "r", "s", "", "q"])
    captured = []

    class Writer:
        reply = None

        def poll(self):
            result, self.reply = self.reply, None
            return result

        def put(self, operation, payload=None):
            if operation == "frame":
                captured.append(payload)
            elif operation == "save":
                self.reply = ("saved", {})

    record.control_loop(
        config,
        FakeRobot(),
        FakeRobot(),
        record.Compensation(config),
        SimpleNamespace(check=lambda: None),
        {},
        None,
        Writer(),
        lambda: next(commands),
    )
    assert captured
    assert np.all(captured[0]["action"] == 0.5)
    assert np.all(captured[0]["observation.state"] == 0)


def test_recording_queue_overflow_is_explicit():
    import queue

    writer = record.RecordingProcess.__new__(record.RecordingProcess)
    writer.process = SimpleNamespace(is_alive=lambda: True)
    writer.messages = queue.Queue(maxsize=1)
    writer.put("frame", {})
    with pytest.raises(RuntimeError, match="queue full"):
        writer.put("frame", {})


def test_dead_tactile_worker_is_detected():
    tactile = TactileProcess({"finger": "fake"}, height=4, width=6, manager_factory=FakeSensors)
    try:
        tactile.start(10)
        tactile.process.terminate()
        tactile.process.join(timeout=2)
        with pytest.raises(RuntimeError):
            tactile.get_latest()
    finally:
        tactile.stop()


def test_camera_nonblocking_snapshot_and_depth(config):
    import threading

    camera = SimpleNamespace(
        thread=SimpleNamespace(is_alive=lambda: True),
        frame_lock=threading.Lock(),
        latest_timestamp=time.perf_counter(),
        latest_color_frame=np.full((4, 6, 3), 10, np.uint8),
        latest_depth_frame=np.full((4, 6), 50, np.uint16),
        _record_depth_scale=0.001,
    )
    config["USE_DEPTH"] = True
    frame = record.camera_frames({"base": camera}, config)
    assert np.all(frame["observation.camera_depth.base"] == 50)
    camera.frame_lock.acquire()
    try:
        assert record.camera_frames({"base": camera}, config)["observation.images.base"].shape == (4, 6, 3)
        camera._record_snapshot = (
            time.perf_counter() - 10,
            camera.latest_color_frame,
            camera.latest_depth_frame,
        )
        with pytest.raises(TimeoutError):
            record.camera_frames({"base": camera}, config)
    finally:
        camera.frame_lock.release()


def test_control_timeout(config, monkeypatch):
    config["COMPENSATION"]["gravity"] = False
    cfg_time = [0.0]
    monkeypatch.setattr(record.time, "monotonic", lambda: cfg_time[0])
    leader, follower = FakeRobot(), FakeRobot()

    def delayed_send(action):
        cfg_time[0] += 1
        return action

    follower.send_action = delayed_send
    with pytest.raises(TimeoutError, match="Control tick"):
        record.control_loop(
            config,
            leader,
            follower,
            record.Compensation(config),
            SimpleNamespace(check=lambda: None),
            {},
            None,
            None,
            lambda: "",
        )


def test_lerobot_device_configuration(config):
    pytest.importorskip("lerobot.robots.bi_openarm_follower")
    leader, follower = record.make_devices(config)
    for side in record.SIDES:
        arm = getattr(leader, f"{side}_arm")
        assert arm.config.port == config["CAN"]["leader"][side]
        assert not arm.config.manual_control
        assert callable(arm.bus._mit_control_batch)
        arm = getattr(follower, f"{side}_arm")
        assert arm.config.joint_limits == config["CONTROL"]["joint_limits"][side]
        assert arm.config.max_relative_target == config["CONTROL"]["max_relative_target_deg"]
        assert not arm.cameras
