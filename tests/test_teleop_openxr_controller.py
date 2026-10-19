"""Pure-logic checks for the VR adapter; Isaac Sim is never imported or started."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "sim" / "teleop_openxr.py"


def _module(monkeypatch):
    class FakeKeyboard:
        def __init__(self, _cfg):
            self.callbacks = {}

        def add_callback(self, key, callback):
            self.callbacks[key] = callback

        def reset(self):
            pass

        def advance(self):
            return np.zeros(6)

    class FakeOpenXRDevice:
        def __init__(self, _cfg):
            pass

        def reset(self):
            pass

        def advance(self):
            return self._get_raw_data()

    class FakeXRCore:
        current = None

        @classmethod
        def get_singleton(cls):
            return cls.current

    packages = {
        "isaaclab": {},
        "isaaclab.devices": {"Se3Keyboard": FakeKeyboard, "Se3KeyboardCfg": lambda **kw: kw},
        "isaaclab.devices.openxr": {},
        "isaaclab.devices.openxr.openxr_device": {
            "OpenXRDevice": FakeOpenXRDevice,
            "OpenXRDeviceCfg": object,
        },
        "omni": {},
        "omni.kit": {},
        "omni.kit.xr": {},
        "omni.kit.xr.core": {
            "XRCore": FakeXRCore,
            "XRPoseValidityFlags": types.SimpleNamespace(POSITION_VALID=1, ORIENTATION_VALID=2),
        },
    }
    for name, attrs in packages.items():
        fake = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(fake, key, value)
        monkeypatch.setitem(sys.modules, name, fake)
    spec = importlib.util.spec_from_file_location("teleop_openxr_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, FakeXRCore


def _pose(x=0.0):
    return np.array([x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def _sample(module, x, source="controller", squeeze=1.0, trigger=0.0, button_0=False, button_1=False):
    return module.XRHandSample(
        _pose(x), source, "grip" if source == "controller" else "palm", trigger, squeeze, button_0, button_1
    )


class FakeHarness:
    control_dt = 0.05

    def __init__(self):
        self.selected = "right"
        self.commands = {side: np.zeros(3) for side in ("right", "left")}
        self.jaws = {side: 0.04 for side in ("right", "left")}
        self._chains = {"right": {"grip_ids": [3]}, "left": {"grip_ids": [7]}}
        self._position_ids = {"right": slice(0, 3), "left": slice(4, 7)}
        self._hold = None
        self.ik_options = []
        self.robot = types.SimpleNamespace(
            set_joint_position_target=self._commit_latch,
            write_data_to_sim=lambda: None,
        )

    def select_arm(self, side):
        self.selected = side

    def reset_ik(self):
        pass

    def grip_measured(self):
        return self.jaws[self.selected]

    def tcp_pos(self):
        return self.commands[self.selected].copy()

    def tcp_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def latch(self):
        self._hold = np.array([
            [*self.commands["right"], self.jaws["right"], *self.commands["left"], self.jaws["left"]]
        ])

    def hold_grip(self, side, jaw):
        self._hold[0, self._chains[side]["grip_ids"][0]] = jaw

    def _commit_latch(self, target):
        self.commands = {side: target[0, ids].copy() for side, ids in self._position_ids.items()}
        self.jaws = {side: float(target[0, chain["grip_ids"][0]]) for side, chain in self._chains.items()}

    def step_ik_toward(self, pos, jaw, _quat, **options):
        self.ik_options.append(options)
        # Like Harness, each IK starts from the latched command vector.
        target = self._hold.copy()
        target[0, self._position_ids[self.selected]] = pos
        target[0, self._chains[self.selected]["grip_ids"][0]] = jaw
        self._commit_latch(target)


def _adapter(module, *, vr_position_gain=1.0, knobs=None):
    harness = FakeHarness()
    device = module.OpenXRHandDevice(
        None, harness, knobs={"open_width": 0.04} if knobs is None else knobs,
        pos_speed=0.08, rot_speed=0.6,
        jaw_speed=0.02, vr_position_gain=vr_position_gain,
    )
    device.begin_episode({"side": "right"})
    return device, harness


def test_controller_read_requires_active_pose_and_squeeze(monkeypatch):
    module, xr_core = _module(monkeypatch)

    class Quat:
        def GetReal(self):
            return 1.0

        def GetImaginary(self):
            return (0.0, 0.0, 0.0)

    class Matrix:
        def ExtractTranslation(self):
            return (0.1, 0.0, 0.0)

        def ExtractRotationQuat(self):
            return Quat()

    class Input:
        active = True
        squeeze = 1.0

        def get_all_virtual_world_poses(self):
            return {"grip": types.SimpleNamespace(validity_flags=3, pose_matrix=Matrix())}

        def is_active(self):
            return self.active

        def has_input_gesture(self, name, kind):
            return (name, kind) in {("squeeze", "value"), ("trigger", "value")}

        def get_input_gesture_value(self, name, _kind):
            return self.squeeze if name == "squeeze" else 0.25

    left, right = Input(), Input()
    xr_core.current = types.SimpleNamespace(
        get_input_device=lambda path: left if path.endswith("left") else right
    )
    reader = module.TrackedHands(object())
    data = reader.advance()
    assert set(data) == {"left", "right"}
    assert data["left"].source == "controller"
    assert np.allclose(data["left"].pose, _pose(0.1))
    assert data["left"].squeeze == 1.0
    class OldKitInput(Input):
        def is_active(self):
            raise AttributeError("Kit 107 does not expose is_active")

    left = OldKitInput()
    assert set(reader.advance()) == {"left", "right"}
    left = Input()
    left.active = False
    assert set(reader.advance()) == {"right"}
    left.active = True
    left.squeeze = float("nan")
    assert set(reader.advance()) == {"right"}
    left.squeeze = 1.0
    left.has_input_gesture = lambda *_: (_ for _ in ()).throw(RuntimeError("XR detached"))
    assert set(reader.advance()) == {"right"}


def test_bimanual_commands_merge_and_tracking_loss_rebaselines(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {side: _sample(module, 0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples

    device._dev.callbacks["SPACE"]()  # The vector collector starts paused.
    cmd = device.advance(0.0)
    assert cmd.tracking_valid and cmd.toggle_pause
    device.apply(cmd)  # Clutch first press sets the pose baseline.
    samples = {"right": _sample(module, 0.02), "left": _sample(module, -0.02)}
    device.advance(0.05)  # First engaged frame establishes the baseline.
    samples = {"right": _sample(module, 0.04), "left": _sample(module, -0.04)}
    cmd = device.advance(0.05)
    device.apply(cmd)
    assert np.isclose(harness.commands["right"][0], 0.004)
    assert np.isclose(harness.commands["left"][0], -0.004)
    assert len(harness.ik_options) == 2
    for options in harness.ik_options:
        assert options == dict(max_speed=0.08, max_ang_speed=float(np.degrees(0.6)), max_joint_speed_deg_s=30.0)
    snapshot = device.diagnostics_snapshot()
    assert snapshot["tracking_valid"] and snapshot["recording_active"]
    assert snapshot["position_gain"] == 1.0
    assert snapshot["hands"]["right"]["pose_xyz_wxyz"][0] == 0.04
    assert snapshot["hands"]["left"]["squeeze"] == 1.0

    before = {side: value.copy() for side, value in harness.commands.items()}
    samples = {"right": _sample(module, 0.03)}  # Left controller disappears.
    cmd = device.advance(0.10)
    assert not cmd.tracking_valid
    # collection.command() skips apply and simulation stepping for invalid tracking.
    assert all(np.array_equal(harness.commands[side], before[side]) for side in before)

    samples = {"right": _sample(module, 1.0), "left": _sample(module, -1.0)}
    cmd = device.advance(0.15)
    assert cmd.tracking_valid
    device.apply(cmd)
    assert all(np.array_equal(harness.commands[side], before[side]) for side in before)


@pytest.mark.parametrize("gain, expected", [(1.0, 0.001), (2.0, 0.002)])
def test_controller_position_gain_scales_small_motion(monkeypatch, gain, expected):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module, vr_position_gain=gain)
    samples = {side: _sample(module, 0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.0))
    device.advance(0.05)  # Establish both grip-pose baselines.
    samples["right"] = _sample(module, 0.001)
    device.apply(device.advance(0.10))
    assert np.isclose(harness.commands["right"][0], expected)
    assert device.diagnostics_snapshot()["position_gain"] == gain


def test_position_gain_clamps_after_raw_controller_jump_check(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module, vr_position_gain=2.0)
    samples = {side: _sample(module, 0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.0))
    device.advance(0.05)

    # Raw 0.2 m is below the tracking-jump threshold; gain is applied before
    # the per-frame 0.004 m motion limit.
    samples["right"] = _sample(module, 0.2)
    device.apply(device.advance(0.10))
    assert np.isclose(harness.commands["right"][0], 0.004)

    # A raw 0.3 m step is a tracking jump and must still be rejected.
    samples["right"] = _sample(module, 0.5)
    device.apply(device.advance(0.15))
    assert np.isclose(harness.commands["right"][0], 0.004)
    samples["right"] = _sample(module, 0.501)
    device.apply(device.advance(0.20))
    assert np.isclose(harness.commands["right"][0], 0.006)


def test_controller_identical_pose_and_trigger_only_hold_joints(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {side: _sample(module, 0.0, squeeze=0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.0))

    # Trial 15's re-clutch had an old joint target and a different measured
    # TCP. Even with that mismatch, an unchanged grip pose must not run IK.
    harness.tcp_pos = lambda: np.array([0.1, 0.0, 0.0])
    samples["right"] = _sample(module, 0.0, squeeze=1.0)
    device.apply(device.advance(0.05))
    for frame in range(2, 22):
        device.apply(device.advance(frame * 0.05))
    assert harness.ik_options == []
    np.testing.assert_array_equal(harness.commands["right"], np.zeros(3))

    samples["right"] = _sample(module, 0.0, squeeze=1.0, trigger=1.0)
    device.apply(device.advance(1.10))
    assert harness.jaws["right"] < 0.04
    assert harness.ik_options == []
    samples["right"] = _sample(module, 0.0, squeeze=0.0, trigger=1.0)
    device.apply(device.advance(1.15))
    samples["right"] = _sample(module, 0.0, squeeze=1.0, trigger=1.0)
    device.apply(device.advance(1.20))
    assert harness.ik_options == []


@pytest.mark.parametrize(
    "knobs, expected_close",
    [({"open_width": 0.04, "close_width": 0.02225}, 0.02225),
     ({"open_width": 0.04}, 0.0),
     ({"open_width": 0.04, "close_width": 0.08}, 0.04)],
)
def test_full_trigger_uses_task_close_width_without_moving_arm(monkeypatch, knobs, expected_close):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module, knobs=knobs)
    samples = {side: _sample(module, 0.0, squeeze=0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.0))
    samples["right"] = _sample(module, 0.0, squeeze=1.0, trigger=1.0)
    for frame in range(1, 46):
        device.apply(device.advance(frame * 0.05))
    assert np.isclose(device._jaw_targets["right"], expected_close)
    assert np.isclose(harness.jaws["right"], expected_close)
    np.testing.assert_array_equal(harness.commands["right"], np.zeros(3))
    assert harness.ik_options == []
    samples["right"] = _sample(module, 0.0, squeeze=1.0, trigger=0.5)
    device.apply(device.advance(2.30))
    assert np.isclose(device._jaw_targets["right"], (knobs["open_width"] + expected_close) / 2)
    assert harness.ik_options == []


@pytest.mark.parametrize("close_width", [None, -0.001, float("nan"), float("inf"), "bad"])
def test_invalid_task_close_width_is_rejected(monkeypatch, close_width):
    module, _ = _module(monkeypatch)
    with pytest.raises(ValueError, match="VR close_width must be finite and non-negative"):
        _adapter(module, knobs={"open_width": 0.04, "close_width": close_width})


def test_controller_deadband_accumulates_slow_motion_and_rotation_only(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {side: _sample(module, 0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.0))
    device.apply(device.advance(0.05))  # First clutch frame sets the anchor.

    quats = []
    original_step = harness.step_ik_toward

    def record_pose(pos, jaw, quat, **options):
        quats.append(np.asarray(quat))
        return original_step(pos, jaw, quat, **options)

    harness.step_ik_toward = record_pose
    for tick, x in ((0.10, 0.0002), (0.15, 0.0004)):
        samples["right"] = _sample(module, x)
        device.apply(device.advance(tick))
    assert quats == []
    samples["right"] = _sample(module, 0.0006)
    device.apply(device.advance(0.20))
    assert len(quats) == 1
    assert np.isclose(harness.commands["right"][0], 0.0006)

    rotated = _sample(module, 0.0006)
    rotated.pose[3] = np.cos(0.005)
    rotated.pose[6] = np.sin(0.005)
    samples["right"] = rotated
    device.apply(device.advance(0.25))
    assert len(quats) == 2
    assert quats[-1][3] > 0  # Rotation-only command reached IK.
    assert np.isclose(harness.commands["right"][0], 0.0006)


def test_controller_targets_rebase_on_measured_tcp_each_moving_frame(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {side: _sample(module, 0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.0))
    device.apply(device.advance(0.05))
    samples["right"] = _sample(module, 0.02)
    device.apply(device.advance(0.10))
    assert np.isclose(harness.commands["right"][0], 0.004)
    harness.commands["right"] = np.zeros(3)  # Simulate physical lag behind the target.
    samples["right"] = _sample(module, 0.04)
    device.apply(device.advance(0.15))
    assert np.isclose(harness.commands["right"][0], 0.004)


@pytest.mark.parametrize("gain, expected", [(1.0, 0.001), (2.0, 0.002)])
def test_palm_position_gain_leaves_rotation_unchanged(monkeypatch, gain, expected):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module, vr_position_gain=gain)
    samples = {"right": _sample(module, 0.0, source="palm")}
    device._xr.advance = lambda: samples
    device.advance(0.0)
    moved = _sample(module, 0.001, source="palm")
    moved.pose[3] = np.cos(0.005)
    moved.pose[6] = np.sin(0.005)
    samples = {"right": moved}
    cmd = device.advance(0.05)
    assert np.isclose(cmd.delta_pos[0], expected)
    assert np.isclose(cmd.delta_rot[2], 0.01)
    device.apply(cmd)
    assert np.isclose(harness.commands["right"][0], expected)

    saturated = _sample(module, 0.011, source="palm")
    saturated.pose[3:] = moved.pose[3:]
    samples = {"right": saturated}
    cmd = device.advance(0.10)
    assert np.isclose(cmd.delta_pos[0], 0.004)
    assert np.isclose(cmd.delta_rot[2], 0.0)


@pytest.mark.parametrize("gain", [0.0, -1.0, float("nan"), float("inf"), None, "bad"])
def test_position_gain_requires_positive_finite_value(monkeypatch, gain):
    module, _ = _module(monkeypatch)
    with pytest.raises(ValueError, match="VR position gain must be finite and positive"):
        _adapter(module, vr_position_gain=gain)


def test_controller_clutch_and_button_edges(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {side: _sample(module, 0.0, squeeze=0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device.advance(0.0)  # Buttons must first be seen released.

    samples["left"] = _sample(module, 0.2, squeeze=0.0, button_0=True)
    cmd = device.advance(0.05)
    assert cmd.toggle_pause
    assert not device.advance(0.10).toggle_pause
    device.apply(cmd)
    assert np.array_equal(harness.commands["left"], np.zeros(3))

    samples["left"] = _sample(module, 0.2, squeeze=0.0, trigger=1.0)
    cmd = device.advance(0.12)
    device.apply(cmd)
    assert harness.jaws["left"] < 0.04
    assert np.array_equal(harness.commands["left"], np.zeros(3))

    samples["left"] = _sample(module, 0.2, squeeze=1.0)
    device.advance(0.15)  # Clutch acquisition is motionless.
    samples["left"] = _sample(module, 0.21, squeeze=1.0)
    cmd = device.advance(0.20)
    device.apply(cmd)
    assert harness.commands["left"][0] > 0

    samples["right"] = _sample(module, 0.0, squeeze=0.0, button_1=True)
    assert device.advance(0.25).mark_success
    assert not device.advance(0.30).mark_success


def test_right_b_can_finish_practice_without_operator_success(monkeypatch):
    monkeypatch.setenv("AIRSEAI_XR_B_FINISH_ONLY", "1")
    module, _ = _module(monkeypatch)
    device, _ = _adapter(module)
    samples = {side: _sample(module, 0.0, squeeze=0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    assert "right B finishes without marking success" in device.help_text()
    device.advance(0.0)  # Both buttons were released, so edges are armed.

    device._dev.callbacks["SPACE"]()
    assert device.advance(0.05).toggle_pause
    samples["right"] = _sample(module, 0.0, squeeze=0.0, button_1=True)
    cmd = device.advance(0.10)
    assert cmd.tracking_valid and cmd.finish and not cmd.mark_success
    assert not device.advance(0.15).finish  # Holding B does not finish again.

    samples["right"] = _sample(module, 0.0, squeeze=0.0)
    device.advance(0.20)
    samples["right"] = _sample(module, 0.0, squeeze=0.0, button_1=True)
    assert device.advance(0.25).finish  # Release and press creates a new edge.

    samples["right"] = _sample(module, 0.0, squeeze=0.0)
    device._dev.callbacks["ENTER"]()  # Keyboard success remains explicit.
    cmd = device.advance(0.30)
    assert cmd.mark_success and not cmd.finish


def test_paused_trigger_does_not_accumulate_unapplied_jaw_motion(monkeypatch):
    module, _ = _module(monkeypatch)
    device, _ = _adapter(module)
    samples = {side: _sample(module, 0.0, squeeze=0.0, trigger=1.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    for t in (0.0, 0.05, 0.10):
        device.advance(t)
    assert device._jaws == {"right": 0.04, "left": 0.04}
    device._dev.callbacks["SPACE"]()
    device.advance(0.15)
    assert all(0.039 <= jaw < 0.04 for jaw in device._jaws.values())


def test_valid_palm_keeps_selected_arm_behavior(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {"right": _sample(module, 0.0, source="palm")}
    device._xr.advance = lambda: samples
    device.advance(0.0)
    samples = {"right": _sample(module, 0.02, source="palm")}
    cmd = device.advance(0.05)
    assert cmd.tracking_valid and cmd.side == "right"
    device.apply(cmd)
    assert harness.commands["right"][0] > 0
    assert harness.ik_options[-1]["max_joint_speed_deg_s"] == 30.0
    assert np.array_equal(harness.commands["left"], np.zeros(3))
    old_target = device._cmd_pos.copy()
    samples = {}
    assert not device.advance(0.10).tracking_valid
    samples = {"right": _sample(module, 0.03, source="palm")}
    device.advance(0.15)
    assert np.array_equal(device._cmd_pos, old_target)


def test_palm_stationary_pose_holds_joints_and_jaw_then_switches_arm(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {"right": _sample(module, 0.0, source="palm")}
    device._xr.advance = lambda: samples
    device.apply(device.advance(0.0))
    for frame in range(1, 12):
        device.apply(device.advance(frame * 0.05))
    assert harness.ik_options == []
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.575))
    device._dev.callbacks["K"]()
    device.apply(device.advance(0.60))
    assert harness.jaws["right"] < 0.04
    assert harness.ik_options == []

    device._dev.callbacks["TAB"]()
    samples = {"left": _sample(module, 0.0, source="palm")}
    device.apply(device.advance(0.65))
    assert device.side == harness.selected == "left"
    assert harness.ik_options == []
    samples = {"left": _sample(module, 0.001, source="palm")}
    device.apply(device.advance(0.70))
    assert harness.commands["left"][0] > 0


def test_palm_moving_target_uses_current_tcp_without_pose_debt(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {"right": _sample(module, 0.0, source="palm")}
    device._xr.advance = lambda: samples
    device.apply(device.advance(0.0))
    samples = {"right": _sample(module, 0.02, source="palm")}
    device.apply(device.advance(0.05))
    assert np.isclose(harness.commands["right"][0], 0.004)
    harness.commands["right"] = np.zeros(3)
    samples = {"right": _sample(module, 0.04, source="palm")}
    device.apply(device.advance(0.10))
    assert np.isclose(harness.commands["right"][0], 0.004)


def test_palm_deadband_accumulates_slow_motion_and_rebases_after_jump(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {"right": _sample(module, 0.0, source="palm")}
    device._xr.advance = lambda: samples
    device.apply(device.advance(0.0))
    for tick, x in ((0.05, 0.0002), (0.10, 0.0004)):
        samples = {"right": _sample(module, x, source="palm")}
        device.apply(device.advance(tick))
    assert harness.ik_options == []
    samples = {"right": _sample(module, 0.0006, source="palm")}
    device.apply(device.advance(0.15))
    assert np.isclose(harness.commands["right"][0], 0.0006)

    samples = {"right": _sample(module, 0.301, source="palm")}
    device.apply(device.advance(0.20))
    assert len(harness.ik_options) == 1
    samples = {"right": _sample(module, 0.3018, source="palm")}
    device.apply(device.advance(0.25))
    assert len(harness.ik_options) == 2
    assert np.isclose(harness.commands["right"][0], 0.0014)


def test_palm_latches_each_step_and_rejection_preserves_held_command(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {"right": _sample(module, 0.0, source="palm")}
    device._xr.advance = lambda: samples
    device.advance(0.0)

    references = []
    original_step = harness.step_ik_toward

    def track_step(*args, **kwargs):
        references.append(harness._hold[0, :3].copy())
        return original_step(*args, **kwargs)

    latches = []
    original_latch = harness.latch

    def track_latch():
        original_latch()
        latches.append(harness._hold[0, :3].copy())

    harness.step_ik_toward = track_step
    harness.latch = track_latch
    for tick, x in ((0.05, 0.02), (0.10, 0.04)):
        samples = {"right": _sample(module, x, source="palm")}
        cmd = device.advance(tick)
        assert cmd.tracking_valid
        device.apply(cmd)

    assert len(references) == len(latches) == 2
    np.testing.assert_allclose(references[0], np.zeros(3))
    np.testing.assert_allclose(references[1], latches[0])
    np.testing.assert_allclose(latches[1], harness.commands["right"])
    assert latches[1][0] > latches[0][0] > 0

    # Harness rejects before writing a new target; latching must retain the
    # previous command while the palm pose target is re-baselined.
    harness.step_ik_toward = lambda *args, **kwargs: False
    samples = {"right": _sample(module, 0.06, source="palm")}
    device.apply(device.advance(0.15))
    assert len(latches) == 3
    np.testing.assert_allclose(latches[-1], latches[-2])
    np.testing.assert_allclose(harness._hold[0, :3], harness.commands["right"])
    assert device.diagnostics_snapshot()["safety_rejected"] == ["right"]
    assert device._cmd_pos is device._cmd_quat is None


def test_rejected_ik_holds_and_rebaselines_controller_targets(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {side: _sample(module, 0.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.0))
    device.apply(device.advance(0.05))
    before = {side: value.copy() for side, value in harness.commands.items()}
    harness.step_ik_toward = lambda *args, **kwargs: False
    samples = {side: _sample(module, 0.001) for side in ("right", "left")}
    device.apply(device.advance(0.10))
    assert all(np.array_equal(harness.commands[side], before[side]) for side in before)
    assert device.diagnostics_snapshot()["safety_rejected"] == ["right", "left"]
    assert device._arm_targets == device._controller_previous == {}


def test_palm_invalid_flags_are_not_reused(monkeypatch):
    module, _ = _module(monkeypatch)
    pose = types.SimpleNamespace(validity_flags=1, pose_matrix=object())
    assert module._tracked_pose({"palm": pose}, "palm") is None
    assert module._controller_pose({"grip": pose}) == (None, "")
    assert module._controller_pose({"": pose}) == (None, "")


def test_paused_palm_gripper_stays_frozen_until_resume(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    device._xr.advance = lambda: {"right": _sample(module, 0.0, source="palm")}
    device.apply(device.advance(0.0))
    device._dev.callbacks["K"]()
    for frame in range(1, 11):
        device.apply(device.advance(frame * 0.05))
    assert harness.jaws["right"] == 0.04
    assert harness.ik_options == []
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.55))
    assert 0.0 < harness.jaws["right"] < 0.04
    assert harness.ik_options == []


def test_rejected_controller_ik_restores_held_gripper_command(monkeypatch):
    module, _ = _module(monkeypatch)
    device, harness = _adapter(module)
    samples = {side: _sample(module, 0.0, trigger=1.0) for side in ("right", "left")}
    device._xr.advance = lambda: samples
    device._dev.callbacks["SPACE"]()
    device.apply(device.advance(0.0))
    device.apply(device.advance(0.05))
    held_jaw = harness.jaws["right"]
    harness.step_ik_toward = lambda *args, **kwargs: False
    samples["right"] = _sample(module, 0.001, trigger=1.0)
    device.apply(device.advance(0.10))
    assert device.diagnostics_snapshot()["safety_rejected"] == ["right"]
    assert device._jaws["right"] == harness.jaws["right"] == held_jaw


def test_controller_reader_rejects_unflagged_default_pose(monkeypatch):
    module, xr_core = _module(monkeypatch)

    class Input:
        def get_all_virtual_world_poses(self):
            return {}

        def is_active(self):
            return True

        def get_virtual_world_pose(self, *_args):
            pytest.fail("Unflagged controller poses must not bypass tracking validity")

    xr_core.current = types.SimpleNamespace(get_input_device=lambda _path: Input())
    assert module.TrackedHands(object()).advance() == {}


def test_opt_in_diagnostic_survives_late_xr_core(monkeypatch, capsys):
    monkeypatch.setenv("AIRSEAI_XR_DIAG", "1")
    module, xr_core = _module(monkeypatch)
    reader = module.TrackedHands(object())
    assert reader.advance() == {}
    assert "right:core_none" in capsys.readouterr().out
    assert reader.advance() == {}
    assert capsys.readouterr().out == ""  # Repeated missing input is throttled.
    xr_core.current = types.SimpleNamespace(get_input_device=lambda _path: None)
    assert reader.advance() == {}
    assert "left:device_none" in capsys.readouterr().out

