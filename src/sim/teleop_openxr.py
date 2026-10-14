"""OpenXR palm and bimanual controller input with bounded joint commands."""

import os
import time
from dataclasses import dataclass

import numpy as np
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from omni.kit.xr.core import XRCore, XRPoseValidityFlags

from sim.teleop import Command, KeyboardDevice, rotate_quaternion

_SIDES = ("right", "left")
_REQUIRED_POSE_FLAGS = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
_MAX_POSE_STEP_M = 0.25  # Re-localization is not a hand movement.
_MAX_POSE_STEP_RAD = np.pi / 2
_POSE_DEADBAND_M = 0.0005
_POSE_DEADBAND_RAD = np.deg2rad(0.5)
_XR_DIAG_ENABLED = os.environ.get("AIRSEAI_XR_DIAG") == "1"
_XR_PHYSICS_DIAG_ENABLED = os.environ.get("AIRSEAI_XR_PHYSICS_DIAG") == "1"
_XR_B_FINISH_ONLY = os.environ.get("AIRSEAI_XR_B_FINISH_ONLY") == "1"


@dataclass
class XRHandSample:
    pose: np.ndarray  # world xyz + wxyz
    source: str  # palm or controller
    pose_name: str
    trigger: float | None = None
    squeeze: float | None = None
    button_0: bool = False  # left X / right A
    button_1: bool = False  # left Y / right B


def _pose_vector(matrix):
    if matrix is None:
        return None
    try:
        quat = matrix.ExtractRotationQuat()
        value = np.asarray([*matrix.ExtractTranslation(), quat.GetReal(), *quat.GetImaginary()], dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    norm = np.linalg.norm(value[3:])
    if value.shape != (7,) or not np.isfinite(value).all() or not 0.5 < norm < 1.5:
        return None
    value[3:] /= norm
    return value


def _tracked_pose(poses, name):
    pose = poses.get(name)
    if pose is None:
        return None
    try:
        if pose.validity_flags & _REQUIRED_POSE_FLAGS != _REQUIRED_POSE_FLAGS:
            return None
        return _pose_vector(pose.pose_matrix)
    except (AttributeError, TypeError, ValueError):
        return None


def _gesture(device, name, kind):
    try:
        if not device.has_input_gesture(name, kind):
            return None
        value = float(device.get_input_gesture_value(name, kind))
    except Exception:  # A disconnected XR runtime must fail closed.
        return None
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else None


def _axis_input(device, name):
    value = _gesture(device, name, "value")
    return value if value is not None else _gesture(device, name, "click")


def _controller_pose(poses):
    """Accept only poses carrying position and orientation validity flags."""
    names = ("grip", "aim") if any(name in poses for name in ("grip", "aim")) else ("",)
    for name in names:
        value = _tracked_pose(poses, name)
        if value is not None:
            return value, name or "default"
    return None, ""


class TrackedHands(OpenXRDevice):
    """Read current XR inputs without Isaac Lab's stale-joint-pose substitution."""

    def _emit_diag(self, core, samples):
        """Report XR input availability without changing the fail-closed read path."""
        if not _XR_DIAG_ENABLED:
            return
        now = time.monotonic()
        signature = (core is None,) + tuple(
            (
                side,
                None if samples.get(side) is None else samples[side].source,
                False if samples.get(side) is None else samples[side].button_0,
                False if samples.get(side) is None else samples[side].button_1,
                False
                if samples.get(side) is None or samples[side].squeeze is None
                else samples[side].squeeze > 0.5,
            )
            for side in _SIDES
        )
        if signature == getattr(self, "_diag_signature", None) and now - getattr(self, "_diag_time", 0) < 2.0:
            return
        self._diag_signature = signature
        self._diag_time = now
        details = []
        for side in _SIDES:
            sample = samples.get(side)
            if core is None:
                details.append(f"{side}:core_none")
                continue
            try:
                device = core.get_input_device(f"/user/hand/{side}")
            except Exception as error:
                details.append(f"{side}:device_error={type(error).__name__}")
                continue
            if device is None:
                details.append(f"{side}:device_none")
                continue
            active_api = "present"
            try:
                active = bool(device.is_active())
            except AttributeError:
                active = True
                active_api = "missing"
            except Exception as error:
                details.append(f"{side}:active_error={type(error).__name__}")
                continue
            try:
                poses = device.get_all_virtual_world_poses() or {}
                pose_names = [str(name) for name in poses]
                pose_flags = {
                    name: str(poses[name].validity_flags)
                    for name in ("palm", "grip", "aim", "")
                    if name in poses
                }
                pose, pose_name = _controller_pose(poses)
            except Exception as error:
                pose_names, pose_flags, pose, pose_name = [f"error:{type(error).__name__}"], {}, None, ""
            squeeze = _axis_input(device, "squeeze")
            trigger = _axis_input(device, "trigger")
            x_or_a = _gesture(device, "x" if side == "left" else "a", "click")
            y_or_b = _gesture(device, "y" if side == "left" else "b", "click")
            try:
                input_names = [str(name) for name in (device.get_input_names() or ())]
            except Exception as error:
                input_names = [f"error:{type(error).__name__}"]
            gestures = {}
            for name in ("squeeze", "trigger", "x", "y", "a", "b"):
                if name in input_names:
                    try:
                        gestures[name] = [str(kind) for kind in device.get_input_gesture_names(name)]
                    except Exception as error:
                        gestures[name] = [f"error:{type(error).__name__}"]
            reason = (
                "inactive"
                if not active
                else "palm"
                if sample is not None and sample.source == "palm"
                else "pose_unavailable"
                if pose is None
                else "squeeze_unavailable"
                if squeeze is None
                else "controller_ready"
                if sample is not None
                else "sample_missing"
            )
            details.append(
                f"{side}:{reason} active={int(active)} active_api={active_api} pose={pose_name or '-'} "
                f"xyz={np.round(pose[:3], 3).tolist() if pose is not None else '-'} "
                f"poses={pose_names[:8]} flags={pose_flags} inputs={input_names[:16]} gestures={gestures} "
                f"squeeze={squeeze} trigger={trigger} "
                f"{'x' if side == 'left' else 'a'}={x_or_a} "
                f"{'y' if side == 'left' else 'b'}={y_or_b}"
            )
        print("[AIRSEAI_XR_DIAG] " + " | ".join(details), flush=True)

    def _get_raw_data(self):
        result = {}
        core = XRCore.get_singleton()
        if core is None:
            self._emit_diag(None, result)
            return result
        for side in _SIDES:
            try:
                device = core.get_input_device(f"/user/hand/{side}")
            except Exception:
                continue
            if device is None:
                continue
            try:
                poses = device.get_all_virtual_world_poses() or {}
            except Exception:
                poses = {}
            palm = _tracked_pose(poses, "palm")
            if palm is not None:
                result[side] = XRHandSample(palm, "palm", "palm")
                continue
            try:
                active = bool(device.is_active())
            except AttributeError:
                # Pose validity remains required when activity is unavailable.
                active = True
            except Exception:
                active = False
            if not active:
                continue
            pose, pose_name = _controller_pose(poses)
            squeeze = _axis_input(device, "squeeze")
            if pose is None or squeeze is None:
                continue
            result[side] = XRHandSample(
                pose=pose,
                source="controller",
                pose_name=pose_name,
                trigger=_axis_input(device, "trigger"),
                squeeze=squeeze,
                button_0=bool((_gesture(device, "x" if side == "left" else "a", "click") or 0) > 0.5),
                button_1=bool((_gesture(device, "y" if side == "left" else "b", "click") or 0) > 0.5),
            )
        self._emit_diag(core, result)
        return result


def rotation_delta(current, previous):
    w1, v1 = current[0], current[1:]
    w0, v0 = previous[0], -previous[1:]
    delta = np.r_[w1 * w0 - np.dot(v1, v0), w1 * v0 + w0 * v1 + np.cross(v1, v0)]
    delta /= np.linalg.norm(delta)
    if delta[0] < 0:
        delta = -delta
    sine = np.linalg.norm(delta[1:])
    return np.zeros(3) if sine < 1e-9 else delta[1:] * (2 * np.arctan2(sine, delta[0]) / sine)


def _bounded_delta(delta, max_step):
    norm = np.linalg.norm(delta)
    return delta * min(1.0, max_step / max(norm, 1e-12))


def _pose_delta(current, previous):
    return current[:3] - previous[:3], rotation_delta(current[3:], previous[3:])


def _pose_moved(delta_pos, delta_rot):
    return np.linalg.norm(delta_pos) >= _POSE_DEADBAND_M or np.linalg.norm(delta_rot) >= _POSE_DEADBAND_RAD


class OpenXRHandDevice(KeyboardDevice):
    def __init__(self, task, h, *, vr_joint_speed_deg_s=30.0, vr_position_gain=1.0, **speeds):
        super().__init__(task, h, **speeds)
        if not np.isfinite(vr_joint_speed_deg_s) or vr_joint_speed_deg_s <= 0:
            raise ValueError("VR joint speed must be finite and positive")
        try:
            position_gain = float(vr_position_gain)
        except (TypeError, ValueError) as error:
            raise ValueError("VR position gain must be finite and positive") from error
        if not np.isfinite(position_gain) or position_gain <= 0:
            raise ValueError("VR position gain must be finite and positive")
        try:
            opening = float(self.knobs.get("open_width", 0.04))
        except (TypeError, ValueError) as error:
            raise ValueError("VR open_width must be finite and positive") from error
        if not np.isfinite(opening) or opening <= 0:
            raise ValueError("VR open_width must be finite and positive")
        try:
            closing = float(self.knobs.get("close_width", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError("VR close_width must be finite and non-negative") from error
        if not np.isfinite(closing) or closing < 0:
            raise ValueError("VR close_width must be finite and non-negative")
        self.vr_joint_speed_deg_s = float(vr_joint_speed_deg_s)
        self.vr_position_gain = position_gain
        self._controller_open_width = opening
        self._controller_close_width = min(closing, opening)
        self._xr = TrackedHands(OpenXRDeviceCfg())
        self._previous = {}  # Last palm pose consumed as motion.
        self._palm_sample_previous = {}  # Adjacent sample for tracking-jump detection.
        self._controller_previous = {}  # Last controller pose consumed as motion.
        self._controller_sample_previous = {}
        self._arm_targets = {}
        self._pending = {}
        self._mode = None
        self._button_previous = {}
        self._buttons_armed = False
        self._recording_active = False
        self._xr_samples = {}
        self._last_tracking_valid = False
        self._safety_rejected = []
        self._episode_wall_start = time.monotonic()
        self._input_wall_time = self._episode_wall_start

    def begin_episode(self, ctx):
        super().begin_episode(ctx)
        self._xr.reset()
        self._previous.clear()
        self._palm_sample_previous.clear()
        self._controller_previous.clear()
        self._controller_sample_previous.clear()
        self._arm_targets.clear()
        self._pending.clear()
        self._mode = None
        self._button_previous.clear()
        self._buttons_armed = False
        self._recording_active = False
        self._xr_samples = {}
        self._last_tracking_valid = False
        self._safety_rejected = []
        self._episode_wall_start = time.monotonic()
        self._input_wall_time = self._episode_wall_start

    def _ik_options(self):
        return dict(
            max_speed=self.pos_speed,
            max_ang_speed=float(np.degrees(self.rot_speed)),
            max_joint_speed_deg_s=self.vr_joint_speed_deg_s,
        )

    def diagnostics_snapshot(self):
        snapshot = dict(
            mode=self._mode,
            tracking_valid=self._last_tracking_valid,
            recording_active=self._recording_active,
            right_b_action="finish" if _XR_B_FINISH_ONLY else "mark_success",
            joint_speed_deg_s=self.vr_joint_speed_deg_s,
            position_gain=self.vr_position_gain,
            jaw_open_width_m=self._controller_open_width,
            jaw_close_width_m=self._controller_close_width,
            elapsed_wall_s=self._input_wall_time - self._episode_wall_start,
            safety_rejected=list(self._safety_rejected),
            hands={
                side: dict(
                    source=sample.source,
                    pose_name=sample.pose_name,
                    pose_xyz_wxyz=sample.pose.tolist(),
                    squeeze=sample.squeeze,
                    trigger=sample.trigger,
                    button_0=bool(sample.button_0),
                    button_1=bool(sample.button_1),
                )
                for side, sample in self._xr_samples.items()
            },
        )
        if _XR_PHYSICS_DIAG_ENABLED:
            snapshot["physics"] = self._physics_snapshot()
        return snapshot

    def _physics_snapshot(self):
        """Debug snapshots only; implicit-actuator torques are SDK estimates."""
        try:
            data = self.h.robot.data
            result = {}
            for side in _SIDES:
                chain = self.h._chains[side]
                joints = chain["arm"].joint_ids

                def values(name, *, degrees=False, joints=joints):
                    value = getattr(data, name, None)
                    if value is None:
                        return None
                    row = value[0, joints].detach().cpu().numpy()
                    return (np.degrees(row) if degrees else row).tolist()

                result[side] = dict(
                    joint_actual_deg=values("joint_pos", degrees=True),
                    joint_target_deg=values("joint_pos_target", degrees=True),
                    joint_velocity_deg_s=values("joint_vel", degrees=True),
                    estimated_computed_torque_nm=values("computed_torque"),
                    estimated_applied_torque_nm=values("applied_torque"),
                    effort_limit_nm=values("joint_effort_limits"),
                    tcp_xyz_wxyz=data.body_state_w[0, chain["ee_body"], :7].detach().cpu().tolist(),
                )
            try:
                result["scene"] = dict(
                    contact_areas_mm2=self.h.pad_areas(),
                    objects={
                        name: self.h.scene[name].data.root_state_w[0].detach().cpu().tolist()
                        for name in ("cube_a", "cube_b")
                        if name in self.h.scene.keys()
                    },
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, IndexError) as error:
                result["scene"] = {"error": f"{type(error).__name__}: {error}"}
            return result
        except (AttributeError, KeyError, RuntimeError, TypeError, IndexError) as error:
            return {"error": f"{type(error).__name__}: {error}"}

    def help_text(self):
        right_b = "finishes without marking success" if _XR_B_FINISH_ONLY else "marks success/saves"
        return (
            "VR palms: move selected hand; TAB selects arm, K controls grip.\n"
            "VR controllers: both must track; hold each squeeze to move its arm; "
            "full trigger closes its jaw to the task's close width.\n"
            f"Left X starts/pauses, right B {right_b}, left Y discards. "
            "Right A remains Isaac Lab's XR anchor control.\n"
            "Keyboard SPACE starts/pauses, ENTER saves, BACKSPACE discards, R resets.\n"
            "Tracking loss freezes simulation; re-acquisition re-baselines poses."
        )

    def _switch_mode(self, mode):
        if mode == self._mode:
            return
        previous_mode = self._mode
        self._mode = mode
        self._previous.clear()
        self._palm_sample_previous.clear()
        self._controller_previous.clear()
        self._controller_sample_previous.clear()
        self._arm_targets.clear()
        self._pending.clear()
        self._button_previous.clear()
        self._buttons_armed = False
        if previous_mode == "controller" or mode == "controller":
            self._cmd_pos = None
            self._cmd_quat = None

    def _controller_buttons(self, samples, cmd):
        now = {
            (side, index): getattr(samples[side], f"button_{index}") for side in _SIDES for index in (0, 1)
        }
        if self._buttons_armed:

            def edge(side, index):
                return now[side, index] and not self._button_previous.get((side, index), False)

            cmd.toggle_pause |= edge("left", 0)
            right_b = edge("right", 1)
            if _XR_B_FINISH_ONLY:
                cmd.finish |= right_b
            else:
                cmd.mark_success |= right_b
            cmd.discard |= edge("left", 1)
        elif not any(now.values()):
            self._buttons_armed = True
        self._button_previous = now

    def advance(self, t_ep):
        jaws_before = dict(self._jaws)
        cmd = super().advance(t_ep)
        cmd.delta_pos[:] = 0
        cmd.delta_rot[:] = 0
        samples = self._xr.advance()
        self._input_wall_time = time.monotonic()
        self._xr_samples = samples
        self._last_tracking_valid = False
        selected = samples.get(cmd.side)
        if selected is not None and selected.source == "palm":
            self._switch_mode("palm")
            self._last_tracking_valid = True
            if cmd.toggle_pause:
                self._recording_active = not self._recording_active
            if not self._recording_active:
                self._jaws.update(jaws_before)
                cmd.jaw = self._jaws[cmd.side]
            pose = selected.pose
            previous = self._previous.get(cmd.side)
            previous_sample = self._palm_sample_previous.get(cmd.side)
            self._palm_sample_previous = {cmd.side: pose.copy()}
            if previous is None or previous_sample is None or cmd.toggle_pause or cmd.side != self.side:
                self._previous = {cmd.side: pose.copy()}
                return cmd
            sample_pos, sample_rot = _pose_delta(pose, previous_sample)
            if (
                np.linalg.norm(sample_pos) > _MAX_POSE_STEP_M
                or np.linalg.norm(sample_rot) > _MAX_POSE_STEP_RAD
            ):
                self._previous = {cmd.side: pose.copy()}
                return cmd
            raw_pos, raw_rot = _pose_delta(pose, previous)
            if not _pose_moved(raw_pos, raw_rot):
                return cmd
            self._previous = {cmd.side: pose.copy()}
            cmd.delta_pos[:] = _bounded_delta(
                raw_pos * self.vr_position_gain, self.pos_speed * self.h.control_dt
            )
            cmd.delta_rot[:] = _bounded_delta(raw_rot, self.rot_speed * self.h.control_dt)
            return cmd

        if not all(samples.get(side) is not None and samples[side].source == "controller" for side in _SIDES):
            self._switch_mode(None)
            self._jaws.update(jaws_before)
            if cmd.toggle_pause:
                self._recording_active = not self._recording_active
            cmd.tracking_valid = False
            return cmd

        self._switch_mode("controller")
        self._last_tracking_valid = True
        self._controller_buttons(samples, cmd)
        if cmd.toggle_pause:
            self._recording_active = not self._recording_active
            self._controller_previous.clear()
            self._controller_sample_previous.clear()
            self._arm_targets.clear()
        if _XR_DIAG_ENABLED and (cmd.toggle_pause or cmd.mark_success or cmd.finish or cmd.discard):
            print(
                "[AIRSEAI_XR_DIAG] command "
                f"toggle_pause={cmd.toggle_pause} mark_success={cmd.mark_success} "
                f"finish={cmd.finish} discard={cmd.discard} "
                f"recording_active={self._recording_active}",
                flush=True,
            )
        # Advance each controller jaw once per control step.
        self._jaws.update(jaws_before)
        if not self._recording_active:
            self._pending.clear()
            return cmd
        jaw_step = self.jaw_speed * self.h.control_dt
        for side in _SIDES:
            sample = samples[side]
            if sample.trigger is not None:
                self._jaw_targets[side] = (
                    self._controller_open_width
                    + (self._controller_close_width - self._controller_open_width) * sample.trigger
                )
            jaw_delta = self._jaw_targets[side] - self._jaws[side]
            self._jaws[side] += float(np.clip(jaw_delta, -jaw_step, jaw_step))
            delta_pos = np.zeros(3)
            delta_rot = np.zeros(3)
            engaged = sample.squeeze > 0.5 and not cmd.toggle_pause
            if engaged:
                previous = self._controller_previous.get(side)
                previous_sample = self._controller_sample_previous.get(side)
                self._controller_sample_previous[side] = sample
                if (
                    previous is None
                    or previous_sample is None
                    or previous_sample.pose_name != sample.pose_name
                ):
                    self._controller_previous[side] = sample
                else:
                    sample_pos, sample_rot = _pose_delta(sample.pose, previous_sample.pose)
                    if (
                        np.linalg.norm(sample_pos) > _MAX_POSE_STEP_M
                        or np.linalg.norm(sample_rot) > _MAX_POSE_STEP_RAD
                    ):
                        self._controller_previous[side] = sample
                    else:
                        raw_pos, raw_rot = _pose_delta(sample.pose, previous.pose)
                        if _pose_moved(raw_pos, raw_rot):
                            delta_pos = _bounded_delta(
                                raw_pos * self.vr_position_gain, self.pos_speed * self.h.control_dt
                            )
                            delta_rot = _bounded_delta(raw_rot, self.rot_speed * self.h.control_dt)
                            self._controller_previous[side] = sample
            else:
                self._controller_previous.pop(side, None)
                self._controller_sample_previous.pop(side, None)
                self._arm_targets.pop(side, None)
            self._pending[side] = (delta_pos, delta_rot, self._jaws[side], engaged)
        cmd.jaw = self._jaws[cmd.side]
        return cmd

    def apply(self, cmd: Command):
        self._safety_rejected = []
        if self._mode != "controller":
            if not np.any(cmd.delta_pos) and not np.any(cmd.delta_rot):
                if cmd.side != self.side:
                    self.h.latch()
                    self.side = cmd.side
                    self.h.select_arm(self.side)
                    self.h.reset_ik()
                    self._cmd_pos = self._cmd_quat = None
                self.h.hold_grip(cmd.side, cmd.jaw)
                self.h.robot.set_joint_position_target(self.h._hold)
                self.h.robot.write_data_to_sim()
                self.h.latch()
                return
            # Anchor commands at the measured TCP to prevent target windup.
            if cmd.side == self.side:
                self._cmd_pos = np.asarray(self.h.tcp_pos(), dtype=float).copy()
                self._cmd_quat = np.asarray(self.h.tcp_quat(), dtype=float).copy()
            accepted = super().apply(cmd)
            if accepted is False:
                self._safety_rejected.append(cmd.side)
                self._cmd_pos = self._cmd_quat = None
                self._jaws[cmd.side] = float(self.h._hold[0, self.h._chains[cmd.side]["grip_ids"][0]])
            self.h.latch()
            return
        # Preserve the first arm's command before solving the second arm.
        for side in _SIDES:
            delta_pos, delta_rot, jaw, engaged = self._pending[side]
            if not engaged or (not np.any(delta_pos) and not np.any(delta_rot)):
                self._arm_targets.pop(side, None)
                self.h.hold_grip(side, jaw)
                continue
            self.h.select_arm(side)
            if side not in self._arm_targets:
                self.h.reset_ik()
            pos = np.asarray(self.h.tcp_pos(), dtype=float).copy() + delta_pos
            quat = rotate_quaternion(np.asarray(self.h.tcp_quat(), dtype=float).copy(), delta_rot)
            self._arm_targets[side] = pos, quat
            accepted = self.h.step_ik_toward(pos.tolist(), jaw, quat.tolist(), **self._ik_options())
            if accepted is False:
                self._safety_rejected.append(side)
                self._arm_targets.pop(side, None)
                self._controller_previous.pop(side, None)
                self._jaws[side] = float(self.h._hold[0, self.h._chains[side]["grip_ids"][0]])
            else:
                self.h.latch()
        # Commit jaw changes even when neither arm needed IK.
        self.h.robot.set_joint_position_target(self.h._hold)
        self.h.robot.write_data_to_sim()
        self.side = cmd.side
        self.h.select_arm(self.side)
