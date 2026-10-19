"""Isaac Lab OpenXR palm or bimanual controller teleoperation.

Palm tracking retains the original keyboard-driven, selected-arm behavior.
When palms are unavailable, both active motion controllers are required. Their
grip poses drive the two simulated arms, squeeze clutches motion, and trigger
controls the corresponding jaw. No physical-robot interface is used here.
"""

from dataclasses import dataclass
import os
import sys
import time

import numpy as np
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from omni.kit.xr.core import XRCore, XRPoseValidityFlags

from sim.teleop import Command, KeyboardDevice, rotate_quaternion
from sim.vr_pose_solver import make_harness_compute


_SIDES = ("right", "left")
_REQUIRED_POSE_FLAGS = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
_MAX_POSE_STEP_M = 0.25  # Re-localization is not a hand movement.
_MAX_POSE_STEP_RAD = np.pi / 2
_POSE_DEADBAND_M = 0.0005
_POSE_DEADBAND_RAD = np.deg2rad(0.5)
_TARGET_REACHED_M = 0.0001
_TARGET_REACHED_RAD = np.deg2rad(0.05)
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
        value = np.asarray(
            [*matrix.ExtractTranslation(), quat.GetReal(), *quat.GetImaginary()], dtype=float
        )
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


def _controller_pose(device, poses):
    """Prefer validity-bearing poses; the active device's named pose is a fallback."""
    reported = any(name in poses for name in ("grip", "aim"))
    for name in ("grip", "aim"):
        if name in poses:
            value = _tracked_pose(poses, name)
            if value is not None:
                return value, name
    # A reported but invalid grip/aim cannot be replaced by a stale matrix.
    if reported:
        return None, ""
    if "" in poses:
        return _tracked_pose(poses, ""), "default"
    for name in ("grip", "aim"):
        try:
            if device.has_pose(name):
                value = _pose_vector(device.get_virtual_world_pose(name))
                if value is not None:
                    return value, name
        except Exception:  # SDK versions differ in their named-pose support.
            pass
    # Isaac Lab 2.3.2 queries the default controller pose with no name.
    try:
        value = _pose_vector(device.get_virtual_world_pose())
    except Exception:
        value = None
    return value, "default"


class TrackedHands(OpenXRDevice):
    """Read current XR inputs without Isaac Lab's stale-joint-pose substitution."""

    def _request_table_view(self, core, samples, placement=None, placement_hash=None):
        """Calibrate the initial view from valid head tracking alone, once."""
        try:
            head = core.get_input_device("/user/head")
            desc = head.get_pose_desc("") if head is not None else None
            source_head = _tracked_pose({"": desc}, "")
        except Exception:
            return False
        if source_head is None:
            return False
        import carb
        from pxr import Gf

        from sim.vr_view import (
            TABLE_FRONT_EYE, TABLE_FRONT_TARGET,
            TABLE_OVERVIEW_EYE, TABLE_OVERVIEW_TARGET,
            fixed_horizontal_controller_frame, placed_table_front_camera_pose,
            table_front_camera_pose, table_overview_camera_pose,
        )

        profile = self._view_profile
        if profile == "table_front":
            pose = (table_front_camera_pose(Gf) if placement is None else placed_table_front_camera_pose(
                Gf, placement["world_down_m"], placement["world_yaw_deg"],
                placement.get("world_toward_viewer_m", 0.),
            ))
            eye, target = TABLE_FRONT_EYE, TABLE_FRONT_TARGET
            if placement is not None:
                distance = np.linalg.norm(np.asarray(eye) - target)
                eye = tuple(pose.ExtractTranslation())
                target = tuple(np.asarray(eye) + distance * np.asarray(pose.TransformDir(Gf.Vec3d(0., 0., -1.))))
        else:
            pose = table_overview_camera_pose(Gf)
            eye, target = TABLE_OVERVIEW_EYE, TABLE_OVERVIEW_TARGET
        frame = fixed_horizontal_controller_frame(core, desc.pose_matrix, pose, Gf)
        if _pose_vector(pose) is None or _pose_vector(frame) is None:
            return False
        settings_path = core.get_profile("ar").get_persistent_path()
        settings = carb.settings.get_settings()
        settings.set_bool(settings_path + "adjustForUserHeight", False)
        settings.set_bool(settings_path + "overrideFloorAlignedAnchor", True)
        # This is SDK107's initial-view operation, not a repeated head/camera
        # anchor update. Never read the XR output camera back into this frame.
        core.schedule_set_camera(pose)
        self._horizontal_controller_frame = frame
        self._head_view_pending = False
        print("[AIRSEAI_XR_VIEW_REQUEST]", dict(
            profile=profile, method="schedule_set_camera",
            camera=f"explicit_stage_z_up_{profile}", eye=eye,
            target=target, up=(0.0, 0.0, 1.0),
            source_head="/user/head", head_validity_flags=str(desc.validity_flags),
            source_head_xyz_wxyz=source_head.tolist(), first_valid_head=placement is None,
            placement_request=placement, placement_sha256=placement_hash,
            both_hand_samples_valid=all(side in samples for side in _SIDES),
            input_frame="fixed_horizontal_physical", head_tracking="enabled",
            anchor_mode=settings.get(settings_path + "anchorMode"),
            overrideFloorAlignedAnchor=settings.get(settings_path + "overrideFloorAlignedAnchor"),
        ), flush=True)
        return True

    def _placement_diag(self, state, **details):
        signature = (state, repr(details))
        if signature != getattr(self, "_placement_diag_signature", None):
            self._placement_diag_signature = signature
            print("[AIRSEAI_XR_VIEW_PLACEMENT]", dict(state=state, **details), flush=True)

    def _placement_controllers(self, core, samples):
        """Confirmation requires fresh validity-bearing physical controller poses."""
        for side in _SIDES:
            sample = samples.get(side)
            if sample is None or sample.source != "controller" or sample.squeeze is None:
                return False
            try:
                device = core.get_input_device(f"/user/hand/{side}")
                name = "" if sample.pose_name == "default" else sample.pose_name
                if _tracked_pose({name: device.get_pose_desc(name)}, name) is None:
                    return False
            except Exception:
                return False
        return True

    def _process_view_placement(self, core, samples):
        """Return True for consumed frames; never change placement while recording."""
        from sim.vr_view import load_view_placement

        self._view_placement_consuming = False
        self._view_placement_hold_motion = False
        ready = self._placement_controllers(core, samples)
        released = ready and all(samples[side].squeeze <= .5 for side in _SIDES)
        if getattr(self, "_placement_release_required", False):
            if released:
                self._placement_release_required = False
                self._placement_diag("RELEASED")
            self._view_placement_consuming = True
            return True
        try:
            request, digest = load_view_placement(self._view_placement_file)
        except Exception as error:
            self._placement_pending = None
            self._placement_gesture_armed = False
            self._placement_diag("REJECTED", reason=str(error))
            if self._head_view_pending and not getattr(self, "_view_recenter_allowed", False):
                self._view_placement_hold_motion = True
                return False
            self._view_placement_consuming = bool(self._head_view_pending)
            return self._view_placement_consuming
        if (request["request_id"] in getattr(self, "_placement_applied_ids", set())
                or digest in getattr(self, "_placement_applied_hashes", set())):
            self._placement_pending = None
            self._placement_gesture_armed = False
            return False
        if getattr(self, "_placement_pending", None) != (request, digest):
            self._placement_pending = (request, digest)
            self._placement_gesture_armed = False
            self._placement_diag("PENDING", request_id=request["request_id"], sha256=digest)
        if not getattr(self, "_view_recenter_allowed", False):
            self._placement_gesture_armed = False
            # An existing mapping remains fully usable so X can pause. If no
            # initial mapping exists, expose buttons only through the adapter.
            self._view_placement_hold_motion = bool(self._head_view_pending)
            return False
        self._view_placement_consuming = True
        if not ready:
            self._placement_gesture_armed = False
            return True
        if released:
            self._placement_gesture_armed = True
            self._placement_diag("ARMED", request_id=request["request_id"], sha256=digest)
            return True
        if (getattr(self, "_placement_gesture_armed", False)
                and all(samples[side].squeeze > .5 for side in _SIDES)):
            self._placement_gesture_armed = False
            try:
                applied = self._request_table_view(core, samples, request, digest)
            except Exception as error:
                self._placement_diag("REJECTED", reason=f"camera request: {type(error).__name__}")
                return True
            if applied:
                self._placement_applied_ids = getattr(self, "_placement_applied_ids", set()) | {request["request_id"]}
                self._placement_applied_hashes = getattr(self, "_placement_applied_hashes", set()) | {digest}
                self._placement_pending = None
                self._placement_release_required = True
                self._placement_diag("APPLIED_WAIT_RELEASE", request_id=request["request_id"], sha256=digest)
            else:
                self._placement_diag("WAIT_VALID_HEAD", request_id=request["request_id"], sha256=digest)
        return True

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
                # Kit 107's XR input device lacks is_active(); the checked
                # controller pose and squeeze gesture gate readiness below.
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
                pose, pose_name = _controller_pose(device, poses)
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
                "inactive" if not active else
                "palm" if sample is not None and sample.source == "palm" else
                "pose_unavailable" if pose is None else
                "squeeze_unavailable" if squeeze is None else
                "controller_ready" if sample is not None else "sample_missing"
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
                # Older Kit wrappers expose pose/gesture methods but no
                # is_active(); pose validity and squeeze are checked below.
                active = True
            except Exception:
                active = False
            if not active:
                continue
            pose, pose_name = _controller_pose(device, poses)
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
        if getattr(self, "_view_placement_file", None) is not None:
            if self._process_view_placement(core, result):
                self._emit_diag(core, {})
                return {}
        elif (getattr(self, "_head_view_pending", False)
                and getattr(self, "_view_profile", None) in ("table_overview", "table_front")):
            self._request_table_view(core, result)
            # Drop all old-frame hands on the scheduling step. Subsequent
            # motion retains the original independent hand-validity gates.
            self._emit_diag(core, {})
            return {}
        elif getattr(self, "_head_view_pending", False):
            if all(side in result for side in _SIDES):
                head = core.get_input_device("/user/head")
                desc = head.get_pose_desc("") if head is not None else None
                if (desc is not None
                        and (desc.validity_flags & _REQUIRED_POSE_FLAGS) == _REQUIRED_POSE_FLAGS
                        and _pose_vector(desc.pose_matrix) is not None):
                    import carb
                    from pxr import Gf, UsdGeom
                    import omni.usd

                    profile = core.get_profile("ar")
                    settings_path = profile.get_persistent_path()
                    settings = carb.settings.get_settings()
                    settings.set_bool(settings_path + "adjustForUserHeight", False)
                    settings.set_bool(settings_path + "overrideFloorAlignedAnchor", True)
                    # Identical to SDK107 XRTest.set_xr_camera; let Kit resolve the
                    # current head height/orientation. Never feed the XR output camera
                    # back into an active-camera anchor, and never repeat per head move.
                    camera_path = "/World/envs/env_0/Robot/openarm_body_link/head_camera/camera"
                    prim = omni.usd.get_context().get_stage().GetPrimAtPath(camera_path)
                    if not prim or not prim.IsA(UsdGeom.Camera):
                        raise RuntimeError(f"Robot head camera missing or not Camera: {camera_path}")
                    pose = core.get_world_transform_matrix(camera_path)
                    if _pose_vector(pose) is None:
                        return {}
                    # The display may pitch down; hand translation stays in a
                    # fixed horizontal frame. Kit poses here are physical Y-up.
                    # Use Kit's existing upright helper before mapping camera-local
                    # axes into the task's Z-up world; do not level a mixed basis.
                    head_forward = desc.pose_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
                    if np.hypot(head_forward[0], head_forward[2]) < 1e-6:
                        return {}  # Vertical gaze has no calibratable heading.
                    head_yaw = core.reorient_transform_matrix_up_right(desc.pose_matrix, y_up=True)
                    head_yaw.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 0.0))
                    camera_forward = pose.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
                    horizontal = Gf.Vec3d(camera_forward[0], camera_forward[1], 0.0)
                    if not np.isfinite([*horizontal]).all() or np.hypot(horizontal[0], horizontal[1]) < 1e-6:
                        return {}  # A vertical camera needs an explicit table heading.
                    control_camera = Gf.Matrix4d().SetLookAt(
                        Gf.Vec3d(0.0, 0.0, 0.0), horizontal, Gf.Vec3d(0.0, 0.0, 1.0),
                    ).GetInverse()
                    # Gf matrices act on row vectors: physical pose * fixed frame.
                    control_frame = head_yaw.GetInverse() * control_camera
                    core.schedule_set_camera(pose)
                    self._horizontal_controller_frame = control_frame
                    self._head_view_pending = False
                    print("[AIRSEAI_XR_VIEW_REQUEST]", dict(
                        method="schedule_set_camera", first_valid_head=True, both_hand_samples_valid=True,
                        camera=camera_path, input_frame="fixed_horizontal_physical",
                        anchor_mode=settings.get(settings_path + "anchorMode"),
                        overrideFloorAlignedAnchor=settings.get(settings_path + "overrideFloorAlignedAnchor"),
                    ), flush=True)
                    # The scheduled transform resolves on the next Kit frame. Consume
                    # no controller pose from the old frame during the transition.
            self._emit_diag(core, {})
            return {}
        if hasattr(self, "_horizontal_controller_frame"):
            for side, sample in list(result.items()):
                if sample.source != "controller":
                    continue
                # Retain the existing grip/aim/default choice and validity gate.
                # Never fall back to pitched virtual-world input if real pose fails.
                try:
                    device = core.get_input_device(f"/user/hand/{side}")
                    name = "" if sample.pose_name == "default" else sample.pose_name
                    desc = device.get_pose_desc(name)
                    real = _tracked_pose({name: desc}, name)
                    pose = None if real is None else _pose_vector(
                        desc.pose_matrix * self._horizontal_controller_frame
                    )
                except Exception:
                    pose = None
                if pose is None:
                    result.pop(side)
                else:
                    sample.pose = pose
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
    def __init__(self, task, h, *, vr_joint_speed_deg_s=30.0, vr_position_gain=1.0, vr_pose_feedback=False, **speeds):
        if task.name == "blind_peg_insertion":
            authored = sys.modules[type(task).__module__]
            knobs = dict(speeds["knobs"])
            knobs.setdefault("open_width", authored.OPEN_WIDTH)
            knobs.setdefault("close_width", authored.CLOSE_WIDTH)
            speeds["knobs"] = knobs
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
        self._pose_solver_feedback = bool(vr_pose_feedback)
        self._pose_solver_diagnostics = []
        self.vr_joint_speed_deg_s = float(vr_joint_speed_deg_s)
        self.vr_position_gain = position_gain
        self._controller_open_width = opening
        self._controller_close_width = min(closing, opening)
        if os.environ.get("AIRSEAI_XR_VIEW_PROFILE") in ("table_overview", "table_front"):
            self._xr = TrackedHands(OpenXRDeviceCfg())
            self._xr._view_profile = os.environ["AIRSEAI_XR_VIEW_PROFILE"]
            self._xr._head_view_pending = True
            if self._xr._view_profile == "table_front":
                self._xr._view_placement_file = os.environ.get("AIRSEAI_XR_VIEW_PLACEMENT_FILE")
        elif os.environ.get("AIRSEAI_XR_VIEW_PROFILE") == "blindpeg_head":
            self._xr = TrackedHands(OpenXRDeviceCfg())
            # The official camera operation must see a valid physical head pose.
            # Defer until tracking arrives; calibrate once per device lifetime.
            self._xr._head_view_pending = True
        elif os.environ.get("AIRSEAI_XR_VIEW_PROFILE") == "blindpeg_front":
            # Static desktop-workspace placement for a 1.6 m local-floor head.
            # Only yaw the XR origin; head pitch/roll remain user-controlled.
            from isaaclab.devices.openxr.xr_cfg import XrCfg

            self._xr = TrackedHands(OpenXRDeviceCfg(xr_cfg=XrCfg(
                anchor_pos=(0.9964, -0.7248, -0.8514),
                anchor_rot=(0.9059928578305452, 0.0, 0.0, 0.4232929736719493),
                anchor_prim_path=None,
            )))
        else:
            self._xr = TrackedHands(OpenXRDeviceCfg())
        self._previous = {}  # Last palm pose consumed as motion.
        self._palm_sample_previous = {}  # Adjacent sample for tracking-jump detection.
        self._controller_previous = {}  # Last controller pose consumed as motion.
        self._controller_sample_previous = {}
        self._controller_anchors = {}
        self._controller_release_targets = {}
        self._controller_target_error = {}
        self._motion_rebased = []
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
        self._controller_anchors.clear()
        self._controller_release_targets.clear()
        self._controller_target_error.clear()
        self._motion_rebased = []
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
            pose_solver=list(self._pose_solver_diagnostics),
            tracking_valid=self._last_tracking_valid,
            recording_active=self._recording_active,
            joint_speed_deg_s=self.vr_joint_speed_deg_s,
            position_gain=self.vr_position_gain,
            jaw_open_width_m=self._controller_open_width,
            jaw_close_width_m=self._controller_close_width,
            elapsed_wall_s=self._input_wall_time - self._episode_wall_start,
            safety_rejected=list(self._safety_rejected),
            motion_rebased=list(self._motion_rebased),
            controller_targets={
                side: dict(
                    desired_xyz_wxyz=np.r_[pos, quat].tolist(),
                    anchor_hand_xyz_wxyz=self._controller_anchors[side][0].tolist(),
                    anchor_tcp_xyz_wxyz=np.r_[self._controller_anchors[side][1], self._controller_anchors[side][2]].tolist(),
                    error_before_ik=self._controller_target_error.get(side),
                )
                for side, (pos, quat) in self._arm_targets.items()
                if side in self._controller_anchors
            },
            hands={
                side: dict(
                    source=sample.source, pose_name=sample.pose_name,
                    pose_xyz_wxyz=sample.pose.tolist(),
                    squeeze=sample.squeeze, trigger=sample.trigger,
                    button_0=bool(sample.button_0), button_1=bool(sample.button_1),
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

                def values(name, *, degrees=False):
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
        self._controller_anchors.clear()
        self._controller_release_targets.clear()
        self._controller_target_error.clear()
        self._arm_targets.clear()
        self._pending.clear()
        self._button_previous.clear()
        self._buttons_armed = False
        if previous_mode == "controller" or mode == "controller":
            self._cmd_pos = None
            self._cmd_quat = None

    def _controller_buttons(self, samples, cmd):
        now = {
            (side, index): getattr(samples[side], f"button_{index}")
            for side in _SIDES
            for index in (0, 1)
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
        self._motion_rebased = []
        jaws_before = dict(self._jaws)
        jaw_targets_before = dict(self._jaw_targets)
        cmd = super().advance(t_ep)
        cmd.delta_pos[:] = 0
        cmd.delta_rot[:] = 0
        self._xr._view_recenter_allowed = not self._recording_active
        samples = self._xr.advance()
        self._input_wall_time = time.monotonic()
        self._xr_samples = samples
        self._last_tracking_valid = False
        if getattr(self._xr, "_view_placement_consuming", False):
            self._switch_mode(None)
            self._jaws.update(jaws_before)
            self._jaw_targets.update(jaw_targets_before)
            cmd.side = self.side
            cmd.jaw = self._jaws[cmd.side]
            cmd.toggle_pause = cmd.mark_success = cmd.finish = cmd.discard = cmd.reset_episode = False
            cmd.tracking_valid = False
            return cmd
        if getattr(self._xr, "_view_placement_hold_motion", False):
            self._switch_mode(None)
            self._jaws.update(jaws_before)
            self._jaw_targets.update(jaw_targets_before)
            cmd.side = self.side
            cmd.jaw = self._jaws[cmd.side]
            if all(samples.get(side) is not None and samples[side].source == "controller" for side in _SIDES):
                self._controller_buttons(samples, cmd)
            cmd.mark_success = cmd.finish = cmd.discard = cmd.reset_episode = False
            if cmd.toggle_pause:
                self._recording_active = not self._recording_active
            cmd.tracking_valid = False
            return cmd
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
            if np.linalg.norm(sample_pos) > _MAX_POSE_STEP_M or np.linalg.norm(sample_rot) > _MAX_POSE_STEP_RAD:
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
            self._controller_anchors.clear()
            self._controller_release_targets.clear()
            self._controller_target_error.clear()
            self._arm_targets.clear()
        if _XR_DIAG_ENABLED and (cmd.toggle_pause or cmd.mark_success or cmd.finish or cmd.discard):
            print(
                "[AIRSEAI_XR_DIAG] command "
                f"toggle_pause={cmd.toggle_pause} mark_success={cmd.mark_success} "
                f"finish={cmd.finish} discard={cmd.discard} "
                f"recording_active={self._recording_active}",
                flush=True,
            )
        # KeyboardDevice advances one jaw; restore it before applying both
        # controller targets so no arm can move its jaw twice per control step.
        self._jaws.update(jaws_before)
        if not self._recording_active:
            self._pending.clear()
            return cmd
        jaw_step = self.jaw_speed * self.h.control_dt
        for side in _SIDES:
            sample = samples[side]
            if sample.trigger is not None:
                self._jaw_targets[side] = self._controller_open_width + (
                    self._controller_close_width - self._controller_open_width
                ) * sample.trigger
            jaw_delta = self._jaw_targets[side] - self._jaws[side]
            self._jaws[side] += float(np.clip(jaw_delta, -jaw_step, jaw_step))
            delta_pos = np.zeros(3)
            delta_rot = np.zeros(3)
            engaged = sample.squeeze > 0.5 and not cmd.toggle_pause
            if engaged:
                previous = self._controller_previous.get(side)
                previous_sample = self._controller_sample_previous.get(side)
                self._controller_sample_previous[side] = sample
                if previous is None or previous_sample is None or previous_sample.pose_name != sample.pose_name:
                    self._controller_previous[side] = sample
                    self._controller_anchors.pop(side, None)
                    self._controller_target_error.pop(side, None)
                else:
                    sample_pos, sample_rot = _pose_delta(sample.pose, previous_sample.pose)
                    if np.linalg.norm(sample_pos) > _MAX_POSE_STEP_M or np.linalg.norm(sample_rot) > _MAX_POSE_STEP_RAD:
                        self._controller_previous[side] = sample
                        self._controller_anchors.pop(side, None)
                        self._controller_target_error.pop(side, None)
                        self._motion_rebased.append(f"{side}:tracking_jump")
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
                self._controller_anchors.pop(side, None)
                self._controller_target_error.pop(side, None)
                self._arm_targets.pop(side, None)
            self._pending[side] = (delta_pos, delta_rot, self._jaws[side], engaged)
        cmd.jaw = self._jaws[cmd.side]
        return cmd

    def _cancel_controller_target(self, side, *, released=False):
        """Withdraw an old arm command through the existing joint target bounds.

        Release captures one fixed measured pose, so a loaded idle arm does not
        continually chase its own sag. Re-engagement catches the current measured
        joints before a fresh Cartesian anchor is accepted. No physical state is
        written; cancellation can take several control steps.
        """
        from sim.joint_target_safety import bound_joint_target

        if self.h._hold is None:
            self.h.latch()
        data = self.h.robot.data
        ids = self.h._chains[side]["arm"].joint_ids
        measured = data.joint_pos[:, ids]
        if released:
            desired = self._controller_release_targets.setdefault(side, measured.clone())
        else:
            self._controller_release_targets.pop(side, None)
            desired = measured
        limits = getattr(data, "soft_joint_pos_limits", None)
        if limits is None:
            self._safety_rejected.append(side)
            return False
        bounded, accepted = bound_joint_target(
            desired, self.h._hold[:, ids], limits[:, ids, :],
            np.radians(self.vr_joint_speed_deg_s) * self.h.control_dt,
        )
        if not accepted:
            self._safety_rejected.append(side)
            return False
        target = self.h._hold.clone()
        target[:, ids] = bounded
        self.h.robot.set_joint_position_target(target)
        self.h.robot.write_data_to_sim()
        self.h.latch()
        # Actual feedback can overshoot a soft limit slightly. Completion is
        # relative to the feasible stop target, never an unreachable raw value.
        feasible_desired = desired.clamp(min=limits[:, ids, 0], max=limits[:, ids, 1])
        return float(abs(bounded - feasible_desired).max().item()) <= 1e-7

    def _apply_with_current_ik(self, cmd: Command):
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
            # Start each commanded step at the measured TCP, so a slow or
            # constrained arm does not accumulate an unreachable pose target.
            if cmd.side == self.side:
                self._cmd_pos = np.asarray(self.h.tcp_pos(), dtype=float).copy()
                self._cmd_quat = np.asarray(self.h.tcp_quat(), dtype=float).copy()
            accepted = super().apply(cmd)
            if accepted is False:
                self._safety_rejected.append(cmd.side)
                self._cmd_pos = self._cmd_quat = None
            # The next IK step must start from this frame's commanded joints.
            # A rejected step leaves those joints unchanged, so latch them too.
            self.h.latch()
            return
        # Harness.step_ik_toward starts from its latched joint target. Refresh
        # the latch after *each* arm, or the second IK call erases the first.
        for side in _SIDES:
            delta_pos, delta_rot, jaw, engaged = self._pending[side]
            if not engaged:
                self._arm_targets.pop(side, None)
                self._controller_anchors.pop(side, None)
                self._controller_target_error.pop(side, None)
                self._cancel_controller_target(side, released=True)
                self.h.hold_grip(side, jaw)
                continue
            self.h.select_arm(side)
            here = np.asarray(self.h.tcp_pos(), dtype=float).copy()
            orientation = np.asarray(self.h.tcp_quat(), dtype=float).copy()
            # Clutch anchors preserve the complete relative hand displacement.
            # Speed limits restrict each output step, never consume/discard the
            # remaining displacement. A held, stationary hand can finish a move.
            hand = self._controller_previous[side].pose
            if side not in self._controller_anchors:
                if not self._cancel_controller_target(side):
                    self._arm_targets.pop(side, None)
                    self._motion_rebased.append(f"{side}:canceling_previous_target")
                    self.h.hold_grip(side, jaw)
                    continue
                self.h.reset_ik()
                self._controller_anchors[side] = (hand.copy(), here.copy(), orientation.copy())
            hand_anchor, position_anchor, rotation_anchor = self._controller_anchors[side]
            hand_pos, hand_rot = _pose_delta(hand, hand_anchor)
            pos = position_anchor + self.vr_position_gain * hand_pos
            quat = rotate_quaternion(rotation_anchor, hand_rot)
            self._arm_targets[side] = pos, quat
            position_error = float(np.linalg.norm(pos - here))
            rotation_error = float(np.linalg.norm(rotation_delta(quat, orientation)))
            self._controller_target_error[side] = dict(position_m=position_error, rotation_rad=rotation_error)
            if position_error <= _TARGET_REACHED_M and rotation_error <= _TARGET_REACHED_RAD:
                self.h.hold_grip(side, jaw)
                continue
            accepted = self.h.step_ik_toward(pos.tolist(), jaw, quat.tolist(), **self._ik_options())
            if accepted is False:
                self._safety_rejected.append(side)
                # IK rejection preserves the latched command, including its
                # gripper. Keep the next jaw slew relative to that command.
                grip_id = self.h._chains[side]["grip_ids"][0]
                self._jaws[side] = float(self.h._hold[0, grip_id])
                self._arm_targets.pop(side, None)
                self._controller_previous.pop(side, None)
                self._controller_sample_previous.pop(side, None)
                self._controller_anchors.pop(side, None)
                self._controller_target_error.pop(side, None)
            self.h.latch()
        # A jaw can change without an IK call when its squeeze is released.
        # Commit the complete latch once, preserving both arms' joint targets.
        self.h.robot.set_joint_position_target(self.h._hold)
        self.h.robot.write_data_to_sim()
        self.side = cmd.side
        self.h.select_arm(self.side)


    def apply(self, cmd: Command):
        """Use the new solver only for this controller-mode apply invocation."""
        self._pose_solver_diagnostics = []
        if self._mode != "controller":
            return self._apply_with_current_ik(cmd)
        original_compute = self.h.ik_pose.compute
        self.h.ik_pose.compute = make_harness_compute(
            self.h, self.vr_joint_speed_deg_s,
            feedback=self._pose_solver_feedback,
            diagnostic_sink=self._pose_solver_diagnostics.append,
        )
        try:
            return self._apply_with_current_ik(cmd)
        finally:
            self.h.ik_pose.compute = original_compute
