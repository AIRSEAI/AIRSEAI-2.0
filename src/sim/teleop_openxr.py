# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab OpenXR hand poses with keyboard recording controls."""

import numpy as np
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from omni.kit.xr.core import XRCore, XRPoseValidityFlags

from sim.teleop import KeyboardDevice


class TrackedHands(OpenXRDevice):
    """Keep tracking validity rather than replacing missing poses with stale data."""

    def _get_raw_data(self):
        result = {}
        core = XRCore.get_singleton()
        if core is None:
            return result
        required = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
        for side in ("right", "left"):
            device = core.get_input_device(f"/user/hand/{side}")
            if device is None:
                continue
            poses = device.get_all_virtual_world_poses()
            palm = poses.get("palm")
            if palm is None or palm.validity_flags & required != required:
                continue
            matrix = palm.pose_matrix
            quaternion = matrix.ExtractRotationQuat()
            result[side] = np.asarray(
                [*matrix.ExtractTranslation(), quaternion.GetReal(), *quaternion.GetImaginary()], dtype=float
            )
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


class OpenXRHandDevice(KeyboardDevice):
    def __init__(self, task, h, **speeds):
        super().__init__(task, h, **speeds)
        self._xr = TrackedHands(OpenXRDeviceCfg())
        self._previous = {}

    def begin_episode(self, ctx):
        super().begin_episode(ctx)
        self._xr.reset()
        self._previous.clear()

    def help_text(self):
        return (
            "VR: move the selected hand; TAB selects right/left arm, K controls grip.\n"
            "SPACE starts/pauses, ENTER saves, BACKSPACE discards, R resets.\n"
            "Tracking loss freezes simulation until a valid hand pose returns."
        )

    def advance(self, t_ep):
        cmd = super().advance(t_ep)
        cmd.delta_pos[:] = 0
        cmd.delta_rot[:] = 0
        pose = self._xr.advance().get(cmd.side)
        if pose is None or not np.isfinite(pose).all():
            self._previous.clear()
            cmd.tracking_valid = False
            return cmd
        previous = self._previous.get(cmd.side)
        self._previous = {cmd.side: pose.copy()}
        if previous is None or cmd.toggle_pause or cmd.side != self.side:
            return cmd
        for destination, delta, speed in (
            (cmd.delta_pos, pose[:3] - previous[:3], self.pos_speed),
            (cmd.delta_rot, rotation_delta(pose[3:], previous[3:]), self.rot_speed),
        ):
            norm = np.linalg.norm(delta)
            destination[:] = delta * min(1.0, speed * self.h.control_dt / max(norm, 1e-12))
        return cmd
