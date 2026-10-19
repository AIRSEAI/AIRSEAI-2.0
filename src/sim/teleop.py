# SPDX-License-Identifier: Apache-2.0
"""Keyboard and scripted controls producing joint-position actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Command",
    "KeyboardDevice",
    "ScriptedDevice",
    "TeleopDevice",
    "make_device",
]


@dataclass
class Command:
    """What the operator asked for this step."""

    delta_pos: np.ndarray  # metres, world frame
    delta_rot: np.ndarray  # radians, world frame
    jaw: float  # commanded opening, metres
    side: str  # which arm is being driven
    mark_success: bool = False
    discard: bool = False
    reset_episode: bool = False
    finish: bool = False
    toggle_pause: bool = False
    tracking_valid: bool = True


def rotate_quaternion(quat: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a world-frame rotation vector to a wxyz quaternion."""
    angle = np.linalg.norm(delta)
    if angle == 0:
        return quat.copy()
    w = np.cos(angle / 2)
    xyz = delta * (np.sin(angle / 2) / angle)
    result = np.concatenate(
        ([w * quat[0] - np.dot(xyz, quat[1:])], w * quat[1:] + quat[0] * xyz + np.cross(xyz, quat[1:]))
    )
    return result / np.linalg.norm(result)


class TeleopDevice:
    """One command per control step, and how to apply it."""

    def __init__(self, task, h, *, knobs: dict, pos_speed: float, rot_speed: float, jaw_speed: float):
        self.task, self.h = task, h
        self.knobs = knobs
        self.pos_speed, self.rot_speed, self.jaw_speed = pos_speed, rot_speed, jaw_speed
        self.side = "right"
        self.jaw = 0.0
        self._cmd_pos: np.ndarray | None = None
        self._cmd_quat: np.ndarray | None = None

    def help_text(self) -> str:
        return ""

    def begin_episode(self, ctx: dict) -> None:
        """Take the arm the task chose, and start from where the TCP is."""
        self.side = ctx["side"]
        self.h.select_arm(self.side)
        self.h.reset_ik()
        self.jaw = float(self.h.grip_measured())
        self._cmd_pos = None
        self._cmd_quat = None

    def advance(self, t_ep: float) -> Command:
        raise NotImplementedError

    def apply(self, cmd: Command) -> bool | None:
        """Apply a pose delta while holding the idle arm's last command."""
        h = self.h
        if cmd.side != self.side:
            h.latch()
            self.side = cmd.side
            h.select_arm(self.side)
            h.reset_ik()
            self._cmd_pos = None
            self._cmd_quat = None
        if self._cmd_pos is None:
            self._cmd_pos = np.asarray(h.tcp_pos(), dtype=float).copy()
            self._cmd_quat = np.asarray(h.tcp_quat(), dtype=float).copy()
        self._cmd_pos = self._cmd_pos + cmd.delta_pos
        self._cmd_quat = rotate_quaternion(self._cmd_quat, cmd.delta_rot)
        return h.step_ik_toward(
            self._cmd_pos.tolist(), cmd.jaw, self._cmd_quat.tolist(), **self._ik_options()
        )

    def _ik_options(self) -> dict:
        return {}

    def after_step(self, sig: dict) -> None:
        pass


class KeyboardDevice(TeleopDevice):
    """Keyboard pose control with independent gripper targets for each arm."""

    KEYS = """TELEOP keys
  W/S A/D Q/E      move the tool along x / y / z
  Z/X T/G C/V      rotate around x / y / z
  K                toggle the jaws open or shut
  TAB              switch arm
  ENTER            mark success and save
  BACKSPACE        throw it away
  R                reset (vector collection discards and consumes this attempt)"""

    def __init__(self, task, h, **speeds):
        super().__init__(task, h, **speeds)
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

        self._dev = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=self.pos_speed * h.control_dt, rot_sensitivity=self.rot_speed * h.control_dt
            )
        )
        self._flags = dict.fromkeys(("mark", "discard", "reset", "switch", "grip", "pause"), False)
        self._jaws = {}
        self._jaw_targets = {}
        for key, flag in (
            ("ENTER", "mark"),
            ("BACKSPACE", "discard"),
            ("R", "reset"),
            ("TAB", "switch"),
            ("K", "grip"),
            ("SPACE", "pause"),
        ):
            self._dev.add_callback(key, lambda f=flag: self._flags.__setitem__(f, True))

    def help_text(self) -> str:
        return self.KEYS + "\n  SPACE            start / pause recording (vector environment)"

    def begin_episode(self, ctx: dict) -> None:
        super().begin_episode(ctx)
        self._dev.reset()
        self._flags = dict.fromkeys(self._flags, False)
        for side in ("left", "right"):
            self.h.select_arm(side)
            self._jaws[side] = float(self.h.grip_measured())
        self._jaw_targets = dict(self._jaws)
        self.h.select_arm(self.side)
        self.h.latch()

    def advance(self, t_ep: float) -> Command:
        delta = self._dev.advance()
        delta = delta.detach().cpu().numpy() if hasattr(delta, "detach") else np.asarray(delta)
        flags, self._flags = dict(self._flags), dict.fromkeys(self._flags, False)
        side = ("left" if self.side == "right" else "right") if flags["switch"] else self.side
        opening = float(self.knobs.get("open_width", 0.04))
        if flags["grip"]:
            closing = float(self.knobs.get("close_width", 0.0))
            self._jaw_targets[side] = closing if self._jaw_targets[side] > opening / 2 else opening
        step = self.jaw_speed * self.h.control_dt
        self._jaws[side] += float(np.clip(self._jaw_targets[side] - self._jaws[side], -step, step))
        return Command(
            delta_pos=np.asarray(delta[:3]),
            delta_rot=np.asarray(delta[3:6]),
            jaw=self._jaws[side],
            side=side,
            mark_success=flags["mark"],
            discard=flags["discard"],
            reset_episode=flags["reset"],
            toggle_pause=flags["pause"],
        )


class ScriptedDevice(TeleopDevice):
    """Run expert phases through the recording loop without a keyboard."""

    def __init__(self, task, h, **speeds):
        super().__init__(task, h, **speeds)
        self._knobs = self.knobs
        self._phase = 0
        self._t_phase = 0.0
        self._ctx: dict | None = None
        self._target = None
        self._quat = None

    def help_text(self) -> str:
        return "TELEOP scripted: the expert drives; no keys"

    def begin_episode(self, ctx: dict) -> None:
        super().begin_episode(ctx)
        self._ctx, self._phase, self._t_phase = ctx, 0, 0.0
        self._sig = self.task.signals(self.h, ctx)
        self.timeouts = []

    def advance(self, t_ep: float) -> Command:
        ctx, task, h = self._ctx, self.task, self.h
        if self._phase >= len(task.phases):
            return Command(np.zeros(3), np.zeros(3), self.jaw, self.side, finish=True)
        name, _ = task.phases[self._phase]
        sig = self._sig
        target, grip, quat = task.phase_target(h, ctx, name, self._t_phase, sig, self._knobs)
        self._target, self._quat = target, quat
        self.jaw = float(grip)
        return Command(np.zeros(3), np.zeros(3), self.jaw, ctx["side"])

    def apply(self, cmd: Command) -> None:
        name, _ = self.task.phases[self._phase]
        if name in getattr(self.task, "joint_phases", ()):
            self.h.step_joint_toward(self._target, cmd.jaw)
        else:
            speed, angular_speed = getattr(self.task, "speed_for", lambda _: (None, None))(name)
            self.h.step_ik_toward(
                self._target, cmd.jaw, self._quat, max_speed=speed, max_ang_speed=angular_speed
            )

    def after_step(self, sig: dict) -> None:
        name, timeout = self.task.phases[self._phase]
        self._t_phase += self.h.control_dt
        self._sig = sig
        done = self.task.phase_done(self.h, self._ctx, name, self._t_phase, sig, self._target, self._knobs)
        if done or self._t_phase >= timeout:
            if not done:
                self.timeouts.append(name)
            jump = self.task.next_phase(self._ctx, name)
            names = [n for n, _ in self.task.phases]
            self._phase = names.index(jump) if jump is not None else self._phase + 1
            self._t_phase = 0.0


def make_device(name: str, task, h, **speeds) -> TeleopDevice:
    if name == "keyboard":
        return KeyboardDevice(task, h, **speeds)
    if name == "scripted":
        return ScriptedDevice(task, h, **speeds)
    if name == "vr":
        from sim.teleop_openxr import OpenXRHandDevice

        return OpenXRHandDevice(task, h, **speeds)
    raise ValueError(f"unknown teleop device {name!r}")
