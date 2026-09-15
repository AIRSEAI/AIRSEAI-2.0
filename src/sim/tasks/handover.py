"""handover expert.

Phases::

    clear -> aim_r -> reach -> aim_l -> load -> check -> present -> approach
        -> receive -> release -> retreat -> hold
    present                  policy takes over here
"""

from __future__ import annotations

import numpy as _np
import torch as _torch
from isaaclab.utils.math import quat_mul

from sim.criteria.probes import axis_from_quat, tilt_from_horizontal_deg
from sim.criteria.tasks import handover as _verdict
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.envs.bimanual_tactile import TACTILE_SENSORS
from sim.envs.handover import (
    BAR_LENGTH,
    GIVE_XY,
    HANDOVER_Z,
    SPAWN_JITTER,
    SPAWN_Z_JITTER,
    STANDOFF_Z,
    HandoverSceneCfg,
)
from sim.tactile import apply_sdf_collision
from sim.tasks.spec import Knob, Task

__all__ = ["HandoverTask", "TASK"]

DT = 1.0 / 60.0
SIDE_PADS = {"left": ("left_left_finger_tactile", "left_right_finger_tactile"),
             "right": ("right_left_finger_tactile", "right_right_finger_tactile")}

# Controller parameters.
# Jaw command while approaching and releasing, metres per jaw.
OPEN_WIDTH = 0.026
# Runaway floor for the closing ramp, metres per jaw: reaching it means the
# bar is not between the jaws.
GRIP_MIN_WIDTH = 0.016

# --- grasp geometry --------------------------------------------------------
# The jaws close along world x: the bar lies along y, the axis the arms are
# separated along.
JAW_AXIS_W = (1.0, 0.0, 0.0)
# Where along the bar each gripper takes it, metres from its centre: this
# much inboard of each end.
BAR_GRASP_INSET = 0.060
GRASP_OFFSET_Y = {"left": BAR_LENGTH / 2.0 - BAR_GRASP_INSET,
                  "right": -(BAR_LENGTH / 2.0 - BAR_GRASP_INSET)}
APPROACH_ABOVE = 0.040        # m the receiver stops above the bar before the final descent
HELD_TOL_M = 0.015            # how far the bar may sit from the gel midpoint and still count as held

# --- closing ramp ----------------------------------------------------------
CLOSE_RATE = 0.015            # m/s per jaw
OPEN_RATE = 0.040             # m/s per jaw; fast, so the release does not spend the gap budget
GRIP_TARGET_MM2 = 120.0       # per side; first solid contact, where the preload starts
RECEIVE_EXIT_MM2 = 150.0      # per side; the receiver has load, the giver may let go

# --- phase completion ------------------------------------------------------
SETTLE_STEPS = 30             # control steps held after reset, split across both arms
RETREAT_UP = 0.10             # m; the giver leaves up and outboard
RETREAT_OUT = 0.12            # m outboard


def _aim_quat(h, side: str):
    """Wrist orientation putting `side`'s jaw axis across the bar: the minimal
    rotation carrying the measured jaw axis onto JAW_AXIS_W, composed with
    the current TCP orientation. Recomputed each step."""
    q_now = h.robot.data.body_state_w[:, h.ee_body(side), 3:7]
    ax = h.jaw_axis(side)
    if ax is None:
        return q_now[0]
    tgt = _np.asarray(JAW_AXIS_W, dtype=float)
    tgt = tgt * (1.0 if float(ax @ tgt) >= 0.0 else -1.0)   # a line, not a direction
    v = _np.cross(ax, tgt)
    s, c = float(_np.linalg.norm(v)), float(ax @ tgt)
    if s < 1e-6:
        return q_now[0]
    half = float(_np.arctan2(s, c)) / 2.0
    q_d = _torch.tensor([[_np.cos(half), *(v / s * _np.sin(half))]],
                        dtype=_torch.float32, device=q_now.device)
    return quat_mul(q_d, q_now)[0]


def _jaw_err_deg(h, side: str) -> float:
    """Angle between `side`'s jaw axis and JAW_AXIS_W, unsigned."""
    ax = h.jaw_axis(side)
    if ax is None:
        return 180.0
    return float(_np.degrees(_np.arccos(
        _np.clip(abs(float(ax @ _np.asarray(JAW_AXIS_W))), 0.0, 1.0))))


def _gel_mid(h, side: str):
    c = h.gel_centres(side)
    return None if c is None else 0.5 * (c[0] + c[1])


def _bar_pq(h):
    bar = h.scene["manipuland"]
    return (bar.data.root_pos_w[0].detach().cpu().numpy().copy(),
            bar.data.root_quat_w[0].detach().cpu().numpy().copy())


def _bar_grasp_point(h, side: str) -> _np.ndarray:
    """World position of `side`'s grasp point on the bar, honouring its tilt."""
    p, q = _bar_pq(h)
    return p + axis_from_quat(q, (0.0, GRASP_OFFSET_Y[side], 0.0))


class HandoverTask(Task):
    name = "handover"
    instruction = "take the bar from the other hand"
    report_name = "handover_scripted.json"
    knobs = {
        "max_gap_s": Knob(0.5, "seconds the giver may keep holding after the receiver takes load"),
        "max_tilt_deg": Knob(15.0, "bar tilt during the transfer above which the "
                                   "episode fails"),
        "grip_preload": Knob(0.004, "how far past first contact the jaws keep "
                                    "closing, metres of jaw command"),
        "jaw_tol_deg": Knob(4.0, "how close the jaw axis must come to the bar's "
                                 "normal before a grasp is attempted"),
    }
    phases = [
        ("clear", 4.0),      # receiver to standoff           right, position IK
        ("aim_r", 5.0),      # receiver's jaws onto x         right, pose IK in place
        ("reach", 4.0),      # giver to the handover pose     left, position IK
        ("aim_l", 3.0),      # giver's jaws onto x            left, pose IK in place
        ("load", 3.0),       # bar pinned between the giver's jaws while they close
        ("check", 2.0),      # pin off: does the giver hold it?
        ("present", 1.5),    # -- recording starts here --
        ("approach", 8.0),   # receiver down to just above the bar
        ("receive", 4.0),    # receiver onto the bar and closing
        ("release", 1.0),    # receiver keeps closing, giver opens
        ("retreat", 3.0),    # giver up and away
        ("hold", 1.5),
    ]
    handover_phase = "present"
    horizon_s = 20.0
    scene_cam = ((0.90, -0.70, 0.55), (0.26, 0.0, HANDOVER_Z))

    _ARM = {"clear": "right", "aim_r": "right", "reach": "left", "aim_l": "left",
            "load": "left", "check": "left", "present": "left", "approach": "right",
            "receive": "right", "release": "right", "retreat": "left", "hold": "right"}
    _AIM = {"aim_r": "right", "aim_l": "left"}

    # ---- scene -------------------------------------------------------------
    def make_scene_cfg(self, knobs: dict):
        return HandoverSceneCfg(num_envs=1, env_spacing=3.0)

    def apply_collision(self, h) -> None:
        apply_sdf_collision("/World/envs/env_0/manipuland")

    # ---- episode -----------------------------------------------------------
    def reset(self, h, rng, ep, knobs) -> dict:
        h.unlatch()
        give = _np.asarray(GIVE_XY) + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
        take = _np.asarray([give[0] + rng.uniform(-0.010, 0.010),
                            give[1] + GRASP_OFFSET_Y["right"] - GRASP_OFFSET_Y["left"]])
        z_hand = HANDOVER_Z + float(rng.uniform(-SPAWN_Z_JITTER, SPAWN_Z_JITTER))
        h.place(h.scene["manipuland"], (0.0, 0.0), -5.0)     # parked until `load`
        h.reset_robot()
        for side in ("right", "left"):
            h.select_arm(side)
            for _ in range(SETTLE_STEPS // 2):
                h.step_ik_toward(h.tcp_pos(), OPEN_WIDTH)
                h.step()
        return {
            "side": "right",
            "give": give, "take": take, "z_hand": float(z_hand),
            "phase_i": 0,
            "grip": {"left": OPEN_WIDTH, "right": OPEN_WIDTH},
            "touched": {},                 # side -> jaw command at first solid contact
            "frozen": {},                  # side -> TCP position held through an aim
            "aimed": {},                   # side -> wrist quaternion after its aim
            "goal": None, "goal2": None,   # receiver's waypoints, read from the bar
            "preload": float(knobs["grip_preload"]),
        }

    # ---- probes ------------------------------------------------------------
    def _squeeze(self, h, ctx, side: str, *, pin: bool) -> float:
        """Ramp `side`'s jaws to first solid contact, then a fixed preload
        past it. With `pin`, hold the bar at the gel midpoint meanwhile."""
        g = ctx["grip"][side]
        areas = h.pad_areas()
        if side not in ctx["touched"]:
            if pin:
                self._pin_bar(h, side)
            if sum(areas[p] for p in SIDE_PADS[side]) >= GRIP_TARGET_MM2:
                ctx["touched"][side] = g
                return g
            return max(GRIP_MIN_WIDTH, g - CLOSE_RATE * DT)
        floor = self._grip_floor(ctx, side)
        if g > floor and pin:
            self._pin_bar(h, side)
        return max(floor, g - CLOSE_RATE * DT)

    def _grip_floor(self, ctx, side: str) -> float:
        return max(GRIP_MIN_WIDTH, ctx["touched"][side] - ctx["preload"])

    def _squeezed(self, ctx, side: str) -> bool:
        return side in ctx["touched"] and ctx["grip"][side] <= self._grip_floor(ctx, side) + 1e-9

    def _pin_bar(self, h, side: str) -> None:
        """Hold the bar with `side`'s grasp point at the measured gel midpoint."""
        mid = _gel_mid(h, side)
        if mid is not None:
            h.place(h.scene["manipuland"],
                    (float(mid[0]), float(mid[1]) - GRASP_OFFSET_Y[side]), float(mid[2]))

    def _bar_held_by(self, h, side: str) -> bool:
        """Is the bar where `side`'s jaws are, with nothing pinning it?"""
        mid = _gel_mid(h, side)
        return mid is not None and bool(
            _np.linalg.norm(mid - _bar_grasp_point(h, side)) < HELD_TOL_M)

    def signals(self, h, ctx) -> dict:
        areas = h.pad_areas()
        p, q = _bar_pq(h)
        return {
            "areas": areas,
            "side_area": {s: sum(areas[x] for x in SIDE_PADS[s]) for s in SIDE_PADS},
            "bar_pos": p,
            "tilt": tilt_from_horizontal_deg(q, (0.0, 1.0, 0.0)),
            "phase_i": ctx["phase_i"],
        }

    # ---- phases ------------------------------------------------------------
    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        arm = self._ARM[name]
        names = [n for n, _ in self.phases]
        if t_phase == 0.0:
            h.latch()                      # the arm that just moved holds where it is
            # An aim that timed out still freezes its wrist, so what follows
            # starts from a known orientation.
            for aim, side in self._AIM.items():
                if names.index(name) > names.index(aim) and side not in ctx["aimed"]:
                    self._freeze_aim(h, ctx, side)
        h.select_arm(arm)
        ctx["phase_i"] = names.index(name)
        give, take, z = ctx["give"], ctx["take"], ctx["z_hand"]
        grip = ctx["grip"]
        quat = None

        if name == "clear":
            grip["right"] = OPEN_WIDTH
            tgt = [take[0], take[1], STANDOFF_Z]
        elif name in self._AIM:
            # Rotate in place: the position is frozen where the arm stands.
            if arm not in ctx["frozen"]:
                ctx["frozen"][arm] = list(h.tcp_pos())
            tgt = ctx["frozen"][arm]
            quat = _aim_quat(h, arm)
        elif name == "reach":
            grip["left"] = OPEN_WIDTH
            tgt = [give[0], give[1], z]
        elif name in ("load", "check", "present"):
            tgt = ctx["frozen"]["left"]
            quat = ctx["aimed"]["left"]
            if name == "load":
                grip["left"] = self._squeeze(h, ctx, "left", pin=True)
        elif name == "approach":
            # Read once from the bar's actual pose and held; stop above it.
            if ctx["goal"] is None:
                ctx["goal"] = list(_bar_grasp_point(h, "right"))
            tgt = list(ctx["goal"])
            tgt[2] += APPROACH_ABOVE
            grip["right"] = OPEN_WIDTH
        elif name == "receive":
            # A second waypoint on arrival: the bar has sagged since the first.
            if ctx["goal2"] is None:
                ctx["goal2"] = list(_bar_grasp_point(h, "right"))
            tgt = ctx["goal2"]
            if h.tcp_err(tgt) < 0.012:
                grip["right"] = self._squeeze(h, ctx, "right", pin=False)
        elif name == "release":
            # Independent actuators: the receiver keeps closing while the
            # giver opens.
            tgt = ctx["goal2"]
            grip["right"] = self._squeeze(h, ctx, "right", pin=False)
            grip["left"] = min(OPEN_WIDTH, grip["left"] + OPEN_RATE * DT)
            h.hold_grip("left", grip["left"])
        elif name == "retreat":
            # Up and outboard, so a retreating pad does not brush the bar
            # once more. The receiver's preload finishes meanwhile.
            grip["left"] = OPEN_WIDTH
            tgt = [give[0], give[1] + RETREAT_OUT, z + RETREAT_UP]
            grip["right"] = self._squeeze(h, ctx, "right", pin=False)
            h.hold_grip("right", grip["right"])
        else:                                          # hold
            tgt = ctx["goal2"]
            grip["right"] = self._squeeze(h, ctx, "right", pin=False)
        return tgt, grip[arm], quat

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        arm = self._ARM[name]
        if name == "clear":
            return h.tcp_err(target) < 0.020
        if name in self._AIM:
            done = _jaw_err_deg(h, arm) < float(knobs["jaw_tol_deg"])
            if done:
                self._freeze_aim(h, ctx, arm)
            return done
        if name == "reach":
            return h.tcp_err(target) < 0.015
        if name == "load":
            return self._squeezed(ctx, "left")
        if name == "check":
            return t_phase > 0.4 and self._bar_held_by(h, "left")
        if name == "present":
            return t_phase > 0.5
        if name == "approach":
            return h.tcp_err(target) < 0.015
        if name == "receive":
            return sig["side_area"]["right"] >= RECEIVE_EXIT_MM2
        if name == "release":
            return sig["side_area"]["left"] < 1.0
        if name == "retreat":
            return h.tcp_err(target) < 0.025
        return t_phase > 0.9 * dict(self.phases)["hold"]

    def _freeze_aim(self, h, ctx, side: str) -> None:
        ctx["aimed"][side] = h.robot.data.body_state_w[0, h.ee_body(side), 3:7].clone()
        ctx["frozen"][side] = list(h.tcp_pos())

    def handover_ok(self, h, ctx, sig) -> bool:
        return self._bar_held_by(h, "left")

    # ---- scoring -----------------------------------------------------------
    def build_trace(self, h, ctx, hist) -> _Trace:
        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={"manipuland": _np.asarray(hist["bar_pos"])},
            extra={"tilt_deg": _np.asarray(hist["tilt"])},
            phase=_np.asarray(hist["phase_i"]),
            phase_names=tuple(n for n, _ in self.phases),
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(trace, max_gap_s=float(knobs["max_gap_s"]),
                        max_tilt_deg=float(knobs["max_tilt_deg"]))

    def episode_extra(self, ctx, hist) -> dict:
        names = [n for n, _ in self.phases]
        phase = _np.asarray(hist["phase_i"])
        tilt = _np.asarray(hist["tilt"])
        held = phase <= names.index("receive")
        return {
            "give": [round(float(v), 4) for v in ctx["give"]],
            "take": [round(float(v), 4) for v in ctx["take"]],
            "z_hand": round(ctx["z_hand"], 4),
            "tilt_during_transfer_deg": round(float(tilt[held].max()), 1) if held.any() else None,
        }

    def report_extra(self, knobs) -> dict:
        return {"max_gap_s": float(knobs["max_gap_s"]),
                "max_tilt_deg": float(knobs["max_tilt_deg"])}


TASK = HandoverTask()
