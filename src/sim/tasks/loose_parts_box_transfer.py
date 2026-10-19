# SPDX-License-Identifier: Apache-2.0
"""Bimanual box transfer with contact-area grip feedback and synchronized transport."""

from __future__ import annotations

import copy

import numpy as np

from sim.criteria.loose_parts_box_transfer import part_escaped
from sim.criteria.loose_parts_box_transfer import verdict as _verdict
from sim.criteria.trace import EpisodeTrace
from sim.dims.loose_parts_box_transfer import (
    BOX_LENGTH,
    BOX_WALL,
    BOX_WIDTH,
    NUT_FLATS,
    SHELF_HEIGHT,
)
from sim.envs.bimanual_tactile import TABLE_TOP_Z, TACTILE_SENSORS, gel_gap
from sim.envs.loose_parts_box_transfer import (
    BALL_NAMES,
    BALL_SPECS,
    BOX_XY,
    GOAL_XY,
    LOOSE_NAMES,
    NUT_NAMES,
    NUT_OFFSETS,
    LoosePartsBoxTransferSceneCfg,
)
from sim.tactile import apply_sdf_collision
from sim.tasks.scene_utils import _mat_from_quat, _quat_from_mat, oblique_grasp_quat
from sim.tasks.spec import Knob, Task

OPEN_WIDTH = 0.018
CLOSE_WIDTH = 0.004
MIN_WIDTH = max(0.0, (BOX_WALL - 0.004) / 2)
GRASP_Z = TABLE_TOP_Z + 0.035
GRASP_X = {"right": -0.06, "left": 0.0}
GRASP_Y = {"right": -(BOX_WIDTH - BOX_WALL) / 2, "left": (BOX_WIDTH - BOX_WALL) / 2}
APPROACH_BACKOFF = 0.055
LIFT_HEIGHT = SHELF_HEIGHT + 0.05
SETTLE_STEPS = 90
CONTACT_AREA = 70.0
GRASP_MIN_AREA = 50.0


def _part_pose(h, name):
    data = h.scene[name].data
    return data.root_pos_w[0].detach().cpu().numpy().copy()


class _JointLimitedIK:
    def __init__(self, controller, harness):
        self.controller = controller
        self.harness = harness

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def compute(self, position, orientation, jacobian, joints):
        desired = self.controller.compute(position, orientation, jacobian, joints)
        from scipy.optimize import lsq_linear

        h = self.harness
        limits = h.robot.data.soft_joint_pos_limits[0, h.ch["arm"].joint_ids].cpu().numpy()
        q = joints[0].cpu().numpy().astype(float)
        jac = jacobian[0].cpu().numpy().astype(float)
        delta = (desired[0] - joints[0]).cpu().numpy()
        if jac.shape[0] == 6:
            jac[3:] *= 0.2
        margin = np.radians(3.0)
        low = np.clip(limits[:, 0] + margin - q, -0.04, 0.04)
        high = np.clip(limits[:, 1] - margin - q, -0.04, 0.04)
        high = np.maximum(high, low + 1e-08)
        preferred = 0.05 * (np.clip(q, limits[:, 0] + 0.26, limits[:, 1] - 0.26) - q)
        reference = getattr(h, "grasp_posture_reference", {}).get(h.ch["side"])
        if reference is not None:
            preferred = 0.15 * (reference - q)
        matrix = np.vstack((jac, 0.005 * np.eye(len(q))))
        residual = np.concatenate((jac @ delta, 0.005 * preferred))
        result = lsq_linear(matrix, residual, bounds=(low, high), tol=1e-08)
        return joints + joints.new_tensor(result.x).unsqueeze(0)


class LoosePartsBoxTransferTask(Task):
    name = "loose_parts_box_transfer"
    instruction = "place the open parts box on the green shelf mat without spilling its contents"
    report_name = "loose_parts_box_transfer_scripted.json"
    ball_names = BALL_NAMES
    loose_names = LOOSE_NAMES
    ball_specs = BALL_SPECS
    nut_offsets = NUT_OFFSETS
    knobs = {
        "carry_yaw": Knob(30.0, "horizontal box rotation during transport (degrees)"),
        "sync_lead": Knob(0.006, "carrying target lead allowed when advancing (m)"),
        "sync_tolerance": Knob(0.006, "relative hand displacement allowed during carry (m)"),
        "close_width": Knob(CLOSE_WIDTH, "jaw command while holding the box (m)"),
        "grip_min_width": Knob(0.0028, "minimum jaw width during tactile correction (m)"),
        "grip_rate": Knob(0.012, "tactile jaw closing speed (m/s)"),
        "contact_area_target": Knob(90.0, "minimum desired pad contact area (mm2)"),
        "transport_s": Knob(4.0, "time to move the box to the mat (s)"),
        "lift_height": Knob(LIFT_HEIGHT, "height above the initial box position (m)"),
        "lift_s": Knob(4.0, "time to lift the box (s)"),
        "grasp_height": Knob(GRASP_Z - TABLE_TOP_Z, "grasp height above the box bottom (m)"),
        "grasp_x_right": Knob(GRASP_X["right"], "right grasp offset along the box (m)"),
        "grasp_x_left": Knob(GRASP_X["left"], "left grasp offset along the box (m)"),
        "grasp_y_left": Knob(0.0, "left grasp shift across the box wall (m)"),
        "grasp_pitch": Knob(45.0, "finger blade downward pitch (degrees)"),
        "grasp_pitch_left": Knob(45.0, "left finger blade downward pitch (degrees)"),
        "grasp_yaw": Knob(0.0, "heading of the right hand (degrees)"),
        "grasp_yaw_left": Knob(0.0, "left finger blade heading (degrees)"),
        "tactile_feedback": Knob(1, "enable contact-area grip feedback", int),
    }
    joint_phases = ("prepare",)
    phases = [
        ("prepare", 3.0),
        ("home", 3.0),
        ("approach", 3.0),
        ("orient", 2.0),
        ("descend", 3.0),
        ("close", 2.0),
        ("lift", 5.0),
        ("transport", 35.0),
        ("lower", 3.0),
        ("support", 2.0),
        ("clear_right", 2.0),
        ("square", 5.0),
        ("release", 1.0),
        ("retreat", 2.0),
        ("settle", 1.5),
    ]
    handover_phase = None
    horizon_s = 65.0
    completion_on_success = True
    scene_cam = ((0.62, -0.42, 0.85), (0.29, 0.0, 0.22))

    def make_scene_cfg(self, knobs):
        if not 0.0 <= knobs["grip_min_width"] <= MIN_WIDTH:
            raise ValueError("grip_min_width is outside the supported range")
        for name in (
            "grip_rate",
            "contact_area_target",
            "sync_lead",
            "sync_tolerance",
            "transport_s",
            "lift_s",
        ):
            if not np.isfinite(knobs[name]) or knobs[name] <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(knobs["lift_height"]) or knobs["lift_height"] < SHELF_HEIGHT + 0.03:
            raise ValueError("lift_height must clear the shelf by at least 0.03 m")
        return LoosePartsBoxTransferSceneCfg(num_envs=1, env_spacing=3.0)

    def apply_collision(self, h):
        paths = apply_sdf_collision("/World/envs/env_0/parts_box", sdf_resolution=256)
        if not paths:
            raise RuntimeError("parts_box has no tactile collision meshes")

    def _new_ctx(self, h, knobs):
        pos = _part_pose(h, "parts_box")
        h.grasp_posture_reference = {
            "right": np.radians((-23.83, 6.7, -36.38, 77.5, 16.74, -33.23, -1.4)),
            "left": np.radians((2.23, 3.55, 22.96, 61.54, -8.14, 21.97, 13.42)),
        }
        for name in ("ik_pos", "ik_pose"):
            controller = getattr(h, name)
            if not isinstance(controller, _JointLimitedIK):
                controller = _JointLimitedIK(controller, h)
                setattr(h, name, controller)
        h.set_posture_bias(None)
        return {
            "prepare_start": h.robot.data.joint_pos[0].cpu().numpy().copy(),
            "side": "right",
            "start": pos,
            "active": "right",
            "quat": {"right": None, "left": None},
            "grip": {"right": 0.0, "left": 0.0},
            "spilled": False,
            "first_spill_object": None,
            "adjustments": 0,
            "grasp_good_s": 0.0,
            "both_closed": False,
            "orientation_ready": False,
            "aligned_for_close": False,
            "lift_loss_s": 0.0,
            "lift_lost": False,
            "grasp_failed": False,
            "carry_origins": None,
            "carry_u": 0.0,
            "carry_pair_error": 0.0,
            "carry_aborted": False,
            "square_done": False,
            "knobs": knobs,
        }

    def reset(self, h, rng, ep, knobs):
        h.unlatch()
        h.select_arm("right")
        h.place(h.scene["parts_box"], BOX_XY, TABLE_TOP_Z)
        for name, offset in zip(NUT_NAMES, self.nut_offsets, strict=True):
            h.place(h.scene[name], np.asarray(BOX_XY) + offset, TABLE_TOP_Z + BOX_WALL)
        for name, (x, y, radius, z_offset) in zip(self.ball_names, self.ball_specs, strict=True):
            h.place(h.scene[name], np.asarray(BOX_XY) + (x, y), TABLE_TOP_Z + BOX_WALL + radius + z_offset)
        h.reset_robot()
        h.hold_default(SETTLE_STEPS)
        return self._new_ctx(h, knobs)

    settle_seconds = SETTLE_STEPS / 60.0

    def sample_init_state(self, robot, rng, state_id, knobs):
        objects = {"parts_box": [*BOX_XY, TABLE_TOP_Z, 1.0, 0.0, 0.0, 0.0]}
        for name, offset in zip(NUT_NAMES, self.nut_offsets, strict=True):
            xy = np.asarray(BOX_XY) + offset
            objects[name] = [*xy.tolist(), TABLE_TOP_Z + BOX_WALL, 1.0, 0.0, 0.0, 0.0]
        for name, (x, y, radius, z_offset) in zip(self.ball_names, self.ball_specs, strict=True):
            xy = np.asarray(BOX_XY) + (x, y)
            objects[name] = [*xy.tolist(), TABLE_TOP_Z + BOX_WALL + radius + z_offset, 1.0, 0.0, 0.0, 0.0]
        return {
            "id": state_id,
            "robot": copy.deepcopy(robot),
            "objects": objects,
            "task_vars": self._state_vars(),
        }

    @staticmethod
    def _state_vars():
        return {"side": "right", "goal_surface_z": TABLE_TOP_Z + SHELF_HEIGHT, "goal_xy": list(GOAL_XY)}

    def validate_init_state(self, state):
        if state["task_vars"].get("side") != "right":
            raise ValueError("loose_parts_box_transfer requires side=right")
        for key in ("goal_surface_z", "goal_xy"):
            if state["task_vars"].get(key) != self._state_vars()[key]:
                raise ValueError("initial state belongs to a different target arrangement")

    def begin_restored_episode(self, h, state, knobs):
        self.validate_init_state(state)
        h.unlatch()
        h.select_arm("right")
        return self._new_ctx(h, knobs)

    def init_state_objects(self):
        return ("parts_box", *self.loose_names)

    def snapshot_init_state(self, h, ctx):
        return self._state_vars()

    def restore_init_state(self, h, state, knobs):
        h.hold_default(SETTLE_STEPS)
        return self.begin_restored_episode(h, state, knobs)

    def _alignment(self, h, ctx):
        box = h.scene["parts_box"].data
        origin = box.root_pos_w[0].detach().cpu().numpy()
        rotation = _mat_from_quat(box.root_quat_w[0].detach().cpu().numpy())
        result = {}
        for side in ("right", "left"):
            centres = h.gel_centres(side)
            if centres is None:
                result.update(
                    {f"{side}_wall_offset": 1.0, f"{side}_jaw_angle": 90.0, f"{side}_straddles_wall": 0.0}
                )
                continue
            local = (np.asarray(centres) - origin) @ rotation
            distances = local[:, 1] - GRASP_Y[side]
            axis = local[1] - local[0]
            cosine = abs(axis[1]) / max(float(np.linalg.norm(axis)), 1e-09)
            result.update(
                {
                    f"{side}_wall_offset": float(abs(distances.mean())),
                    f"{side}_jaw_angle": float(np.degrees(np.arccos(np.clip(cosine, 0, 1)))),
                    f"{side}_straddles_wall": float(distances[0] * distances[1] < 0),
                }
            )
        return result

    def _escaped_parts(self, h):
        box = h.scene["parts_box"].data
        pos = box.root_pos_w[0].detach().cpu().numpy()
        quat = box.root_quat_w[0].detach().cpu().numpy()
        radii = {name: spec[2] for name, spec in zip(self.ball_names, self.ball_specs, strict=True)}
        return [
            name for name in self.loose_names
            if part_escaped(pos, quat, _part_pose(h, name), radii.get(name, NUT_FLATS / 2))
        ]

    def after_control_step(self, h, ctx, dt):
        escaped = self._escaped_parts(h)
        if escaped:
            if ctx["first_spill_object"] is None:
                ctx["first_spill_object"] = escaped[0]
            ctx["spilled"] = True

    def signals(self, h, ctx):
        box = h.scene["parts_box"].data
        pos = box.root_pos_w[0].detach().cpu().numpy().copy()
        quat = box.root_quat_w[0].detach().cpu().numpy().copy()
        rotation = _mat_from_quat(quat)
        escaped_count = len(self._escaped_parts(h))
        return {
            **self._alignment(h, ctx),
            "areas": h.pad_areas(),
            "box_pos": pos,
            "box_z": float(pos[2]),
            "corner_clearance": float(
                pos[2]
                - TABLE_TOP_Z
                - abs(rotation[2, 0]) * BOX_LENGTH / 2
                - abs(rotation[2, 1]) * BOX_WIDTH / 2
            ),
            "box_yaw_deg": float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
            "tilt": float(np.degrees(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))),
            **{
                f"{side}_tcp_{axis}": float(h.robot.data.body_pos_w[0, h.ee_body(side), i])
                for side in ("right", "left")
                for (i, axis) in enumerate("xyz")
            },
            "goal_xy_error": float(np.linalg.norm(pos[:2] - GOAL_XY)),
            "escaped": float(ctx["spilled"]),
            "escaped_count": escaped_count,
            "right_grip_cmd": ctx["grip"]["right"],
            "left_grip_cmd": ctx["grip"]["left"],
            "carry_progress": ctx["carry_u"],
            "carry_pair_error_m": ctx["carry_pair_error"],
        }

    @staticmethod
    def _blade(side, knobs):
        yaw = np.radians(knobs["grasp_yaw"] if side == "right" else knobs["grasp_yaw_left"])
        pitch = np.radians(knobs["grasp_pitch"] if side == "right" else knobs["grasp_pitch_left"])
        return np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])

    def _grasp(self, ctx, side, shift=None):
        point = np.asarray(ctx["start"], dtype=float).copy()
        x = ctx["knobs"]["grasp_x_right"] if side == "right" else ctx["knobs"]["grasp_x_left"]
        y = GRASP_Y[side]
        if side == "left":
            y += ctx["knobs"]["grasp_y_left"]
        point += [x, y, ctx["knobs"]["grasp_height"]]
        if shift is not None:
            point += shift
        return point

    def _select(self, h, ctx, side):
        if ctx["active"] != side:
            h.latch()
            h.select_arm(side)
            ctx["active"] = side
            ctx["side"] = side

    def _wrist(self, h, ctx, side):
        self._select(h, ctx, side)
        if ctx["quat"][side] is None and gel_gap(h.grip_measured()) > 0.01:
            ctx["quat"][side] = oblique_grasp_quat(
                h.tcp_quat(),
                h.jaw_axis(),
                h.approach_axis(),
                ctx["knobs"]["grasp_pitch"] if side == "right" else ctx["knobs"]["grasp_pitch_left"],
                ctx["knobs"]["grasp_yaw"] if side == "right" else ctx["knobs"]["grasp_yaw_left"],
            )
        return ctx["quat"][side]

    def _carry_grip(self, ctx, sig, side, knobs, dt):
        pads = TACTILE_SENSORS[:2] if side == "right" else TACTILE_SENSORS[2:]
        area = min(sig["areas"][name] for name in pads)
        g = ctx["grip"][side]
        target = knobs["contact_area_target"]
        if knobs["tactile_feedback"] and area < target and (g > knobs["grip_min_width"]):
            rate = knobs["grip_rate"]
            rate *= 0.25 + 0.75 * (1.0 - area / target)
            g = max(knobs["grip_min_width"], g - rate * dt)
            ctx["adjustments"] += 1
        ctx["grip"][side] = g
        return g

    def _both(self, h, ctx, targets, commands, quats):
        h.unlatch()
        h.select_arm("right")
        h.step_ik_toward(targets["right"], commands["right"], quats["right"])
        h.latch()
        h.select_arm("left")
        ctx["active"] = ctx["side"] = "left"
        return (targets["left"].tolist(), commands["left"], quats["left"])

    def _dual(self, h, ctx, sig, shift, knobs, grip=None):
        targets = {side: self._grasp(ctx, side, shift) for side in ("right", "left")}
        commands = {
            side: grip if grip is not None else self._carry_grip(ctx, sig, side, knobs, h.control_dt)
            for side in ("right", "left")
        }
        return self._both(h, ctx, targets, commands, ctx["quat"])

    def _synchronized_carry(self, h, ctx, name, t_phase, sig, knobs):
        positions = {
            side: np.array([sig[f"{side}_tcp_{axis}"] for axis in "xyz"]) for side in ("right", "left")
        }
        if ctx["carry_origins"] is None:
            ctx["carry_origins"] = positions
            ctx["carry_box_start"] = sig["box_pos"].copy()
        origins = ctx["carry_origins"]
        delta = np.array([*np.asarray(GOAL_XY) - ctx["carry_box_start"][:2], 0.0])
        yaw = np.radians(knobs["carry_yaw"])
        u = ctx["carry_u"]

        def turn(progress):
            angle = yaw * progress
            (c, s) = (np.cos(angle), np.sin(angle))
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        rotation = turn(u)
        relative = {side: origins[side] - ctx["carry_box_start"] for side in positions}
        expected = {
            side: ctx["carry_box_start"] + delta * u + rotation @ relative[side] for side in positions
        }
        errors = {side: positions[side] - expected[side] for side in positions}
        ctx["carry_pair_error"] = float(np.linalg.norm(errors["left"] - errors["right"]))
        if name == "transport":
            tangents = {
                side: delta + yaw * np.cross([0.0, 0.0, 1.0], rotation @ relative[side]) for side in positions
            }
            progress_limits = []
            for side in positions:
                tangent = tangents[side]
                length = max(float(np.linalg.norm(tangent)), 1e-09)
                progress_limits.append(
                    u + float(np.dot(errors[side], tangent)) / length**2 + knobs["sync_lead"] / length
                )
            advance = h.control_dt / float(knobs["transport_s"])
            if ctx["carry_pair_error"] > knobs["sync_tolerance"]:
                advance = 0.0
            ctx["carry_u"] = max(u, min(1.0, u + advance, *progress_limits))
        rotation = turn(ctx["carry_u"])
        shift = delta * ctx["carry_u"]
        if name in ("lower", "support", "clear_right", "square", "release", "retreat", "settle") and (
            not ctx["carry_aborted"]
        ):
            fraction = min(1.0, t_phase / 2.0) if name == "lower" else 1.0
            shift[2] = fraction * (TABLE_TOP_Z + SHELF_HEIGHT - ctx["carry_box_start"][2])
            if name in ("retreat", "settle"):
                shift[2] += 0.05 * (min(1.0, t_phase) if name == "retreat" else 1.0)
        release = name in ("release", "retreat", "settle") and (not ctx["carry_aborted"])
        commands = {
            side: OPEN_WIDTH if release else self._carry_grip(ctx, sig, side, knobs, h.control_dt)
            for side in positions
        }
        targets = {side: ctx["carry_box_start"] + shift + rotation @ relative[side] for side in positions}
        quats = {side: _quat_from_mat(rotation @ _mat_from_quat(ctx["quat"][side])) for side in positions}
        if name in ("clear_right", "square", "release", "retreat", "settle"):
            commands["right"] = OPEN_WIDTH
            lift = min(1.0, t_phase) if name == "clear_right" else 1.0
            targets["right"][2] = TABLE_TOP_Z + SHELF_HEIGHT + relative["right"][2] + 0.06 * lift
            progress = min(1.0, t_phase / 3.0) if name == "square" else 0.0 if name == "clear_right" else 1.0
            left_rotation = turn(ctx["carry_u"] * (1.0 - progress))
            targets["left"] = ctx["carry_box_start"] + shift + left_rotation @ relative["left"]
            quats["left"] = _quat_from_mat(left_rotation @ _mat_from_quat(ctx["quat"]["left"]))
        return self._both(h, ctx, targets, commands, quats)

    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        if name == "prepare":
            postures = {
                "right": (-46.49, 0.82, -26.99, 102.86, -5.6, -26.33, -12.46),
                "left": (16.95, -0.85, 26.74, 93.47, 0.84, 26.87, 31.5),
            }
            fraction = min(1.0, t_phase / 1.5)
            alpha = fraction**2 * (3.0 - 2.0 * fraction)
            target = {}
            for side, angles in postures.items():
                for j, angle in enumerate(angles, 1):
                    joint = f"openarm_{side}_joint{j}"
                    initial = ctx["prepare_start"][h.robot.joint_names.index(joint)]
                    goal = np.radians(angle)
                    target[joint] = float(initial + alpha * (goal - initial))
                for j in (1, 2):
                    joint = f"openarm_{side}_finger_joint{j}"
                    target[joint] = OPEN_WIDTH
            h.unlatch()
            h.select_arm("left")
            ctx["active"] = ctx["side"] = "left"
            return (target, OPEN_WIDTH, None)
        if not ctx["grasp_failed"] and name in (
            "transport",
            "lower",
            "support",
            "clear_right",
            "square",
            "release",
            "retreat",
            "settle",
        ):
            return self._synchronized_carry(h, ctx, name, t_phase, sig, knobs)
        if ctx["grasp_failed"] and name in ("retreat", "settle"):
            return self._dual(h, ctx, sig, np.array([0.0, 0.0, 0.06]), knobs, OPEN_WIDTH)
        if name == "lift":
            height = knobs["lift_height"] * min(1.0, t_phase / float(knobs["lift_s"]))
            return self._dual(h, ctx, sig, np.array([0.0, 0.0, height]), knobs)
        grasps = {side: self._grasp(ctx, side) for side in ("right", "left")}
        pres = {side: grasps[side] - self._blade(side, knobs) * APPROACH_BACKOFF for side in grasps}
        if name in ("home", "approach"):
            for side in grasps:
                self._wrist(h, ctx, side)
            targets = {side: pres[side] + ([0, 0, 0.03] if name == "home" else [0, 0, 0]) for side in pres}
            return self._both(h, ctx, targets, {side: OPEN_WIDTH for side in pres}, ctx["quat"])
        if name == "orient":
            for side in grasps:
                self._wrist(h, ctx, side)
            return self._both(h, ctx, pres, {side: OPEN_WIDTH for side in pres}, ctx["quat"])
        if name == "descend":
            return self._both(h, ctx, grasps, {side: OPEN_WIDTH for side in grasps}, ctx["quat"])
        if name == "close":
            for side in grasps:
                if ctx["grip"][side] <= 0.0:
                    ctx["grip"][side] = float(knobs["close_width"])
                pads = TACTILE_SENSORS[:2] if side == "right" else TACTILE_SENSORS[2:]
                area = min(sig["areas"][pad] for pad in pads)
                if area < CONTACT_AREA:
                    ctx["grip"][side] = max(MIN_WIDTH, ctx["grip"][side] - 0.004 * h.control_dt)
        return self._both(h, ctx, grasps, ctx["grip"], ctx["quat"])

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs):
        if name == "prepare":
            error = max(
                abs(float(h.robot.data.joint_pos[0, h.robot.joint_names.index(joint)]) - value)
                for (joint, value) in target.items()
                if "finger" not in joint
            )
            return t_phase >= 1.7 and error < np.radians(2.0)
        if name in ("home", "approach", "orient", "descend", "close"):
            targets = {
                side: self._grasp(ctx, side) - self._blade(side, knobs) * APPROACH_BACKOFF
                for side in ("right", "left")
            }
            if name == "home":
                targets = {side: pos + [0, 0, 0.03] for (side, pos) in targets.items()}
            elif name in ("descend", "close"):
                targets = {side: self._grasp(ctx, side) for side in targets}
            errors = {}
            angles = {}
            for side in targets:
                h.select_arm(side)
                errors[side] = h.tcp_err(targets[side])
                if name == "orient" and ctx["quat"][side] is not None:
                    angles[side] = np.degrees(
                        2 * np.arccos(min(1.0, abs(np.dot(h.tcp_quat(), ctx["quat"][side]))))
                    )
            h.select_arm("left")
            if name in ("home", "approach"):
                return max(errors.values()) < 0.015
            if name == "orient":
                ctx["orientation_ready"] = (
                    t_phase > 0.3
                    and len(angles) == 2
                    and (max(angles.values()) < 4.0)
                    and (max(errors.values()) < 0.003)
                )
                return ctx["orientation_ready"]
            aligned = all(
                sig[f"{side}_wall_offset"] < 0.003
                and sig[f"{side}_jaw_angle"] < 8.0
                and sig[f"{side}_straddles_wall"]
                for side in ("right", "left")
            )
            if name == "descend":
                ctx["aligned_for_close"] = aligned and max(errors.values()) < 0.003
                return ctx["aligned_for_close"]
            good = aligned and max(errors.values()) < 0.008 and (min(sig["areas"].values()) >= GRASP_MIN_AREA)
            ctx["grasp_good_s"] = ctx["grasp_good_s"] + h.control_dt if good else 0.0
            ctx["both_closed"] = ctx["grasp_good_s"] >= 0.2
            return ctx["both_closed"]
        if name == "lift":
            weak = min(sig["areas"][pad] for pad in TACTILE_SENSORS) < 15.0
            ctx["lift_loss_s"] = ctx["lift_loss_s"] + h.control_dt if weak else 0.0
            if t_phase > 0.25 and ctx["lift_loss_s"] >= 0.5:
                ctx["lift_lost"] = True
                return True
            if sig["corner_clearance"] >= SHELF_HEIGHT + 0.015:
                ready_after = float(knobs["lift_s"]) + 0.25
                return t_phase >= ready_after
            if t_phase >= float(knobs["lift_s"]) + 0.75:
                ctx["lift_lost"] = True
                return True
            return False
        if name == "transport":
            return ctx["carry_u"] >= 0.99 and sig["goal_xy_error"] < 0.015
        if name == "lower":
            return (
                t_phase >= 1.0
                and sig["goal_xy_error"] < 0.018
                and (abs(sig["box_z"] - (TABLE_TOP_Z + SHELF_HEIGHT)) < 0.012)
            )
        if name == "support":
            return t_phase >= 0.5 and abs(sig["box_z"] - (TABLE_TOP_Z + SHELF_HEIGHT)) < 0.004
        if name == "clear_right":
            return t_phase >= 1.2 and sig["right_tcp_z"] > TABLE_TOP_Z + SHELF_HEIGHT + 0.075
        if name == "square":
            error = abs((sig["box_yaw_deg"] + 90.0) % 180.0 - 90.0)
            ctx["square_done"] = error < 5.0
            return t_phase >= 3.5
        if name == "settle":
            return t_phase >= 1.0
        return t_phase >= dict(self.phases)[name] * 0.9

    def next_phase(self, ctx, name):
        if name == "transport" and ctx["carry_u"] < 0.99:
            ctx["carry_aborted"] = True
            return "settle"
        if (
            name == "orient"
            and (not ctx["orientation_ready"])
            or (name == "descend" and (not ctx["aligned_for_close"]))
        ):
            ctx["grasp_failed"] = True
            return "retreat"
        if name == "close" and (not ctx["both_closed"]):
            ctx["grasp_failed"] = True
            return "retreat"
        if name == "lift" and ctx["lift_lost"]:
            ctx["grasp_failed"] = True
            return "retreat"
        return None

    def build_trace(self, h, ctx, hist):
        return EpisodeTrace(
            t=np.asarray(hist["t"]),
            pad_area={p: np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={"parts_box": np.asarray(hist["box_pos"])},
            extra={"escaped": np.asarray(hist["escaped"])},
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(trace, goal_xy=GOAL_XY, goal_surface_z=TABLE_TOP_Z + SHELF_HEIGHT)

    def episode_extra(self, ctx, hist):
        return {
            "square_done": ctx["square_done"],
            "grip_adjustments": ctx["adjustments"],
            "spilled": bool(ctx["spilled"]),
            "first_spill_object": ctx["first_spill_object"],
            "bilateral_grasp_confirmed": ctx["both_closed"],
            "lift_lost": ctx["lift_lost"],
        }


TASK = LoosePartsBoxTransferTask()
