"""Dual-arm expert for carrying a tray with three glass cups."""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from sim.criteria.glassware_tray_transfer import cup_standing_on_tray
from sim.criteria.glassware_tray_transfer import verdict as _verdict
from sim.criteria.probes import tilt_deg
from sim.criteria.trace import EpisodeTrace
from sim.dims.glassware_tray_transfer import (
    CUP_NAMES,
    CUP_OFFSETS,
    HANDLE_HEIGHT,
    HANDLE_HOLE_HEIGHT,
    HANDLE_HOLE_Z,
    TRAY_THICKNESS,
    TRAY_WIDTH,
)
from sim.envs.bimanual_tactile import REAL_HOME_JOINT_POS, TACTILE_SENSORS, gel_gap
from sim.envs.glassware_tray_transfer import (
    GOAL_XY,
    SPAWN_CLEARANCE,
    START_XY,
    SUPPORT_SIZE,
    SUPPORT_TOP_Z,
    SURFACE_Z,
    GlasswareTrayTransferSceneCfg,
)
from sim.tactile import apply_sdf_collision
from sim.tactile.quantities import quat_apply
from sim.tasks.scene_utils import top_down_grasp_quat
from sim.tasks.spec import Knob, Task

__all__ = ["TASK", "GlasswareTrayTransferTask"]

SIDE_PADS = {"right": TACTILE_SENSORS[:2], "left": TACTILE_SENSORS[2:]}
GRASP_Y = {"right": -(TRAY_WIDTH / 2 - 0.0045), "left": TRAY_WIDTH / 2 - 0.0045}
GRASP_Z = SURFACE_Z + (HANDLE_HEIGHT + HANDLE_HOLE_Z + HANDLE_HOLE_HEIGHT / 2) / 2
OPEN_WIDTH = 0.026
MIN_GRIP = 0.0065
CONTACT_AREA = 5.0
GRIP_RATE = 0.012
EXTRA_SQUEEZE = 0.001
APPROACH_BACKOFF = 0.100
PREP_RISE = 0.035
ORIENT_TOL_DEG = 5.0
PICK_HEIGHT = 0.075
CARRY_RATE = 0.025
LIFT_RATE = 0.018
LOWER_RATE = 0.018
RELEASE_RATE = 0.035
CARRY_XY = (GOAL_XY[0] + 0.010, GOAL_XY[1])
# Wrist joint targets for the outward approach from Home.
OUTWARD_WRIST_DEG = {
    "right": (89.84, 44.00, -11.94),
    "left": (-89.83, -44.00, 12.02),
}
OUTSIDE_HANDLE_GAP = 0.045
OUTSIDE_HOVER_HEIGHT = 0.060


def _pose(h, name):
    data = h.scene[name].data
    return (
        data.root_pos_w[0].detach().cpu().numpy().copy(),
        data.root_quat_w[0].detach().cpu().numpy().copy(),
    )


def _tcp(h, side):
    return h.robot.data.body_state_w[0, h.ee_body(side), :3].detach().cpu().numpy().copy()


def _quat_error_deg(current, target):
    d = abs(float(np.dot(current, target)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def _contact_load(points, quaternions, normal_force, shear_force, inward_normal):
    """Return the object's contact load and its horizontal first moment."""
    local = torch.cat([shear_force, normal_force.unsqueeze(-1)], dim=-1)
    force = -quat_apply(quaternions, local).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    inward = torch.as_tensor(inward_normal, dtype=force.dtype, device=force.device)
    vertical = force[..., 2]
    return {
        "force": force.sum(dim=0),
        "torque": torch.cross(points, force, dim=-1).sum(dim=0),
        "normal": (force * inward).sum(),
        "vertical": vertical.sum(),
        "moment_xy": (points[..., :2] * vertical.unsqueeze(-1)).sum(dim=0),
    }


def _contact_plane(points, depth, inward_normal, shape):
    """Estimate a flat grasp surface from the interior of a tactile depth patch."""
    mask = (depth.reshape(shape) > 2e-5).to(depth.dtype)[None, None]
    interior = -torch.nn.functional.max_pool2d(-mask, 15, stride=1, padding=7)
    selected = interior.reshape(-1) > 0.5
    count = int(selected.sum())
    if count < 64:
        return None
    inward = torch.as_tensor(inward_normal, dtype=points.dtype, device=points.device)
    surface = points[selected] - depth[selected, None] * inward
    surface = surface[:: max(1, len(surface) // 2048)]
    for _ in range(2):
        centered = surface - surface.mean(dim=0)
        values, vectors = torch.linalg.eigh(centered.T @ centered / len(surface))
        normal = vectors[:, 0]
        residual = (centered @ normal).abs()
        surface = surface[residual <= torch.quantile(residual, 0.85) + 1e-8]
    if normal[2] < 0:
        normal = -normal
    if float(normal[2]) < 0.8 or float(values[1]) < 1e-7:
        return None
    return normal, float(torch.sqrt((residual * residual).mean())), count


class GlasswareTrayTransferTask(Task):
    name = "glassware_tray_transfer"
    instruction = "Carry the wooden tray to the placemat without tipping the three glass cups."
    report_name = "glassware_tray_transfer_scripted.json"
    handover_phase = None
    horizon_s = 60.0
    scene_cam = ((0.90, -0.52, 0.68), (0.34, 0.0, SURFACE_Z + 0.06))
    knobs = {
        "start_jitter": Knob(0.005, "initial tray position jitter, metres"),
        "cup_jitter": Knob(0.012, "initial position jitter per cup, metres"),
        "outward_grasp": Knob(1, "move outward from the shared Home pose (0 or 1)", int),
        "use_tactile": Knob(1, "use tactile feedback for grip control (0 or 1)", int),
        "use_balance": Knob(1, "balance from tactile pressure distribution (0 or 1)", int),
    }
    phases = [
        ("home", 8.0),
        ("above", 5.0),
        ("orient", 8.0),
        ("entry", 4.0),
        ("approach", 6.0),
        ("grasp", 3.0),
        ("lift", 8.0),
        ("carry", 16.0),
        ("lower", 5.0),
        ("rest", 1.0),
        ("release", 2.0),
        ("retreat", 3.0),
        ("clear", 2.0),
        ("settle", 1.5),
    ]
    transport_phases = ("lift", "carry", "lower", "rest", "release", "retreat", "clear", "settle")
    joint_phases = tuple(name for name, _ in phases)
    completion_on_success = True
    settle_seconds = 0.5

    def make_scene_cfg(self, knobs):
        return GlasswareTrayTransferSceneCfg(num_envs=1, env_spacing=3.0)

    def apply_collision(self, h):
        return {"tray": apply_sdf_collision(f"{h.env_path}/tray", sdf_resolution=128)}

    def sample_init_state(self, robot, rng, state_id, knobs):
        x, y = (
            float(value)
            for value in np.asarray(START_XY) + rng.uniform(-knobs["start_jitter"], knobs["start_jitter"], 2)
        )
        objects = {
            "tray": [x, y, SURFACE_Z + SPAWN_CLEARANCE, 1.0, 0.0, 0.0, 0.0],
            "goal_support": [*GOAL_XY, SURFACE_Z + SUPPORT_SIZE[2] / 2, 1.0, 0.0, 0.0, 0.0],
        }
        for name, offset in zip(CUP_NAMES, CUP_OFFSETS, strict=True):
            cx, cy = np.asarray(offset) + rng.uniform(-knobs["cup_jitter"], knobs["cup_jitter"], 2)
            objects[name] = [
                x + float(cx),
                y + float(cy),
                SURFACE_Z + TRAY_THICKNESS + 2 * SPAWN_CLEARANCE,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        return dict(id=state_id, robot=copy.deepcopy(robot), objects=objects, task_vars={"side": "right"})

    def validate_init_state(self, state):
        if state["task_vars"].get("side") != "right":
            raise ValueError("glassware_tray_transfer requires side=right")

    def init_state_objects(self):
        return ("tray", *CUP_NAMES, "goal_support")

    def snapshot_init_state(self, h, ctx):
        return {"side": "right"}

    def begin_restored_episode(self, h, state, knobs):
        self.validate_init_state(state)
        h.unlatch()
        h.select_arm("right")
        p, _ = _pose(h, "tray")
        names = h.robot.joint_names
        for side in SIDE_PADS:
            measured = np.array(
                [state["robot"]["joint_pos"][names.index(f"openarm_{side}_joint{i}")] for i in range(1, 8)]
            )
            expected = np.array([REAL_HOME_JOINT_POS[f"openarm_{side}_joint{i}"] for i in range(1, 8)])
            if np.max(np.abs(np.rad2deg(measured - expected))) > 3.0:
                raise ValueError("glassware_tray_transfer init state must start at the shared Home pose")
        return {
            "side": "right",
            "start": p.copy(),
            "grip": {"right": 0.0, "left": 0.0},
            "quat": {"right": None, "left": None},
            "touch": {"right": None, "left": None},
            "squeeze_start": {"right": None, "left": None},
            "grasp_offset": {"right": None, "left": None},
            "command_center": None,
            "phase_goal": None,
            "approach_target": {"right": None, "left": None},
            "pose_target": {"right": None, "left": None},
            "tactile_adjustments": 0,
            "use_tactile": bool(knobs["use_tactile"]),
            "outward_grasp": bool(knobs["outward_grasp"]),
            "balance_angles": np.zeros(2),
            "balance_updates": 0,
            "pressure_tilt_filtered": np.zeros(2),
            "balance_integral": np.zeros(2),
            "contact_deflection": np.zeros(2),
            "deflection_reference": None,
            "balance_target_tilt": np.zeros(2),
        }

    def signals(self, h, ctx):
        tray_pos, tray_quat = _pose(h, "tray")
        areas = h.pad_areas()
        cup_poses = {name: _pose(h, name) for name in CUP_NAMES}
        standing = {
            name: cup_standing_on_tray(pos, quat, tray_pos, tray_quat)[0]
            for name, (pos, quat) in cup_poses.items()
        }
        loads = self._tactile_loads(h)
        return {
            **loads,
            "balance_angles": ctx["balance_angles"].copy(),
            "contact_deflection": ctx["contact_deflection"].copy(),
            "balance_target_tilt": ctx["balance_target_tilt"].copy(),
            "tcp_right": _tcp(h, "right"),
            "tcp_left": _tcp(h, "left"),
            "gel_right": h.gel_centres("right"),
            "gel_left": h.gel_centres("left"),
            "wrist_error_deg": {
                side: _quat_error_deg(
                    h.robot.data.body_state_w[0, h.ee_body(side), 3:7].detach().cpu().numpy(),
                    ctx["quat"][side],
                )
                if ctx["quat"][side] is not None
                else 0.0
                for side in SIDE_PADS
            },
            "areas": areas,
            "tray_pos": tray_pos,
            "tray_quat": tray_quat,
            **{f"{name}_pos": pos for name, (pos, _) in cup_poses.items()},
            **{f"{name}_quat": quat for name, (_, quat) in cup_poses.items()},
            "standing_cups": float(sum(standing.values())),
            "tray_tilt_deg": tilt_deg(tray_quat),
            "right_contact": float(min(areas[p] for p in SIDE_PADS["right"])),
            "left_contact": float(min(areas[p] for p in SIDE_PADS["left"])),
        }

    @staticmethod
    def _grasp_point(ctx, side):
        return np.array([ctx["start"][0], ctx["start"][1] + GRASP_Y[side], GRASP_Z])

    @staticmethod
    def _approach_direction(side):
        return np.array([0.8660254, 0.5 if side == "right" else -0.5, 0.0])

    def _orient(self, h, ctx, side):
        if ctx["quat"][side] is None and gel_gap(h.grip_measured()) >= 0.010:
            jaw = h.jaw_axis()
            approach = self._approach_direction(side)
            jaw_target = np.array([0.0, 0.0, -1.0 if side == "right" else 1.0])
            if ctx["outward_grasp"]:
                jaw_target = -jaw_target
            ctx["quat"][side] = top_down_grasp_quat(
                h.tcp_quat(),
                jaw,
                h.approach_axis(),
                jaw_target=jaw_target,
                approach_target=approach,
            )
        return ctx["quat"][side]

    @staticmethod
    def _tactile_loads(h):
        normal, vertical, moments, alignment = {}, {}, {}, {}
        forces, torques = {}, {}
        fitted_angles, residuals = [], {}
        normals = {}
        for side, pads in SIDE_PADS.items():
            axis = h.jaw_axis(side)
            for pad, sign in zip(pads, (1.0, -1.0), strict=True):
                normals[pad] = np.asarray(axis) * sign if axis is not None else np.zeros(3)
        for pad in TACTILE_SENSORS:
            data = h.sensors[pad].data
            nf = getattr(data, "tactile_normal_force", None)
            shear = getattr(data, "tactile_shear_force", None)
            points = data.tactile_points_pos_w
            quats = getattr(data, "tactile_points_quat_w", None)
            normal[pad] = vertical[pad] = alignment[pad] = 0.0
            moments[pad] = [0.0, 0.0]
            forces[pad] = torques[pad] = [0.0, 0.0, 0.0]
            if any(value is None for value in (nf, shear, points, quats)):
                continue
            load = _contact_load(points[0], quats[0], nf[0], shear[0], normals[pad])
            normal[pad] = float(load["normal"])
            vertical[pad] = float(load["vertical"])
            moments[pad] = load["moment_xy"].detach().cpu().tolist()
            forces[pad] = load["force"].detach().cpu().tolist()
            torques[pad] = load["torque"].detach().cpu().tolist()
            alignment[pad] = float(normals[pad][2])
            plane = _contact_plane(points[0], data.penetration_depth[0], normals[pad], (h.rows, h.cols))
            if plane is not None and plane[1] < 0.0002:
                n = plane[0].detach().cpu().numpy()
                fitted_angles.append([np.arctan2(-n[1], n[2]), np.arctan2(n[0], np.hypot(n[1], n[2]))])
                residuals[pad] = plane[1]

        return {
            "pressure_tilt": np.median(fitted_angles, axis=0) if fitted_angles else np.zeros(2),
            "pressure_tilt_valid": len(fitted_angles) >= 2,
            "pressure_fit_residual": residuals,
            "pad_force_world": forces,
            "pad_torque_world": torques,
            "pad_normal_load": normal,
            "pad_vertical_load": vertical,
            "pad_load_moment_xy": moments,
            "pad_vertical_alignment": alignment,
        }

    def _balance(self, h, ctx, sig, enabled, integral_gain, phase):
        if not enabled or not sig["pressure_tilt_valid"]:
            desired = np.zeros(2)
            ctx["balance_integral"] *= 0.95
        else:
            previous = ctx["pressure_tilt_filtered"].copy()
            alpha = h.control_dt / (0.04 + h.control_dt)
            ctx["pressure_tilt_filtered"] += alpha * (sig["pressure_tilt"] - previous)
            wrist_angles = []
            for side in SIDE_PADS:
                current = h.robot.data.body_state_w[0, h.ee_body(side), 3:7].detach().cpu().numpy()
                delta = (
                    Rotation.from_quat(np.roll(current, -1))
                    * Rotation.from_quat(np.roll(ctx["quat"][side], -1)).inv()
                )
                wrist_angles.append(delta.as_euler("xyz")[:2])
            deflection = sig["pressure_tilt"] - np.mean(wrist_angles, axis=0)
            ctx["contact_deflection"] += (
                h.control_dt / (0.2 + h.control_dt) * (deflection - ctx["contact_deflection"])
            )
            if phase == "carry":
                if ctx["deflection_reference"] is None:
                    ctx["deflection_reference"] = ctx["contact_deflection"].copy()
                ctx["balance_target_tilt"] = np.clip(
                    -0.35 * (ctx["contact_deflection"] - ctx["deflection_reference"]), -0.02, 0.02
                )
            else:
                ctx["balance_target_tilt"] *= 0.9
            error = ctx["pressure_tilt_filtered"] - ctx["balance_target_tilt"]
            velocity = (ctx["pressure_tilt_filtered"] - previous) / h.control_dt
            ctx["balance_integral"] = np.clip(
                ctx["balance_integral"] - integral_gain * error * h.control_dt, -0.14, 0.14
            )
            desired = np.clip(ctx["balance_integral"] - 2.5 * error - 0.20 * velocity, -0.14, 0.14)
        change = np.clip(desired - ctx["balance_angles"], -0.40 * h.control_dt, 0.40 * h.control_dt)
        ctx["balance_angles"] += change
        if np.linalg.norm(change) > 1e-5:
            ctx["balance_updates"] += 1
        roll, pitch = ctx["balance_angles"]
        return Rotation.from_euler("xy", [roll, pitch])

    def _grip_command(self, h, ctx, sig, side):
        current = ctx["grip"][side]
        if not ctx["use_tactile"]:
            return max(MIN_GRIP, current - GRIP_RATE * h.control_dt)
        if ctx["touch"][side] is None:
            if min(sig["areas"][p] for p in SIDE_PADS[side]) > CONTACT_AREA:
                ctx["touch"][side] = h.grip_measured()
            else:
                return max(MIN_GRIP, current - GRIP_RATE * h.control_dt)
        return max(MIN_GRIP, ctx["touch"][side] - EXTRA_SQUEEZE)

    @staticmethod
    def _solve_pose(h, side, target, quat):
        previous = h.ch["side"]
        h.select_arm(side)
        try:
            chain = h.ch
            ids = chain["arm"].joint_ids
            q = h.robot.data.joint_pos[0, ids].detach().cpu().numpy()
            limits = h.robot.data.soft_joint_pos_limits[0, ids].detach().cpu().numpy()
            jac = h.robot.root_physx_view.get_jacobians()[0, chain["jac_idx"]][:, ids].detach().cpu().numpy()
            position = np.asarray(target) - h.tcp_pos()
            position *= min(1.0, 0.010 / max(np.linalg.norm(position), 1e-9))
            if quat is None:
                matrix = jac[:3] * 8.0
                error = position * 8.0
            else:
                now = Rotation.from_quat(np.roll(h.tcp_quat(), -1))
                goal = Rotation.from_quat(np.roll(quat, -1))
                angle = (goal * now.inv()).as_rotvec()
                angle *= min(1.0, 0.12 / max(np.linalg.norm(angle), 1e-9))
                matrix = jac.copy()
                matrix[:3] *= 8.0
                error = np.r_[position * 8.0, angle]
            matrix = np.vstack([matrix, 0.03 * np.eye(7)])
            error = np.r_[error, np.zeros(7)]
            lower = np.maximum(limits[:, 0] - q, -0.10)
            upper = np.minimum(limits[:, 1] - q, 0.10)
            lower = np.minimum(lower, upper - 1e-7)
            delta = lsq_linear(matrix, error, bounds=(lower, upper), tol=1e-5).x
            return {f"openarm_{side}_joint{i + 1}": float(q[i] + delta[i]) for i in range(7)}
        finally:
            h.select_arm(previous)

    def _step_center(self, ctx, goal, rate, dt):
        want = np.asarray(goal, dtype=float)
        current = ctx["command_center"]
        if current is None:
            current = np.asarray(ctx["start"], dtype=float)
        delta = want - current
        length = float(np.linalg.norm(delta))
        step = rate * dt
        ctx["command_center"] = current + delta * min(1.0, step / length) if length > 0 else want
        return ctx["command_center"]

    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        if t_phase == 0.0:
            h.latch()
            ctx["phase_goal"] = None
            ctx["approach_target"] = {"right": None, "left": None}
        if name in self.transport_phases:
            h.select_arm("right")
            if name == "lift":
                goal = ctx["start"].copy()
                goal[2] = SURFACE_Z + PICK_HEIGHT
                rate = LIFT_RATE
            elif name == "carry":
                goal = np.array([*CARRY_XY, SURFACE_Z + PICK_HEIGHT])
                rate = CARRY_RATE
            else:
                goal = np.array([*CARRY_XY, SUPPORT_TOP_Z])
                rate = LOWER_RATE
            if name in ("release", "retreat", "clear", "settle"):
                center = ctx["command_center"] if ctx["command_center"] is not None else goal
            else:
                center = self._step_center(
                    ctx, goal, rate * min(1.0, t_phase / 0.6) if name == "lift" else rate, h.control_dt
                )
            joints = {}
            rotation = self._balance(
                h,
                ctx,
                sig,
                bool(knobs["use_balance"]) and name in ("lift", "carry", "lower"),
                np.array([3.5, 12.0]) if name == "carry" else 3.5,
                name,
            )
            valid_offsets = [v for v in ctx["grasp_offset"].values() if v is not None]
            pivot = np.array([0.0, 0.0, np.mean([v[2] for v in valid_offsets]) if valid_offsets else 0.0])
            for side in ("right", "left"):
                offset = ctx["grasp_offset"][side]
                if offset is None:
                    ctx["grip"][side] = OPEN_WIDTH
                    continue
                target = np.asarray(center) + pivot + rotation.apply(offset - pivot)
                if name in ("retreat", "clear"):
                    target = target + np.array(
                        [0.0, 0.09 if side == "left" else -0.09, 0.060 if name == "clear" else 0.0]
                    )
                if name == "settle":
                    target = _tcp(h, side)
                base = Rotation.from_quat(np.roll(ctx["quat"][side], -1))
                quat = np.roll((rotation * base).as_quat(), 1)
                joints.update(self._solve_pose(h, side, target, quat))
                if name == "release":
                    ctx["grip"][side] = min(OPEN_WIDTH, ctx["grip"][side] + RELEASE_RATE * h.control_dt)
                elif name in ("retreat", "clear", "settle"):
                    ctx["grip"][side] = OPEN_WIDTH
                elif ctx["use_tactile"] and sig[f"{side}_contact"] < CONTACT_AREA:
                    ctx["grip"][side] = max(MIN_GRIP, ctx["grip"][side] - 0.0002)
                    ctx["tactile_adjustments"] += 1
            joints["openarm_left_finger_joint1"] = ctx["grip"]["left"]
            joints["openarm_left_finger_joint2"] = ctx["grip"]["left"]
            ctx["phase_goal"] = goal
            return joints, ctx["grip"]["right"], None

        joints = {}
        for side in SIDE_PADS:
            joints.update(self._grasp_target(h, ctx, name, sig, side))
            for finger in (1, 2):
                joints[f"openarm_{side}_finger_joint{finger}"] = ctx["grip"][side]
        h.select_arm("right")
        return joints, ctx["grip"]["right"], None

    def _grasp_target(self, h, ctx, name, sig, side):
        h.select_arm(side)
        if ctx["outward_grasp"] and name == "home":
            ctx["grip"][side] = OPEN_WIDTH
            return {
                f"openarm_{side}_joint{i}": float(np.deg2rad(OUTWARD_WRIST_DEG[side][i - 5]))
                for i in range(5, 8)
            }
        grasp = self._grasp_point(ctx, side)
        if ctx["outward_grasp"] and name in ("above", "orient", "entry", "approach"):
            ctx["grip"][side] = OPEN_WIDTH
            self._orient(h, ctx, side)
            target = grasp.copy()
            target[1] += OUTSIDE_HANDLE_GAP * (-1 if side == "right" else 1)
            if name in ("above", "orient"):
                target[2] += OUTSIDE_HOVER_HEIGHT
            elif name == "approach":
                if ctx["approach_target"][side] is None:
                    ctx["approach_target"][side] = h.tcp_pos().copy()
                delta = grasp - ctx["approach_target"][side]
                length = float(np.linalg.norm(delta))
                step = 0.075 * h.control_dt
                ctx["approach_target"][side] += delta * min(1.0, step / length) if length > 0 else 0.0
                target = ctx["approach_target"][side].copy()
            ctx["pose_target"][side] = target.copy()
            return self._solve_pose(h, side, target, ctx["quat"][side])
        entry = grasp - APPROACH_BACKOFF * self._approach_direction(side)
        pre = entry + np.array([0.0, 0.0, 0.080])
        if name == "home":
            ctx["grip"][side] = OPEN_WIDTH if ctx["outward_grasp"] else 0.0
            target = pre + np.array([0.0, 0.0, PREP_RISE])
        elif name == "above":
            ctx["grip"][side] = OPEN_WIDTH
            target = pre
        elif name == "orient":
            ctx["grip"][side] = OPEN_WIDTH
            target = pre
            self._orient(h, ctx, side)
        elif name == "entry":
            ctx["grip"][side] = OPEN_WIDTH
            target = entry
        elif name == "approach":
            ctx["grip"][side] = OPEN_WIDTH
            if ctx["approach_target"][side] is None:
                ctx["approach_target"][side] = h.tcp_pos().copy()
            delta = grasp - ctx["approach_target"][side]
            length = float(np.linalg.norm(delta))
            step = 0.075 * h.control_dt
            ctx["approach_target"][side] += delta * min(1.0, step / length) if length > 0 else 0.0
            target = ctx["approach_target"][side].copy()
        else:
            ctx["grip"][side] = self._grip_command(h, ctx, sig, side)
            target = grasp.copy()
            if name == "grasp" and ctx["outward_grasp"]:
                # Follow small tray shifts while the fingers close.
                target[:2] += np.clip(sig["tray_pos"][:2] - ctx["start"][:2], -0.008, 0.008)
        if ctx["outward_grasp"] and name in ("home", "above"):
            self._orient(h, ctx, side)
        quat = None if name in ("home", "above") and not ctx["outward_grasp"] else ctx["quat"][side]
        ctx["pose_target"][side] = np.asarray(target).copy()
        return self._solve_pose(h, side, target, quat)

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs):
        if name in self.transport_phases:
            if name == "lift":
                return (
                    float(np.linalg.norm(sig["tray_pos"] - ctx["phase_goal"])) < 0.018
                    and float(np.linalg.norm(ctx["command_center"] - ctx["phase_goal"])) < 0.005
                )
            if name == "carry":
                return (
                    float(np.linalg.norm(ctx["command_center"] - ctx["phase_goal"])) < 0.005
                    and float(np.linalg.norm(sig["tray_pos"][:2] - ctx["phase_goal"][:2])) < 0.030
                )
            if name == "lower":
                return (
                    ctx["command_center"][2] <= SUPPORT_TOP_Z + 0.005
                    and sig["tray_pos"][2] <= SUPPORT_TOP_Z + 0.005
                )
            if name == "rest":
                return t_phase > 0.6
            if name == "release":
                return all(ctx["grip"][side] >= OPEN_WIDTH - 1e-6 for side in SIDE_PADS) and (
                    not ctx["use_tactile"] or all(sig[f"{side}_contact"] < CONTACT_AREA for side in SIDE_PADS)
                )
            if name in ("retreat", "clear"):
                return t_phase > 1.5
            return t_phase > 1.0
        ready = [self._grasp_done(h, ctx, name, t_phase, sig, side) for side in SIDE_PADS]
        if name == "home" and ctx["outward_grasp"] and all(ready):
            for side in SIDE_PADS:
                h.select_arm(side)
                self._orient(h, ctx, side)
        h.select_arm("right")
        return all(ready)

    def _grasp_done(self, h, ctx, name, t_phase, sig, side):
        h.select_arm(side)
        if name == "home" and ctx["outward_grasp"]:
            names = h.robot.joint_names
            indices = range(5, 8)
            measured = np.rad2deg(
                np.array(
                    [
                        h.robot.data.joint_pos[0, names.index(f"openarm_{side}_joint{i}")].item()
                        for i in indices
                    ]
                )
            )
            target_angles = np.array([OUTWARD_WRIST_DEG[side][i - 5] for i in indices])
            return t_phase >= 0.5 and np.max(np.abs(measured - target_angles)) < 3.0
        if name in ("home", "above"):
            return h.tcp_err(ctx["pose_target"][side]) < 0.015
        if name == "orient":
            tolerance_deg = 13.0 if ctx["outward_grasp"] else ORIENT_TOL_DEG
            return (
                ctx["quat"][side] is not None
                and t_phase > 0.25
                and _quat_error_deg(h.tcp_quat(), ctx["quat"][side]) < tolerance_deg
                and h.tcp_err(ctx["pose_target"][side]) < 0.015
            )
        if name == "entry":
            return h.tcp_err(ctx["pose_target"][side]) < 0.008
        if name == "approach":
            return h.tcp_err(self._grasp_point(ctx, side)) < (0.006 if ctx["outward_grasp"] else 0.009)
        if name == "grasp":
            if not ctx["use_tactile"]:
                if t_phase >= 2.0:
                    ctx["grasp_offset"][side] = h.tcp_pos() - sig["tray_pos"]
                    return True
                return False
            contact = sig[f"{side}_contact"] > CONTACT_AREA
            if contact and t_phase > 0.25:
                if ctx["squeeze_start"][side] is None:
                    ctx["squeeze_start"][side] = t_phase
                if ctx["touch"][side] is None:
                    ctx["touch"][side] = h.grip_measured()
                    ctx["grip"][side] = max(MIN_GRIP, ctx["touch"][side] - EXTRA_SQUEEZE)
                if t_phase - ctx["squeeze_start"][side] >= 0.4:
                    ctx["grasp_offset"][side] = h.tcp_pos() - sig["tray_pos"]
                    return True
            return False
        return False

    def next_phase(self, ctx, name):
        if name == "grasp" and any(offset is None for offset in ctx["grasp_offset"].values()):
            return "release"
        return None

    def build_trace(self, h, ctx, hist):
        return EpisodeTrace(
            t=np.asarray(hist["t"]),
            pad_area={p: np.asarray([sample[p] for sample in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={
                "tray": np.asarray(hist["tray_pos"]),
                **{name: np.asarray(hist[f"{name}_pos"]) for name in CUP_NAMES},
            },
            object_quat={
                "tray": np.asarray(hist["tray_quat"]),
                **{name: np.asarray(hist[f"{name}_quat"]) for name in CUP_NAMES},
            },
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(
            trace,
            goal_xy=GOAL_XY,
            goal_z=SUPPORT_TOP_Z,
            support_size=SUPPORT_SIZE[:2],
            right_pads=SIDE_PADS["right"],
            left_pads=SIDE_PADS["left"],
        )

    def episode_extra(self, ctx, hist):
        return {
            "tactile_adjustments": ctx["tactile_adjustments"],
            "balance_updates": ctx["balance_updates"],
            "max_balance_angle_deg": float(np.rad2deg(np.max(np.abs(hist.get("balance_angles", [0.0]))))),
            "max_tray_tilt_deg": float(max(hist.get("tray_tilt_deg", [0.0]))),
            "standing_cups_final": hist["standing_cups"][-1] if hist.get("standing_cups") else 0.0,
        }


TASK = GlasswareTrayTransferTask()
