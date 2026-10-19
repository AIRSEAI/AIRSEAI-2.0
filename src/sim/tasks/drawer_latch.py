# SPDX-License-Identifier: Apache-2.0
"""Pull the tool drawer until its hidden detent holds it open."""

from __future__ import annotations

import copy
import math
from collections import deque
from math import isfinite
from statistics import median

import numpy as np
import torch

from sim.criteria.drawer_latch import verdict as _verdict
from sim.criteria.trace import EpisodeTrace
from sim.dims.drawer_latch import (
    DETENT_RANGE,
    DRAWER_FRONT_X,
    HANDLE_CENTER_Z,
    HANDLE_STANDOFF,
    MAX_EXTENSION,
    YAW_CENTER_DEG,
    YAW_RANGE_DEG,
)
from sim.envs.bimanual_tactile import TACTILE_SENSORS
from sim.envs.drawer_latch import (
    CASE_XY,
    CUE_LEAD_RANGE,
    DRAWER_Z,
    JOINT_NAME,
    MECHANISM_VERSION,
    PREVIEW,
    DrawerLatchSceneCfg,
    DrawerMechanism,
    create_joint,
    reset_drive,
    set_drive,
    set_joint_frame,
)
from sim.tactile import apply_sdf_collision
from sim.tactile.quantities import quat_apply
from sim.tasks.scene_utils import align_frame_quat
from sim.tasks.spec import Knob, RetryReset, Task

ENV = "/World/envs/env_0"
SETTLE_STEPS = 30
DT = 1.0 / 60.0
HANDLE_OFFSET_X = DRAWER_FRONT_X - HANDLE_STANDOFF
PREGRASP_BACKOFF = 0.100
OPEN_WIDTH = 0.040
RELEASE_WIDTH = 0.040
CLOSE_WIDTH = 0.004
PULL_SPEED = 0.020
GRASP_CONTACT_MM2 = 5.0
GRASP_TCP_DEPTH_OFFSET = 0.010
GRASP_LOCAL_X = HANDLE_OFFSET_X + GRASP_TCP_DEPTH_OFFSET
ARM_PADS = {"right": TACTILE_SENSORS[:2], "left": TACTILE_SENSORS[2:]}
RETREAT_DISTANCE = 0.100
RETREAT_SPEED = 0.040
RETREAT_HEIGHT = 0.060
RETREAT_CLEARANCE = 0.080


EXPERT_VERSION = "tactile_baseline_confirm"


class TactileStop:
    def __init__(self, rise=50.0, confirm=2):
        self.rise = rise
        self.confirm = confirm
        self.baseline = []
        self.samples = deque(maxlen=3)
        self.stability = deque(maxlen=7)
        self.streak = 0
        self.low_streak = 0
        self.triggered = False
        self.ready = False
        self.rejected_count = 0

    def update(self, signed_load, hand_travel, *, enabled=True):
        if not isfinite(signed_load) or not isfinite(hand_travel):
            self.samples.clear()
            self.stability.clear()
            self.streak = 0
            self.triggered = self.ready = False
            return False
        if 0.015 <= hand_travel < 0.035 and not self.triggered:
            self.baseline.append((hand_travel, signed_load))
        if hand_travel < 0.035 or len(self.baseline) < 6 or not enabled:
            self.samples.clear()
            self.stability.clear()
            self.streak = 0
            self.triggered = self.ready = False
            return False
        # The normal sliding load slowly decreases as the fingers move. A
        # per-episode offset prevents contact preload from masquerading as cue.
        x_ref = median(x for x, _ in self.baseline)
        load_ref = median(load + 600.0 * (x - x_ref) for x, load in self.baseline)
        residual = signed_load - (load_ref - 600.0 * (hand_travel - x_ref))
        self.samples.append(residual)
        if len(self.samples) < 3:
            return False
        filtered = median(self.samples)
        if not self.triggered:
            self.streak = self.streak + 1 if filtered >= self.rise else 0
            if self.streak >= self.confirm:
                self.triggered = True
                self.stability.clear()
                self.low_streak = 0
            return self.triggered
        self.low_streak = self.low_streak + 1 if filtered < 20.0 else 0
        if self.low_streak >= 3:
            self.triggered = self.ready = False
            self.streak = 0
            self.stability.clear()
            self.rejected_count += 1
            return False
        self.stability.append((hand_travel, filtered))
        self.ready = (
            len(self.stability) == self.stability.maxlen
            and max(x for x, _ in self.stability) - min(x for x, _ in self.stability) <= 0.0005
            and max(f for _, f in self.stability) - min(f for _, f in self.stability) <= 25.0
            and filtered >= 20.0
        )
        return True


class DrawerLatchTask(Task):
    name = "drawer_latch"
    instruction = (
        "pull the tool drawer until it settles into place, then release it and leave it open for use"
    )
    report_name = "drawer_latch_scripted.json"
    knobs = {
        "detent_m": Knob(0.0, "hidden hold-open position; 0 samples one per episode"),
        "tactile_stop": Knob(1.0, "enable tactile stopping; 0 is the stopping-signal ablation"),
    }
    phases = [
        ("home", 4.0),
        ("pregrasp", 4.0),
        ("orient", 3.0),
        ("approach", 5.0),
        ("close", 2.0),
        ("pull", 10.0),
        ("seat", 3.0),
        ("release", 1.5),
        ("retreat", 4.0),
        ("verify", 1.5),
    ]
    handover_phase = None
    horizon_s = 40.0
    completion_on_success = True
    settle_seconds = SETTLE_STEPS * DT
    scene_cam = (PREVIEW.eye, PREVIEW.look_at)

    def make_scene_cfg(self, knobs):
        return DrawerLatchSceneCfg(num_envs=1, env_spacing=3.0)

    def apply_collision(self, h, *, env_path=ENV):
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        robot = stage.GetPrimAtPath(f"{env_path}/Robot")

        def bodies(prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                yield prim
            for child in prim.GetChildren():
                yield from bodies(child)

        def side_of(prim):
            top = prim.GetPath().MakeRelativePath(robot.GetPath()).pathString.split("/")[0]
            return next((side for side in ("right", "left") if top.startswith(f"openarm_{side}_")), None)

        links = list(bodies(robot))
        for own in links:
            own_side = side_of(own)
            api = (
                UsdPhysics.FilteredPairsAPI(own)
                if own.HasAPI(UsdPhysics.FilteredPairsAPI)
                else UsdPhysics.FilteredPairsAPI.Apply(own)
            )
            rel = api.CreateFilteredPairsRel()
            targets = set(rel.GetTargets())
            for other in links:
                other_side = side_of(other)
                same_arm = own_side is not None and own_side == other_side
                idle_to_body = {own_side, other_side} == {"left", None}
                if own != other and (same_arm or idle_to_body) and other.GetPath() not in targets:
                    rel.AddTarget(other.GetPath())
        meshes = apply_sdf_collision(f"{env_path}/drawer")
        if len(meshes) != 8 or not meshes[0].endswith("/handle"):
            raise RuntimeError(f"drawer tactile SDF requires the handle as the first of 8 meshes: {meshes}")
        create_joint(env_path)
        return {"drawer": meshes}

    def _detent(self, rng, knobs):
        requested = float(knobs["detent_m"])
        if requested == 0.0:
            return float(rng.uniform(*DETENT_RANGE))
        if not 0.04 <= requested <= MAX_EXTENSION - 0.02:
            raise ValueError("detent_m outside the usable drawer travel")
        return requested

    def sample_init_state(self, robot, rng, state_id, knobs):
        yaw = float(rng.uniform(YAW_CENTER_DEG - YAW_RANGE_DEG, YAW_CENTER_DEG + YAW_RANGE_DEG))
        half = math.radians(yaw) / 2
        pose = [*CASE_XY, DRAWER_Z, math.cos(half), 0.0, 0.0, math.sin(half)]
        return {
            "id": state_id,
            "robot": copy.deepcopy(robot),
            "objects": {"case": pose.copy(), "drawer": pose.copy()},
            "task_vars": {
                "side": "right",
                "detent_m": self._detent(rng, knobs),
                "cue_lead_m": float(rng.uniform(*CUE_LEAD_RANGE)),
                "yaw_deg": yaw,
            },
        }

    def validate_init_state(self, state):
        variables = state["task_vars"]
        detent = float(variables["detent_m"])
        if variables.get("side") not in ARM_PADS or not 0.04 <= detent <= MAX_EXTENSION - 0.02:
            raise ValueError("drawer_latch requires a valid arm and hidden detent")
        cue_lead = float(variables.get("cue_lead_m", 0.014))
        if not CUE_LEAD_RANGE[0] <= cue_lead <= CUE_LEAD_RANGE[1]:
            raise ValueError("drawer_latch cue lead lies outside the usable range")

        poses = [np.asarray(state["objects"][name], dtype=float) for name in ("case", "drawer")]
        if any(pose.shape != (7,) or not np.all(np.isfinite(pose)) for pose in poses):
            raise ValueError("drawer_latch requires finite position and quaternion poses")
        if not np.allclose(poses[0], poses[1], atol=1e-4):
            raise ValueError("drawer_latch must start closed with matching case and drawer poses")
        w, x, y, z = poses[0][3:]
        yaw = math.degrees(2 * math.atan2(z, w))
        yaw = (yaw + 180) % 360 - 180
        recorded_yaw = float(variables.get("yaw_deg", yaw))
        if (
            abs(x) > 1e-5
            or abs(y) > 1e-5
            or not math.isclose(w * w + z * z, 1.0, abs_tol=1e-4)
            or not YAW_CENTER_DEG - YAW_RANGE_DEG <= yaw <= YAW_CENTER_DEG + YAW_RANGE_DEG
            or not math.isclose(recorded_yaw, yaw, abs_tol=1e-3)
        ):
            raise ValueError("drawer_latch yaw must match the pose and lie within 20 to 50 degrees")

    def init_state_objects(self):
        return ("case", "drawer")

    def snapshot_init_state(self, h, ctx):
        return {
            "side": ctx["side"],
            "detent_m": ctx["detent_m"],
            "cue_lead_m": ctx["cue_lead_m"],
            "yaw_deg": ctx["yaw_deg"],
        }

    def _new_ctx(self, h, detent, cue_lead, side="right"):
        h.select_arm(side)
        h.latch()
        case = h.scene["case"].data
        w, x, y, z = case.root_quat_w[0].detach().cpu().numpy()
        axis = np.asarray([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 0.0])
        set_joint_frame(getattr(h, "env_path", ENV), float(np.degrees(np.arctan2(axis[1], axis[0]))))
        mechanism = DrawerMechanism(detent, cue_lead)
        path = f"{getattr(h, 'env_path', ENV)}/{JOINT_NAME}"
        set_drive(path, mechanism.update(0.0, 0.0, 0.0))
        return {
            "side": side,
            "pads": ARM_PADS[side],
            "detent_m": detent,
            "cue_lead_m": cue_lead,
            "mechanism": mechanism,
            "last_extension_m": 0.0,
            "mechanism_speed_m_s": 0.0,
            "axis": axis,
            "case_pos": case.root_pos_w[0].detach().cpu().numpy().copy(),
            "yaw_deg": float(np.degrees(np.arctan2(axis[1], axis[0]))),
            "joint": path,
            "grasp_quat": None,
            "pull_cmd_m": None,
            "tactile_stop": TactileStop(),
            "force_cue_x": None,
            "seat_extension_m": None,
            "seat_source": None,
            "seat_hold_s": 0.0,
            "release_pos": None,
            "release_quat": None,
            "open_ready": False,
            "grasp_ready": False,
        }

    def begin_restored_episode(self, h, state, knobs):
        self.validate_init_state(state)
        variables = state["task_vars"]
        return self._new_ctx(
            h,
            float(variables["detent_m"]),
            float(variables.get("cue_lead_m", 0.014)),
            variables["side"],
        )

    def restore_init_state(self, h, state, knobs):
        reset_drive(getattr(h, "env_path", ENV), float(state["task_vars"]["yaw_deg"]))
        h.hold_default(SETTLE_STEPS)
        return self.begin_restored_episode(h, state, knobs)

    def reset(self, h, rng, ep, knobs):
        h.select_arm("right")
        yaw = float(rng.uniform(YAW_CENTER_DEG - YAW_RANGE_DEG, YAW_CENTER_DEG + YAW_RANGE_DEG))
        reset_drive(getattr(h, "env_path", ENV), yaw)
        half = math.radians(yaw) / 2
        quat = (math.cos(half), 0.0, 0.0, math.sin(half))
        h.place(h.scene["case"], CASE_XY, DRAWER_Z, quat)
        h.place(h.scene["drawer"], CASE_XY, DRAWER_Z, quat)
        h.reset_robot()
        h.hold_default(SETTLE_STEPS)
        case_pos = h.scene["case"].data.root_pos_w[0, :2].detach().cpu().numpy()
        drawer_pos = h.scene["drawer"].data.root_pos_w[0, :2].detach().cpu().numpy()
        if float(np.linalg.norm(case_pos - drawer_pos)) > 0.004:
            raise RetryReset("drawer did not settle closed")
        return self._new_ctx(h, self._detent(rng, knobs), float(rng.uniform(*CUE_LEAD_RANGE)))

    def _slide_state(self, h, ctx):
        axis = ctx["axis"][:2]
        case_pos = h.scene["case"].data.root_pos_w[0, :2].detach().cpu().numpy()
        drawer = h.scene["drawer"].data
        drawer_pos = drawer.root_pos_w[0, :2].detach().cpu().numpy()
        extension = float(np.dot(case_pos - drawer_pos, axis))
        return extension

    def after_control_step(self, h, ctx, dt):
        extension = self._slide_state(h, ctx)
        velocity = (extension - ctx["last_extension_m"]) / dt
        ctx["last_extension_m"] = extension
        ctx["mechanism_speed_m_s"] = velocity
        set_drive(ctx["joint"], ctx["mechanism"].update(extension, velocity, dt))

    def signals(self, h, ctx):
        areas = h.pad_areas()
        shear = {}
        slide_load = 0.0
        for pad in ctx["pads"]:
            data = h.sensors[pad].data
            field = data.tactile_shear_force
            shear[pad] = (
                0.0
                if field is None
                else float(np.linalg.norm(field[0].detach().cpu().numpy(), axis=-1).sum())
            )
            if (
                field is not None
                and data.tactile_normal_force is not None
                and data.tactile_points_quat_w is not None
            ):
                local_force = torch.cat([field, data.tactile_normal_force.unsqueeze(-1)], dim=-1)
                world_force = torch.nan_to_num(quat_apply(data.tactile_points_quat_w, local_force)).sum(
                    dim=1
                )[0]
                slide_load += float(
                    torch.dot(
                        world_force,
                        torch.as_tensor(-ctx["axis"], device=world_force.device, dtype=world_force.dtype),
                    )
                )
        tcp = h.tcp_pos()
        gels = h.gel_centres()
        extension = self._slide_state(h, ctx)
        drawer = h.scene["drawer"].data
        handle_pos = drawer.root_pos_w[0].detach().cpu().numpy().copy()
        handle_pos += HANDLE_OFFSET_X * ctx["axis"]
        handle_pos[2] += HANDLE_CENTER_Z
        return {
            "areas": areas,
            "shear": shear,
            "slide_load": slide_load,
            "gel_sep_m": float(np.linalg.norm(gels[1] - gels[0])) if gels is not None else 0.0,
            "extension_m": extension,
            "mechanism_speed_m_s": ctx["mechanism_speed_m_s"],
            "mechanism_dwell_s": ctx["mechanism"].dwell_s,
            "mechanism_force_n": ctx["mechanism"].force_n,
            "mechanism_cue_resistance_n": ctx["mechanism"].cue_resistance_n,
            "mechanism_potential_j": ctx["mechanism"].potential_j,
            "mechanism_bypassed": ctx["mechanism"].bypassed,
            "mechanism_escape_count": ctx["mechanism"].escape_count,
            "mechanism_reset_count": ctx["mechanism"].reset_count,
            "hand_clearance_m": float(np.dot(handle_pos - tcp, ctx["axis"])),
        }

    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        axis = ctx["axis"]
        grip = ctx["case_pos"] + np.array([0.0, 0.0, HANDLE_CENTER_Z]) + GRASP_LOCAL_X * axis
        pre = grip - PREGRASP_BACKOFF * axis
        if name == "home":
            return (pre + np.array([0.0, 0.0, 0.05])).tolist(), OPEN_WIDTH, None
        if name == "pregrasp":
            return pre.tolist(), OPEN_WIDTH, None
        if ctx["grasp_quat"] is None and h.jaw_axis() is not None:
            ctx["grasp_quat"] = align_frame_quat(
                h.tcp_quat(), h.approach_axis(), h.jaw_axis(), axis, (0.0, 0.0, -1.0), signed_b=True
            )
        q = ctx["grasp_quat"]
        if name == "orient":
            return pre.tolist(), OPEN_WIDTH, q
        if name == "approach":
            return grip.tolist(), OPEN_WIDTH, q
        if name == "close":
            return grip.tolist(), CLOSE_WIDTH, q
        if name in ("pull", "seat"):
            if ctx["pull_cmd_m"] is None:
                ctx["pull_cmd_m"] = max(0.0, float(np.dot(grip - h.tcp_pos(), axis)))
            hand_travel = max(0.0, float(np.dot(grip - h.tcp_pos(), axis)))
            detected = ctx["tactile_stop"].update(
                -float(sig["slide_load"]),
                hand_travel,
                enabled=bool(knobs["tactile_stop"]),
            )
            if detected and ctx["seat_extension_m"] is None:
                # Record drawer position after the tactile stop decision.
                ctx["seat_extension_m"] = sig["extension_m"]
                ctx["seat_source"] = "tactile_baseline_confirm"
                ctx["pull_cmd_m"] = hand_travel + 0.001
                ctx["seat_hold_s"] = 0.0
            elif not detected and ctx["seat_extension_m"] is not None:
                ctx["seat_extension_m"] = None
                ctx["seat_source"] = None
                ctx["seat_hold_s"] = 0.0
                ctx["pull_cmd_m"] = hand_travel
            if ctx["seat_extension_m"] is None:
                if hand_travel < 0.115:
                    ctx["pull_cmd_m"] += PULL_SPEED * h.control_dt
            else:
                ctx["seat_hold_s"] += h.control_dt
            return (grip - ctx["pull_cmd_m"] * axis).tolist(), CLOSE_WIDTH, q
        if ctx["release_pos"] is None:
            ctx["release_pos"] = h.tcp_pos().copy()
            ctx["release_quat"] = h.tcp_quat().copy()
        distance = (
            0.0
            if name == "release"
            else (min(RETREAT_DISTANCE, RETREAT_SPEED * t_phase) if name == "retreat" else RETREAT_DISTANCE)
        )
        target = ctx["release_pos"] - distance * axis
        target[2] += RETREAT_HEIGHT * distance / RETREAT_DISTANCE
        return target.tolist(), RELEASE_WIDTH, ctx["release_quat"]

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs):
        if name in ("home", "pregrasp"):
            return h.tcp_err(target) < 0.012
        if name == "orient":
            ctx["open_ready"] = t_phase >= 2.0 and sig["gel_sep_m"] >= 0.065
            return ctx["open_ready"]
        if name == "approach":
            ctx["grasp_ready"] = h.tcp_err(target) < 0.012 and sig["gel_sep_m"] >= 0.035
            return ctx["grasp_ready"]
        if name == "close":
            return t_phase >= 0.4 and any(sig["areas"][p] > GRASP_CONTACT_MM2 for p in ctx["pads"])
        if name == "pull":
            return ctx["tactile_stop"].ready and ctx["seat_hold_s"] >= 1.15
        if name == "seat":
            # Confirm a stable tactile stop before opening the fingers.
            return ctx["tactile_stop"].ready and ctx["seat_hold_s"] >= 1.15
        if name == "release":
            return t_phase > 1.2
        if name == "retreat":
            return (
                t_phase >= RETREAT_DISTANCE / RETREAT_SPEED
                and sig["hand_clearance_m"] >= RETREAT_CLEARANCE
                and all(sig["areas"][pad] <= GRASP_CONTACT_MM2 for pad in ctx["pads"])
            )
        if name == "verify":
            return t_phase >= 1.1
        return False

    def next_phase(self, ctx, name):
        if name in ("pull", "seat") and not ctx["tactile_stop"].ready:
            return "pull"
        if name == "orient" and not ctx["open_ready"]:
            return "release"
        if name == "approach" and not ctx["grasp_ready"]:
            return "release"
        return None

    def build_trace(self, h, ctx, hist):
        return EpisodeTrace(
            t=np.asarray(hist["t"]),
            pad_area={p: np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            extra={
                "extension_m": np.asarray(hist["extension_m"]),
                "hand_clearance_m": np.asarray(hist["hand_clearance_m"]),
            },
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(trace, detent_m=ctx["detent_m"], pads=ctx["pads"])

    def episode_extra(self, ctx, hist):
        return {
            "detent_m": ctx["detent_m"],
            "cue_lead_m": ctx["cue_lead_m"],
            "yaw_deg": ctx["yaw_deg"],
            "force_cue_x": ctx["force_cue_x"],
            "seat_extension_m": ctx["seat_extension_m"],
            "seat_source": ctx["seat_source"],
            "expert_version": EXPERT_VERSION,
            "tactile_rejected_stops": ctx["tactile_stop"].rejected_count,
            "final_extension_m": hist["extension_m"][-1] if hist["extension_m"] else None,
            "mechanism_mode": ctx["mechanism"].mode,
            "mechanism_version": MECHANISM_VERSION,
            "mechanism_escape_count": ctx["mechanism"].escape_count,
            "mechanism_reset_count": ctx["mechanism"].reset_count,
            "latched_extension_m": ctx["mechanism"].latched_extension_m,
        }


TASK = DrawerLatchTask()
