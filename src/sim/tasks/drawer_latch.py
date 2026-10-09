"""Pull the tool drawer until its hidden detent holds it open."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from sim.criteria.drawer_latch import verdict as score_drawer
from sim.criteria.trace import EpisodeTrace
from sim.dims.drawer_latch import (
    CASE_SIZE,
    DETENT_RANGE,
    HANDLE_CENTER_Z,
    HANDLE_STANDOFF,
    MAX_EXTENSION,
    YAW_CENTER_DEG,
    YAW_RANGE_DEG,
)
from sim.envs.bimanual_tactile import TACTILE_SENSORS
from sim.envs.drawer_latch import CASE_XY, DRAWER_Z, PREVIEW, DrawerLatchSceneCfg
from sim.mechanisms.drawer_latch import (
    CUE_LEAD_RANGE,
    RETURN_DAMPING,
    RETURN_STIFFNESS,
    DrawerMechanism,
    DriveSetting,
)
from sim.tactile import apply_sdf_collision
from sim.tactile.quantities import quat_apply
from sim.tasks.scene_utils import align_frame_quat
from sim.tasks.spec import Knob, RetryReset, Task

ENV = "/World/envs/env_0"
JOINT_NAME = "drawer_slide"
SETTLE_STEPS = 30
DT = 1.0 / 60.0
HANDLE_OFFSET_X = -CASE_SIZE[0] / 2 - 0.001 - HANDLE_STANDOFF
PREGRASP_BACKOFF = 0.100
OPEN_WIDTH = 0.040
RELEASE_WIDTH = 0.040
CLOSE_WIDTH = 0.004
PULL_SPEED = 0.020
CUE_PULL_SPEED = 0.006
FORCE_CUE_RISE = 30.0
FORCE_BASELINE_SLOPE = 600.0
LATCH_LOAD_DROP = 85.0
LATCH_CONFIRM_DROP = 40.0
LATCH_CONFIRM_FRAMES = 3
GRASP_CONTACT_MM2 = 5.0
GRASP_TCP_DEPTH_OFFSET = 0.010
GRASP_TCP_HEIGHT_OFFSET = 0.0
GRASP_LOCAL_X = HANDLE_OFFSET_X + GRASP_TCP_DEPTH_OFFSET
ARM_PADS = {"right": TACTILE_SENSORS[:2], "left": TACTILE_SENSORS[2:]}
RETREAT_DISTANCE = 0.100
RETREAT_SPEED = 0.040


def _drawer_joint(env_path: str) -> str:
    import omni.usd
    from pxr import Gf, Sdf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    path = f"{env_path}/{JOINT_NAME}"
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    joint = UsdPhysics.Joint.Define(stage, path)
    prim = joint.GetPrim()
    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{env_path}/case")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{env_path}/drawer")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr(False)
    for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        low, high = (-MAX_EXTENSION, 0.0) if axis == "transX" else (1.0, -1.0)
        limit.CreateLowAttr(low)
        limit.CreateHighAttr(high)
    drive = UsdPhysics.DriveAPI.Apply(prim, "transX")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(RETURN_STIFFNESS)
    drive.CreateDampingAttr(RETURN_DAMPING)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateMaxForceAttr(float("inf"))
    return path


def _set_drive(path: str, setting) -> None:
    import omni.usd
    from pxr import UsdPhysics

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    drive = UsdPhysics.DriveAPI.Apply(prim, "transX")
    drive.GetTargetPositionAttr().Set(float(setting.target_m))
    drive.GetStiffnessAttr().Set(float(setting.stiffness_n_m))
    drive.GetDampingAttr().Set(float(setting.damping_n_s_m))


def _set_joint_frame(env_path: str, yaw_deg: float) -> None:
    import omni.usd
    from pxr import Gf, UsdPhysics

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(f"{env_path}/{JOINT_NAME}")
    joint = UsdPhysics.Joint(prim)
    half = math.radians(yaw_deg - YAW_CENTER_DEG) / 2
    joint.GetLocalRot0Attr().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))


def reset_drive(env_path: str = ENV, yaw_deg: float = YAW_CENTER_DEG) -> None:
    _set_joint_frame(env_path, yaw_deg)
    _set_drive(f"{env_path}/{JOINT_NAME}", DriveSetting(0.0, RETURN_STIFFNESS, RETURN_DAMPING))


class DrawerLatchTask(Task):
    name = "drawer_latch"
    instruction = "pull the tool drawer until it settles into place, then release it and leave it open for use"
    report_name = "drawer_latch_scripted.json"
    knobs = {"detent_m": Knob(0.0, "hidden hold-open position; 0 samples one per episode")}
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
            api = (UsdPhysics.FilteredPairsAPI(own) if own.HasAPI(UsdPhysics.FilteredPairsAPI)
                   else UsdPhysics.FilteredPairsAPI.Apply(own))
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
        _drawer_joint(env_path)
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
        if (abs(x) > 1e-5 or abs(y) > 1e-5
                or not math.isclose(w*w + z*z, 1.0, abs_tol=1e-4)
                or not YAW_CENTER_DEG - YAW_RANGE_DEG <= yaw <= YAW_CENTER_DEG + YAW_RANGE_DEG
                or not math.isclose(recorded_yaw, yaw, abs_tol=1e-3)):
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
        axis = np.asarray([1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 0.0])
        _set_joint_frame(getattr(h, "env_path", ENV), float(np.degrees(np.arctan2(axis[1], axis[0]))))
        mechanism = DrawerMechanism(detent, cue_lead)
        path = f"{getattr(h, 'env_path', ENV)}/{JOINT_NAME}"
        _set_drive(path, mechanism.update(0.0, 0.0, 0.0))
        return {
            "side": side,
            "pads": ARM_PADS[side],
            "detent_m": detent,
            "cue_lead_m": cue_lead,
            "mechanism": mechanism,
            "axis": axis,
            "case_pos": case.root_pos_w[0].detach().cpu().numpy().copy(),
            "yaw_deg": float(np.degrees(np.arctan2(axis[1], axis[0]))),
            "joint": path,
            "grasp_quat": None,
            "pull_cmd_m": None,
            "force_baseline": [],
            "force_cue_hits": [],
            "force_cue_x": None,
            "recent_slide_loads": [],
            "snap_candidate": None,
            "seat_extension_m": None,
            "seat_hold_s": 0.0,
            "release_pos": None,
            "open_ready": False,
            "grasp_ready": False,
            "phase_i": 0,
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
        reset_drive(getattr(h, "env_path", ENV))
        yaw = float(rng.uniform(YAW_CENTER_DEG - YAW_RANGE_DEG, YAW_CENTER_DEG + YAW_RANGE_DEG))
        _set_joint_frame(getattr(h, "env_path", ENV), yaw)
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

    def signals(self, h, ctx):
        areas = h.pad_areas()
        shear = {}
        slide_load = 0.0
        for pad in ctx["pads"]:
            data = h.sensors[pad].data
            field = data.tactile_shear_force
            shear[pad] = (
                0.0 if field is None
                else float(np.linalg.norm(field[0].detach().cpu().numpy(), axis=-1).sum())
            )
            if (field is not None and data.tactile_normal_force is not None
                    and data.tactile_points_quat_w is not None):
                local_force = torch.cat([field, data.tactile_normal_force.unsqueeze(-1)], dim=-1)
                world_force = torch.nan_to_num(quat_apply(data.tactile_points_quat_w, local_force)).sum(dim=1)[0]
                slide_load += float(torch.dot(world_force, torch.as_tensor(
                    -ctx["axis"], device=world_force.device, dtype=world_force.dtype
                )))
        tcp = h.tcp_pos()
        gels = h.gel_centres()
        gel_mid = 0.5 * (gels[0] + gels[1]) if gels is not None else tcp
        jaw = h.jaw_axis()
        approach = h.approach_axis()
        wrist_err_deg = 180.0
        if ctx["grasp_quat"] is not None:
            dot = abs(float(np.dot(h.tcp_quat(), ctx["grasp_quat"])))
            wrist_err_deg = float(np.degrees(2.0 * np.arccos(min(1.0, dot))))
        axis = ctx["axis"][:2]
        case_pos = h.scene["case"].data.root_pos_w[0, :2].detach().cpu().numpy()
        drawer = h.scene["drawer"].data
        drawer_pos = drawer.root_pos_w[0, :2].detach().cpu().numpy()
        extension = float(np.dot(case_pos - drawer_pos, axis))
        drawer_velocity = drawer.root_lin_vel_w[0, :2].detach().cpu().numpy()
        velocity = -float(np.dot(drawer_velocity, axis))
        setting = ctx["mechanism"].update(extension, velocity, h.control_dt)
        _set_drive(ctx["joint"], setting)
        handle_pos = drawer.root_pos_w[0].detach().cpu().numpy().copy()
        handle_pos += HANDLE_OFFSET_X * ctx["axis"]
        handle_pos[2] += HANDLE_CENTER_Z
        return {
            "areas": areas,
            "shear": shear,
            "slide_load": slide_load,
            "tcp_x_m": float(tcp[0]),
            "tcp_y_m": float(tcp[1]),
            "tcp_z_m": float(tcp[2]),
            "gel_x_m": float(gel_mid[0]),
            "gel_y_m": float(gel_mid[1]),
            "gel_z_m": float(gel_mid[2]),
            "gel0_z_m": float(gels[0][2]) if gels is not None else float(tcp[2]),
            "gel1_z_m": float(gels[1][2]) if gels is not None else float(tcp[2]),
            "gel_sep_m": float(np.linalg.norm(gels[1] - gels[0])) if gels is not None else 0.0,
            "jaw_z": float(jaw[2]) if jaw is not None else 0.0,
            "approach_z": float(approach[2]),
            "wrist_err_deg": wrist_err_deg,
            "extension_m": extension,
            "hand_clearance_m": float(np.dot(handle_pos - tcp, ctx["axis"])),
            "drawer_velocity_m_s": velocity,
            "phase_i": ctx["phase_i"],
        }

    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        ctx["phase_i"] = [p for p, _ in self.phases].index(name)
        axis = ctx["axis"]
        grip = ctx["case_pos"] + np.array([0.0, 0.0, HANDLE_CENTER_Z + GRASP_TCP_HEIGHT_OFFSET]) + GRASP_LOCAL_X * axis
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
            extension = sig["extension_m"]
            if name == "pull" and ctx["force_cue_x"] is None:
                if 0.015 <= extension <= 0.040:
                    ctx["force_baseline"].append((extension, sig["slide_load"]))
                baseline = ctx["force_baseline"]
                if extension >= 0.040 and len(baseline) >= 6:
                    x_ref = float(np.median([sample[0] for sample in baseline]))
                    load_ref = float(np.median([sample[1] - FORCE_BASELINE_SLOPE * (sample[0] - x_ref)
                                                for sample in baseline]))
                    expected_load = load_ref + FORCE_BASELINE_SLOPE * (extension - x_ref)
                    if sig["slide_load"] > expected_load + FORCE_CUE_RISE:
                        ctx["force_cue_hits"].append(extension)
                    if len(ctx["force_cue_hits"]) >= 2:
                        ctx["force_cue_x"] = ctx["force_cue_hits"][0]
            seeking_latch = (
                name == "pull"
                and ctx["force_cue_x"] is not None
                and ctx["seat_extension_m"] is None
            )
            if seeking_latch:
                loads = ctx["recent_slide_loads"]
                candidate = ctx["snap_candidate"]
                if candidate is not None:
                    candidate["age"] += 1
                    if sig["slide_load"] > candidate["reference_load"] - LATCH_CONFIRM_DROP:
                        ctx["snap_candidate"] = None
                    elif candidate["age"] >= LATCH_CONFIRM_FRAMES:
                        ctx["seat_extension_m"] = extension
                if ctx["snap_candidate"] is None and ctx["seat_extension_m"] is None and len(loads) >= 3:
                    reference = float(np.median(loads))
                    snap = sig["slide_load"] < reference - LATCH_LOAD_DROP
                else:
                    snap = False
                if snap:
                    ctx["snap_candidate"] = {
                        "age": 0,
                        "reference_load": reference,
                    }
                loads.append(sig["slide_load"])
                del loads[:-4]
                if ctx["seat_extension_m"] is not None:
                    ctx["pull_cmd_m"] = max(0.0, float(np.dot(grip - h.tcp_pos(), axis)))
            seat = ctx["seat_extension_m"]
            if name == "pull" and seat is None and extension < 0.105:
                speed = CUE_PULL_SPEED if ctx["force_cue_x"] is not None else PULL_SPEED
                ctx["pull_cmd_m"] += speed * h.control_dt
            elif seat is not None:
                ctx["seat_hold_s"] += h.control_dt
            return (grip - ctx["pull_cmd_m"] * axis).tolist(), CLOSE_WIDTH, q
        if ctx["release_pos"] is None:
            ctx["release_pos"] = h.tcp_pos().copy()
        distance = 0.0 if name == "release" else (
            min(RETREAT_DISTANCE, RETREAT_SPEED * t_phase) if name == "retreat" else RETREAT_DISTANCE
        )
        return (ctx["release_pos"] - distance * axis).tolist(), RELEASE_WIDTH, q

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
            return ctx["seat_extension_m"] is not None and ctx["seat_hold_s"] >= 0.35
        if name == "seat":
            return ctx["seat_extension_m"] is not None and ctx["seat_hold_s"] >= 0.65
        if name == "release":
            return t_phase > 1.2
        if name == "retreat":
            return t_phase >= RETREAT_DISTANCE / RETREAT_SPEED and h.tcp_err(target) < 0.012
        if name == "verify":
            return t_phase >= 1.1
        return False

    def next_phase(self, ctx, name):
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
        return score_drawer(trace, detent_m=ctx["detent_m"], pads=ctx["pads"])

    def episode_extra(self, ctx, hist):
        return {
            "detent_m": ctx["detent_m"],
            "cue_lead_m": ctx["cue_lead_m"],
            "yaw_deg": ctx["yaw_deg"],
            "force_cue_x": ctx["force_cue_x"],
            "seat_extension_m": ctx["seat_extension_m"],
            "final_extension_m": hist["extension_m"][-1] if hist["extension_m"] else None,
            "mechanism_mode": ctx["mechanism"].mode,
        }


TASK = DrawerLatchTask()
