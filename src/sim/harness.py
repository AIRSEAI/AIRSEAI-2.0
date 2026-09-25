"""The Isaac-side plumbing every runner needs, defined once.

Scene bring-up, both arms' IK chains, the 16-dim named state/action mapping,
the dataset cameras, tactile sampling, and the policy loaders. The expert
runner and the evaluator both drive a task through this object.

Import only after `AppLauncher` has started Kit: everything here touches
Isaac Lab.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, subtract_frame_transforms

from sim.assets.provision import ensure_robot_asset
from sim.envs.bimanual_tactile import SENSOR_FEATURE_NAME, TACTILE_SENSORS
from sim.physx_settings import small_scene_physx_cfg
from sim.tactile import contact_area, taxel_pitch
from sim.tactile.quantities import quat_apply
from tactile_codec.tacff import DEPTH_MAX_MM as TACFF_DEPTH_MAX_MM
from tactile_codec.tacff import SHEAR_MAX as TACFF_SHEAR_MAX

__all__ = [
    "DT",
    "CTRL_FPS",
    "STEPS_PER_CTRL",
    "ARM_PADS",
    "Harness",
    "HoldPosePolicy",
    "RemotePolicy",
    "load_policy",
    "look_at",
    "tacff_image_batch",
    "TACFF_DEPTH_MAX_MM",
    "TACFF_SHEAR_MAX",
]

DT = 1.0 / 60.0
CTRL_FPS = 20  # dataset rate; policy rate too
STEPS_PER_CTRL = int(round(1 / (CTRL_FPS * DT)))
ARM_PADS = {"right": TACTILE_SENSORS[:2], "left": TACTILE_SENSORS[2:]}


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion for a camera at `eye` aimed at `target`, opengl convention."""
    eye, target, up = map(np.asarray, (eye, target, up))
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    w = max(math.sqrt(max(1.0 + R.trace(), 0.0)) / 2.0, 1e-8)
    return (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))


def tacff_image_batch(depth_m: torch.Tensor, shear: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Sensor fields -> (1,3,H,W) float in [0,1], the converter's fixed scales.

    NaN is filled with 0, matching the converter: shear 0 maps to mid-gray.
    """
    d_mm = torch.nan_to_num(depth_m).view(rows, cols) * 1e3
    s = torch.nan_to_num(shear.view(rows, cols, 2) * 1e3, nan=0.0)
    r = (d_mm / TACFF_DEPTH_MAX_MM).clamp(0, 1)
    gb = (s / (2.0 * TACFF_SHEAR_MAX) + 0.5).clamp(0, 1)
    img = torch.cat([r.unsqueeze(-1), gb], dim=-1)
    return img.permute(2, 0, 1).unsqueeze(0).float()


class Harness:
    """One scene, both arms, the cameras and pads a task or policy can see."""

    control_dt = DT

    def __init__(
        self,
        cfg,
        *,
        device: str,
        grid=(288, 384),
        robot_usd: str = "",
        scene_cam: tuple | None = None,
        cameras: bool = True,
    ):
        """`cfg` is a fully-configured scene cfg (knobs already applied).
        `scene_cam=(eye, target)` adds a third-person debug camera.

        `cameras=False` builds no image sensors, for scripts that need only IK
        and physics; without cameras this class does not depend on the RTX
        renderer."""
        self.device = device
        self.rows, self.cols = grid
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=DT, device=device, physx=small_scene_physx_cfg())
        )
        cfg.robot.spawn.usd_path = ensure_robot_asset(robot_usd)
        for pad in TACTILE_SENSORS:
            getattr(cfg, pad).tactile_array_size = (self.rows, self.cols)
        self.cfg = cfg
        self.scene = InteractiveScene(cfg)

        # Dataset cameras: the head camera is the `base` stream; the two D405s
        # ride the hand links. All are prims welded in by the asset builder.
        self.cams: dict[str, TiledCamera] = {}
        for key, prim in (
            ()
            if not cameras
            else (
                ("base", "openarm_body_link/head_camera/camera"),
                ("left_wrist", "openarm_left_hand/wrist_camera/camera"),
                ("right_wrist", "openarm_right_hand/wrist_camera/camera"),
            )
        ):
            self.cams[key] = TiledCamera(
                TiledCameraCfg(
                    prim_path=f"/World/envs/env_0/Robot/{prim}",
                    offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
                    width=640,
                    height=480,
                    data_types=["rgb"],
                    spawn=None,
                )
            )
        self.scene_cam = None
        if scene_cam is not None and cameras:
            eye, target = scene_cam
            self.scene_cam = TiledCamera(
                TiledCameraCfg(
                    prim_path="/World/scene_cam",
                    offset=TiledCameraCfg.OffsetCfg(pos=eye, rot=look_at(eye, target), convention="opengl"),
                    width=960,
                    height=720,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(),
                )
            )

        # Deferred until after collision setup: the caller applies SDF /
        # triangle / CCD to the task's objects, then calls `start()`.
        self.robot = None
        self.sensors = {}
        self._hold = None  # latched joint command; see `latch`
        self._posture = {}  # side -> (rest joints, gain); see `set_posture_bias`
        self._joint_idx = None  # name -> index, built on first use
        self._chains = {}
        self.ch: dict = {}

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        self.sim.reset()
        self._bind_scene()

    def _bind_scene(self) -> None:
        """Bind controllers to an already initialized scene."""
        self.robot = self.scene["robot"]
        self.sensors = {p: self.scene[p] for p in TACTILE_SENSORS}
        self.pitch = taxel_pitch((self.rows, self.cols))

        # One IK chain per arm, resolved once; `ch` is whichever is active.
        for side in ("right", "left"):
            ent = SceneEntityCfg(
                "robot", joint_names=[f"openarm_{side}_joint[1-7]"], body_names=[f"openarm_{side}_ee_tcp"]
            )
            ent.resolve(self.scene)
            self._chains[side] = {
                "side": side,
                "arm": ent,
                "ee_body": ent.body_ids[0],
                "jac_idx": ent.body_ids[0] - 1,  # fixed base: body-1
                "grip_ids": [i for i, n in enumerate(self.robot.joint_names) if f"{side}_finger_joint" in n],
                "pads": ARM_PADS[side],
            }
        self.select_arm("right")

        self.ik_pos = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=self.sim.device,
        )
        self.ik_pose = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=self.sim.device,
        )

        # 16-dim named state/action from the 22 raw joints: 7 arm joints (deg)
        # + gripper (jaw opening, mm) per arm, right first -- the real layout.
        jname = list(self.robot.joint_names)
        idx, scale = [], []
        for a in ("right", "left"):
            for i in range(1, 8):
                idx.append(jname.index(f"openarm_{a}_joint{i}"))
                scale.append(180.0 / math.pi)
            idx.append(jname.index(f"openarm_{a}_finger_joint1"))
            scale.append(1e3)
        self._idx16 = np.asarray(idx)
        self._scale16 = np.asarray(scale, dtype=np.float32)
        self._idx16_t = torch.as_tensor(self._idx16, device=self.sim.device)
        self._scale16_t = torch.as_tensor(self._scale16, device=self.sim.device)
        # One update so every camera is initialised: Kit's teardown hangs on
        # a TiledCamera that was created and never rendered.
        if self.cams or self.scene_cam is not None:
            self.update_cams(include_scene_cam=True)

    def select_arm(self, side: str) -> None:
        self.ch = dict(self._chains[side])

    def ee_body(self, side: str | None = None) -> int:
        """Body index of `side`'s TCP (the active arm's by default)."""
        return self._chains[side]["ee_body"] if side else self.ch["ee_body"]

    # --------------------------------------------------------------- latching
    # Unlatched, the idle arm is commanded to its own measured `joint_pos`,
    # which sags under a held load. Latching freezes the command vector so
    # the IK step builds on it; `hold_grip` still lets a latched arm let go.
    def latch(self) -> None:
        """Freeze all joints at the command currently in flight."""
        self._hold = self.robot.data.joint_pos_target.clone()

    def unlatch(self) -> None:
        self._hold = None

    def hold_grip(self, side: str, grip_m: float) -> None:
        """Change a latched arm's gripper without releasing the latch."""
        if self._hold is None:
            self.latch()
        self._hold[:, self._chains[side]["grip_ids"]] = grip_m

    def reset_ik(self) -> None:
        self.ik_pos.reset()
        self.ik_pose.reset()

    # ----------------------------------------------------------------- stepping
    def step(self, render: bool = True) -> None:
        self.sim.step(render=render)
        self.scene.update(DT)

    def reset_robot(self) -> None:
        r = self.robot
        r.write_joint_state_to_sim(r.data.default_joint_pos, r.data.default_joint_vel)
        self.reset_ik()

    def hold_default(self, n_steps: int) -> None:
        """Settle: command the default joint pose for `n_steps`."""
        for _ in range(n_steps):
            self.robot.set_joint_position_target(self.robot.data.default_joint_pos)
            self.robot.write_data_to_sim()
            self.step()

    def place(self, obj, xy, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = obj.data.default_root_state.clone()
        st[:, 0], st[:, 1], st[:, 2] = float(xy[0]), float(xy[1]), float(z)
        st[:, 3:7] = torch.tensor(quat, device=self.sim.device)
        obj.write_root_pose_to_sim(st[:, :7])
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.sim.device))

    def initial_render(self) -> None:
        for s in self.sensors.values():
            s.get_initial_render()

    # ----------------------------------------------------------------- IK
    def _tcp_pose_b(self):
        root = self.robot.data.root_state_w[:, 0:7]
        ee = self.robot.data.body_state_w[:, self.ch["ee_body"], 0:7]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee[:, 0:3], ee[:, 3:7])

    def step_ik_toward(
        self,
        target_pos_w,
        grip_m: float,
        quat_w=None,
        max_speed: float | None = None,
        max_ang_speed: float | None = None,
    ) -> None:
        """One control step of the active arm: position IK (or pose IK when
        `quat_w` is given) toward the world target, gripper at `grip_m`.

        `max_speed` (m/s) caps how far the *commanded* point may sit ahead of
        the TCP. Without it the command is the final goal from the first step
        and the largest displacement is the first one, next to the object.

        `max_ang_speed` (deg/s) does the same for orientation. Capping one
        without the other is worse than capping neither: the solver spends
        the step on the uncapped error, and the joint motion that serves
        orientation moves the TCP off the position goal.

        Both caps are opt-in and off by default.
        """
        r, ch = self.robot, self.ch
        if max_speed is not None:
            here = self.tcp_pos()
            want = np.asarray(target_pos_w, dtype=float)
            d = want - here
            n = float(np.linalg.norm(d))
            cap = float(max_speed) * self.control_dt
            if n > cap:
                target_pos_w = here + d * (cap / n)
        if max_ang_speed is not None and quat_w is not None:
            q_now = r.data.body_state_w[0, ch["ee_body"], 3:7]
            q_want = torch.as_tensor(quat_w, dtype=q_now.dtype, device=q_now.device).reshape(4)
            # The double cover: take the nearer representative, or a match can
            # read as a half turn and the slew would go the long way round.
            if float(torch.dot(q_now, q_want)) < 0.0:
                q_want = -q_want
            dot = float(torch.clamp(torch.dot(q_now, q_want), -1.0, 1.0))
            ang = 2.0 * math.acos(min(1.0, abs(dot)))
            cap = math.radians(float(max_ang_speed)) * self.control_dt
            if ang > cap and ang > 1e-9:
                t = cap / ang
                sin_a = math.sin(ang / 2.0)
                if sin_a > 1e-9:  # slerp
                    q_want = (
                        math.sin((1 - t) * ang / 2.0) / sin_a * q_now
                        + math.sin(t * ang / 2.0) / sin_a * q_want
                    )
                    q_want = q_want / torch.linalg.norm(q_want)
                quat_w = q_want
        pos_b, quat_b = self._tcp_pose_b()
        pos = torch.tensor(target_pos_w, dtype=torch.float32, device=self.sim.device).unsqueeze(0)
        jac_all = r.root_physx_view.get_jacobians()[:, ch["jac_idx"], :, :]
        jids = ch["arm"].joint_ids
        if quat_w is None:
            self.ik_pos.set_command(pos, ee_quat=quat_b)
            jac = jac_all[:, :3, :][:, :, jids]
            qdes = self.ik_pos.compute(pos_b, quat_b, jac, r.data.joint_pos[:, jids])
        else:
            # Tasks hand in either a tensor or a numpy (w, x, y, z).
            q_w = torch.as_tensor(quat_w, dtype=torch.float32, device=self.sim.device).reshape(1, 4)
            _, q_cmd = subtract_frame_transforms(
                r.data.root_state_w[:, 0:3],
                r.data.root_state_w[:, 3:7],
                r.data.body_state_w[:, ch["ee_body"], 0:3],
                q_w,
            )
            self.ik_pose.set_command(torch.cat([pos, q_cmd], dim=-1))
            jac = jac_all[:, :, jids]
            qdes = self.ik_pose.compute(pos_b, quat_b, jac, r.data.joint_pos[:, jids])
        bias = self._posture.get(ch["side"])
        if bias is not None:
            qdes = qdes + self._null_space_step(jac, r.data.joint_pos[:, jids], *bias)
        target = (self._hold if self._hold is not None else r.data.joint_pos).clone()
        target[:, jids] = qdes
        target[:, ch["grip_ids"]] = grip_m
        r.set_joint_position_target(target)
        r.write_data_to_sim()

    def set_posture_bias(self, joint_pos: dict | None, gain: float = 0.05) -> None:
        """Pull each arm's IK null space toward a rest posture.

        `joint_pos` (name -> radians) names the rest posture; an arm is biased
        only when all seven of its joints are named. `gain` is the fraction of
        the posture error removed per step, projected so that it moves no task
        coordinate the IK is controlling. None clears the bias.

        Without a bias the redundant motion stays wherever the episode
        started, so an arm keeps HOME's shoulder yaw all the way to the object.
        """
        self._posture = {}
        if joint_pos is None:
            return
        names = list(self.robot.joint_names)
        for side, chain in self._chains.items():
            arm = [names[i] for i in chain["arm"].joint_ids]
            if all(n in joint_pos for n in arm):
                rest = torch.tensor(
                    [[float(joint_pos[n]) for n in arm]], dtype=torch.float32, device=self.sim.device
                )
                self._posture[side] = (rest, float(gain))

    @staticmethod
    def _null_space_step(jac: torch.Tensor, q: torch.Tensor, rest: torch.Tensor, gain: float) -> torch.Tensor:
        """Joint step toward `rest` with its task-space component removed:
        (I - J+ J) * gain * (rest - q)."""
        jp = torch.linalg.pinv(jac)
        eye = torch.eye(jac.shape[-1], dtype=jac.dtype, device=jac.device).unsqueeze(0)
        return ((eye - jp @ jac) @ (gain * (rest - q)).unsqueeze(-1)).squeeze(-1)

    def step_joint_toward(self, joint_pos: dict, grip_m: float) -> None:
        """One control step driven in JOINT space: hold `joint_pos` (name ->
        radians) on the joints it names, leave the rest where they are, and
        command the active arm's gripper.

        The IK step leaves the null space where it started, so it cannot change
        the arm's configuration. Phases listed in `Task.joint_phases` use this
        instead, and their `phase_target` returns a joint dict.
        """
        r = self.robot
        names = self._joint_name_index
        target = (self._hold if self._hold is not None else r.data.joint_pos).clone()
        for name, value in joint_pos.items():
            target[:, names[name]] = float(value)
        target[:, self.ch["grip_ids"]] = grip_m
        r.set_joint_position_target(target)
        r.write_data_to_sim()

    @property
    def _joint_name_index(self) -> dict:
        """name -> index in the articulation's joint order. Built on first
        use: the robot does not exist until `start()` runs."""
        if self._joint_idx is None:
            self._joint_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        return self._joint_idx

    def gel_centres(self, side: str | None = None):
        """World centroid of each of one gripper's two taxel clouds.

        Measured taxel positions, not a frame convention, so orientation
        quantities derived from them reflect the real jaw.
        """
        pads = ARM_PADS[side] if side else self.ch["pads"]
        out = []
        for p in pads:
            pts = self.sensors[p].data.tactile_points_pos_w
            if pts is None:
                return None
            out.append(pts[0].mean(dim=0).detach().cpu().numpy())
        return out

    def jaw_axis(self, side: str | None = None):
        """Unit vector from one gel face to the other, in the world.

        A line, not a direction: the pad order is arbitrary, so match either
        end against a target or risk commanding a needless 180 degree flip.
        """
        c = self.gel_centres(side)
        if c is None:
            return None
        v = c[1] - c[0]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None

    def tcp_pos(self) -> np.ndarray:
        return self.robot.data.body_state_w[0, self.ch["ee_body"], 0:3].detach().cpu().numpy()

    def tcp_err(self, target_pos_w) -> float:
        return float(np.linalg.norm(self.tcp_pos() - np.asarray(target_pos_w)))

    def level_wrist(self, obj) -> torch.Tensor:
        """Wrist orientation that would bring the held object's +z upright:
        the minimal rotation taking its axis to world +z, times the current
        TCP orientation."""
        q_obj = obj.data.root_quat_w[0:1]
        axis = quat_apply(q_obj, torch.tensor([[0.0, 0.0, 1.0]], device=self.sim.device))[0]
        axis = axis / axis.norm()
        up = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device)
        cross = torch.linalg.cross(axis, up)
        s_norm = cross.norm()
        q_tcp = self.robot.data.body_state_w[0:1, self.ch["ee_body"], 3:7]
        if float(s_norm) < 1e-6:
            return q_tcp[0].clone()
        angle = torch.arccos(torch.dot(axis, up).clamp(-1.0, 1.0))
        q_fix = quat_from_angle_axis(angle.unsqueeze(0), (cross / s_norm).unsqueeze(0))
        return quat_mul(q_fix, q_tcp)[0]

    # ------------------------------------------------------------ 16-dim
    def to16(self, qvec: torch.Tensor) -> np.ndarray:
        return (qvec.detach().cpu().numpy()[self._idx16] * self._scale16).astype(np.float32)

    def state16(self) -> torch.Tensor:
        """(1,16) float32 on device: what the policy sees as observation.state."""
        q = self.robot.data.joint_pos[0, self._idx16_t] * self._scale16_t
        return q.unsqueeze(0).float()

    def action16_recorded(self) -> np.ndarray:
        """The joint command actually sent, in the 16-dim layout -- what a
        demonstration records as `action`."""
        return self.to16(self.robot.data.joint_pos_target[0])

    def apply_action16(self, a16: torch.Tensor) -> None:
        """Policy action (16, named layout) -> joint targets on all 22 joints;
        held for STEPS_PER_CTRL sim steps by the caller."""
        a = torch.as_tensor(a16, device=self.sim.device).reshape(-1).float()
        target = self.robot.data.joint_pos.clone()
        target[0, self._idx16_t] = a / self._scale16_t
        self.robot.set_joint_position_target(target)
        self.robot.write_data_to_sim()

    # ----------------------------------------------------------- sensing
    def pad_areas(self) -> dict[str, float]:
        px, py = self.pitch
        out = {}
        for p in TACTILE_SENSORS:
            d = self.sensors[p].data
            out[p] = (
                0.0
                if d.penetration_depth is None
                else float(contact_area(d.penetration_depth, px, py)["area_mm2"][0])
            )
        return out

    def pad_penetration(self, side: str | None = None, quantile: float | None = None) -> dict[str, float]:
        """Gel penetration per pad, metres, keyed by pad name: the deepest
        taxel, or the `quantile` (0-1) of all taxels when given, which is
        steadier than the single deepest point."""
        pads = ARM_PADS[side] if side else self.ch["pads"]
        out = {}
        for p in pads:
            d = self.sensors[p].data
            if d.penetration_depth is None:
                out[p] = 0.0
            elif quantile is None:
                out[p] = float(d.penetration_depth[0].max())
            else:
                out[p] = float(torch.quantile(d.penetration_depth[0].flatten().float(), quantile))
        return out

    def grip_measured(self) -> float:
        """The active gripper's measured jaw position, metres (not the
        commanded one, which the drive lags by several steps)."""
        return float(self.robot.data.joint_pos[0, self.ch["grip_ids"][0]])

    def tcp_quat(self) -> np.ndarray:
        """Active arm's TCP orientation in the world, (w, x, y, z)."""
        return self.robot.data.body_state_w[0, self.ch["ee_body"], 3:7].detach().cpu().numpy()

    def approach_axis(self, side: str | None = None) -> np.ndarray:
        """Unit vector from the hand link to the TCP: the direction the
        gripper points. Straight down is (0, 0, -1)."""
        s = side or self.ch["side"]
        i = list(self.robot.data.body_names).index(f"openarm_{s}_hand")
        hand = self.robot.data.body_state_w[0, i, 0:3].detach().cpu().numpy()
        v = self.tcp_pos() - hand
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    def blade_tilt_deg(self, side: str | None = None) -> float:
        """Angle between the approach axis and straight down, degrees.
        0 is a level top-down grasp."""
        a = self.approach_axis(side)
        return float(math.degrees(math.acos(max(-1.0, min(1.0, -float(a[2]))))))

    def update_cams(self, include_scene_cam: bool = False) -> None:
        for c in self.cams.values():
            c.update(DT)
        if include_scene_cam and self.scene_cam is not None:
            self.scene_cam.update(DT)

    def cam_rgb(self, key: str) -> np.ndarray:
        cam = self.scene_cam if key == "scene" else self.cams[key]
        return cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)

    def image_batch(self, key: str) -> torch.Tensor:
        """(1,3,H,W) float32 in [0,1] on device."""
        rgb = self.cams[key].data.output["rgb"][0, ..., :3]
        return (rgb.permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(self.device)

    def tactile_raw(self, pad: str):
        """(depth_mm HxWx1, ff_shear HxWx2) float32 numpy, zeros pre-contact.
        The dataset's raw tactile channels."""
        d = self.sensors[pad].data
        r, c = self.rows, self.cols
        pen = (
            d.penetration_depth[0].view(r, c, 1).detach().cpu().numpy() * 1e3
            if d.penetration_depth is not None
            else np.zeros((r, c, 1), np.float32)
        )
        sh = (
            d.tactile_shear_force[0].view(r, c, 2).detach().cpu().numpy() * 1e3
            if d.tactile_shear_force is not None
            else np.zeros((r, c, 2), np.float32)
        )
        return pen, sh

    def tacrgb_u8(self, pad: str) -> np.ndarray:
        """Taxim render resized 240x320 -> 288x384 (bilinear, clamp, round),
        the real recordings' size. Zeros when the sensor has not rendered."""
        d = self.sensors[pad].data
        if d.tactile_rgb_image is None:
            return np.zeros((288, 384, 3), np.uint8)
        t8 = d.tactile_rgb_image[0, ..., :3].float()
        t8 = torch.nn.functional.interpolate(
            t8.permute(2, 0, 1).unsqueeze(0), size=(288, 384), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0)
        return t8.clamp(0, 255).round().to(torch.uint8).detach().cpu().numpy()

    def tacrgb_batch(self, pad: str) -> torch.Tensor:
        u8 = torch.as_tensor(self.tacrgb_u8(pad), device=self.device)
        return u8.permute(2, 0, 1).unsqueeze(0).float() / 255.0

    def tacff_batch(self, pad: str) -> torch.Tensor:
        """The converter's tacff image, synthesised live. A sensor that has
        not produced a reading yet yields the converter's no-contact image
        (R=0, G/B=0.5), never a black frame."""
        d = self.sensors[pad].data
        n = self.rows * self.cols
        if d.penetration_depth is not None:
            pen = d.penetration_depth[0]
            sh = (
                d.tactile_shear_force[0]
                if d.tactile_shear_force is not None
                else torch.zeros(n, 2, device=pen.device)
            )
        else:
            pen = torch.zeros(n, device=self.sim.device)
            sh = torch.zeros(n, 2, device=self.sim.device)
        return tacff_image_batch(pen, sh, self.rows, self.cols).to(self.device)

    @staticmethod
    def tactile_key_stem(pad: str) -> str:
        """Scene sensor name -> the dataset/policy key stem, e.g.
        'right_left_finger_tactile' -> 'right_middle'. Goes through the
        feature-name map so the jaw->finger mapping is the real robot's."""
        feat = SENSOR_FEATURE_NAME[pad]  # e.g. right_hand_middle_finger
        parts = feat.split("_")
        return f"{parts[0]}_{parts[2]}"

    # -------------------------------------------------------- observations
    def build_obs(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """Exactly the keys a policy was trained on, nothing else. Raises on
        a key this harness cannot provide, naming it."""
        obs = {}
        stems = {self.tactile_key_stem(p): p for p in TACTILE_SENSORS}
        for k in keys:
            if k == "observation.state":
                obs[k] = self.state16().to(self.device)
            elif k in (
                "observation.images.base",
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            ):
                obs[k] = self.image_batch(k.rsplit(".", 1)[-1])
            elif k.startswith("observation.images.tacff_"):
                stem = k[len("observation.images.tacff_") :]
                if stem not in stems:
                    raise KeyError(self._key_error(k, stems))
                obs[k] = self.tacff_batch(stems[stem])
            elif k.startswith("observation.images.tacrgb_"):
                stem = k[len("observation.images.tacrgb_") :]
                if stem not in stems:
                    raise KeyError(self._key_error(k, stems))
                obs[k] = self.tacrgb_batch(stems[stem])
            else:
                raise KeyError(self._key_error(k, stems))
        return obs

    def _key_error(self, key: str, stems: dict) -> str:
        have = (
            [
                "observation.state",
                "observation.images.base",
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            ]
            + [f"observation.images.tacff_{s}" for s in stems]
            + [f"observation.images.tacrgb_{s}" for s in stems]
        )
        return f"the checkpoint expects '{key}', which this harness cannot provide. Keys available: {have}"


# ------------------------------------------------------------------ policies
class HoldPosePolicy:
    """Test policy: action = current state. Exercises the whole evaluation
    loop (reset -> prologue -> policy -> termination -> verdict -> report)
    without a checkpoint. Success is expected to be 0."""

    input_keys = ["observation.state"]

    def reset(self):
        pass

    def select_action(self, obs):
        return obs["observation.state"][0].detach().cpu()


class RemotePolicy:
    """Client for the TCP policy server: same reset/select_action
    surface, plus the checkpoint's input keys so the caller can build exactly
    the observation the policy was trained on."""

    def __init__(self, addr: str):
        import pickle
        import socket
        import struct

        self._pickle, self._struct, self._socket = pickle, struct, socket
        host, port = addr.rsplit(":", 1)
        self._sock = socket.create_connection((host, int(port)), timeout=60)
        meta = self._call({"op": "meta"})
        self.input_keys = list(meta["input_keys"])
        self.policy_type = meta.get("policy_type", "?")

    def _call(self, msg: dict) -> dict:
        data = self._pickle.dumps(msg, protocol=4)
        self._sock.sendall(self._struct.pack("<Q", len(data)) + data)
        hdr = self._sock.recv(8, self._socket.MSG_WAITALL)
        (n,) = self._struct.unpack("<Q", hdr)
        buf = b""
        while len(buf) < n:
            buf += self._sock.recv(min(1 << 20, n - len(buf)))
        out = self._pickle.loads(buf)
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def reset(self):
        self._call({"op": "reset"})

    def select_action(self, obs: dict) -> torch.Tensor:
        np_obs = {k: v.detach().cpu().numpy() for k, v in obs.items()}
        return torch.as_tensor(self._call({"op": "act", "obs": np_obs})["action"])


def load_policy(path: str, device: str):
    """In-process lerobot policy with its pre/post processors. The
    processors are required: lerobot keeps normalisation outside the model."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    policy = get_policy_class(cfg.type).from_pretrained(path)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    keys = list(cfg.input_features.keys())

    class _Wrapped:
        input_keys = keys
        policy_type = cfg.type

        def reset(self):
            policy.reset()

        def select_action(self, obs):
            batch = pre(dict(obs))
            with torch.inference_mode():
                a = policy.select_action(batch)
            return post(a).detach().cpu()

    return _Wrapped()
