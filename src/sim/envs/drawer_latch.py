"""The standard workstation with one spring-return tool drawer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from sim.assets.provision import task_asset_dir
from sim.dims.drawer_latch import (
    HANDLE_CENTER_Z,
    MAX_EXTENSION,
    YAW_CENTER_DEG,
)
from sim.envs.bimanual_tactile import (
    REAL_HOME_JOINT_POS,
    TABLE_TOP_Z,
    BimanualTactileSceneCfg,
    point_pads_at,
    start_joint_pos,
)
from sim.envs.preview import PreviewSpec

__all__ = [
    "ASSETS_DIR",
    "CASE_XY",
    "CUE_LEAD_RANGE",
    "DRAWER_QUAT",
    "DRAWER_Z",
    "DrawerLatchSceneCfg",
    "DrawerMechanism",
    "DriveSetting",
    "JOINT_NAME",
    "PAD_TARGET",
    "PREVIEW",
    "START_JAW_WIDTH",
    "START_JOINT_POS",
    "create_joint",
    "ensure_assets",
    "make_preview_cfg",
    "reset_drive",
    "set_drive",
    "set_joint_frame",
]

ASSETS_DIR = task_asset_dir("drawer_latch")
CASE_XY = (0.46, 0.0)
DRAWER_QUAT = (
    math.cos(math.radians(YAW_CENTER_DEG) / 2),
    0.0,
    0.0,
    math.sin(math.radians(YAW_CENTER_DEG) / 2),
)
DRAWER_Z = TABLE_TOP_Z
START_JOINT_POS = REAL_HOME_JOINT_POS
START_JAW_WIDTH = {"right": 0.0, "left": 0.0}
PAD_TARGET = "{ENV_REGEX_NS}/drawer"


ENV = "/World/envs/env_0"
JOINT_NAME = "drawer_slide"

RETURN_STIFFNESS = 28.0
RETURN_DAMPING = 30.0
WELL_EXTRA_DAMPING = 100.0
DRIVE_STIFFNESS = 500.0
CUE_LEAD_RANGE = (0.010, 0.022)
MAX_FORCE_N = 20.0
WELL_ENERGY_J = 0.025
CUE_RESISTANCE_N = 4.0
CUE_WIDTH_M = 0.008
CUE_SPEED_SCALE_M_S = 0.002
WELL_WIDTH_M = 0.006
RESET_EXTENSION_M = 0.006
MECHANISM_VERSION = "finite_engagement"


def _bump(x: float, centre: float, half_width: float, energy: float) -> tuple[float, float]:
    """Compact quartic potential and its negative gradient; both zero outside."""
    u = (x - centre) / half_width
    if abs(u) >= 1.0:
        return 0.0, 0.0
    return energy * (1.0 - u * u) ** 2, 4.0 * energy * u * (1.0 - u * u) / half_width


@dataclass(frozen=True)
class DriveSetting:
    target_m: float
    stiffness_n_m: float
    damping_n_s_m: float
    target_velocity_m_s: float = 0.0
    max_force_n: float = MAX_FORCE_N


class DrawerMechanism:
    def __init__(self, detent_m: float, cue_lead_m: float = 0.014):
        self.detent_m = float(detent_m)
        self.cue_lead_m = float(cue_lead_m)
        if not (self.detent_m - self.cue_lead_m - CUE_WIDTH_M > RESET_EXTENSION_M):
            raise ValueError("Cue support must lie beyond the closed-position reset zone")
        if not (self.cue_lead_m + WELL_WIDTH_M > CUE_WIDTH_M):
            raise ValueError("Cue support must end before the overtravel boundary")
        self.bypassed = False
        self.escape_count = 0
        self.reset_count = 0
        self.mode = "search"
        self.dwell_s = 0.0
        self.latched_extension_m: float | None = None
        self.force_n = 0.0
        self.cue_resistance_n = 0.0
        self.potential_j = 0.0

    def potential(self, x: float) -> float:
        energy = 0.5 * RETURN_STIFFNESS * x * x
        if self.bypassed:
            return energy
        well, _ = _bump(x, self.detent_m, WELL_WIDTH_M, -WELL_ENERGY_J)
        return energy + well

    def conservative_force(self, x: float) -> float:
        force = -RETURN_STIFFNESS * x
        if self.bypassed:
            return force
        _, well = _bump(x, self.detent_m, WELL_WIDTH_M, -WELL_ENERGY_J)
        return force + well

    def cue_resistance(self, x: float, velocity_m_s: float) -> float:
        """Outward-only dissipative load, smoothly zero at rest and cue edges."""
        if self.bypassed or velocity_m_s <= 0.0:
            return 0.0
        u = (x - (self.detent_m - self.cue_lead_m)) / CUE_WIDTH_M
        if abs(u) >= 1.0:
            return 0.0
        envelope = (1.0 - u * u) ** 2
        return CUE_RESISTANCE_N * envelope * velocity_m_s / (velocity_m_s + CUE_SPEED_SCALE_M_S)

    def damping(self, x: float) -> float:
        """Localized contact damping dissipates entry energy without a capture gate."""
        if self.bypassed:
            return RETURN_DAMPING
        u = (x - self.detent_m) / WELL_WIDTH_M
        if abs(u) >= 1.0:
            return RETURN_DAMPING
        return RETURN_DAMPING + WELL_EXTRA_DAMPING * (1.0 - u * u) ** 2

    def update(self, extension_m: float, velocity_m_s: float, dt: float) -> DriveSetting:
        x, v = float(extension_m), float(velocity_m_s)
        # Overtravel stays bypassed until the drawer returns to the reset zone.
        if self.bypassed and x <= RESET_EXTENSION_M:
            self.bypassed = False
            self.reset_count += 1
        elif not self.bypassed and x >= self.detent_m + WELL_WIDTH_M:
            self.bypassed = True
            self.escape_count += 1
        force = self.conservative_force(x)
        self.cue_resistance_n = self.cue_resistance(x, v)
        damping = self.damping(x)
        self.force_n = max(-MAX_FORCE_N, min(MAX_FORCE_N, force - damping * v - self.cue_resistance_n))
        self.potential_j = self.potential(x)
        # Dwell is diagnostic only; no threshold enables engagement.
        in_well = not self.bypassed and abs(x - self.detent_m) < WELL_WIDTH_M
        self.mode = "bypassed" if self.bypassed else ("engaged" if in_well else "search")
        self.dwell_s = self.dwell_s + dt if in_well else 0.0
        self.latched_extension_m = x if in_well else None
        # Joint transX is minus extension; bound the drive target in concave regions.
        return DriveSetting(
            -x - (force - self.cue_resistance_n) / DRIVE_STIFFNESS,
            DRIVE_STIFFNESS,
            damping,
        )


def create_joint(env_path: str) -> str:
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
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(MAX_FORCE_N)
    return path


def set_drive(path: str, setting: DriveSetting) -> None:
    import omni.usd
    from pxr import UsdPhysics

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    drive = UsdPhysics.DriveAPI.Apply(prim, "transX")
    drive.GetTargetPositionAttr().Set(float(setting.target_m))
    drive.GetStiffnessAttr().Set(float(setting.stiffness_n_m))
    drive.GetDampingAttr().Set(float(setting.damping_n_s_m))
    drive.GetTargetVelocityAttr().Set(float(setting.target_velocity_m_s))
    drive.GetMaxForceAttr().Set(float(setting.max_force_n))


def set_joint_frame(env_path: str, yaw_deg: float) -> None:
    import omni.usd
    from pxr import Gf, UsdPhysics

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(f"{env_path}/{JOINT_NAME}")
    joint = UsdPhysics.Joint(prim)
    half = math.radians(yaw_deg - YAW_CENTER_DEG) / 2
    joint.GetLocalRot0Attr().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))


def reset_drive(env_path: str = ENV, yaw_deg: float = YAW_CENTER_DEG) -> None:
    set_joint_frame(env_path, yaw_deg)
    set_drive(f"{env_path}/{JOINT_NAME}", DriveSetting(0.0, RETURN_STIFFNESS, RETURN_DAMPING))


def ensure_assets() -> None:
    missing = [name for name in ("case", "drawer") if not (ASSETS_DIR / f"{name}.usd").exists()]
    if missing:
        raise FileNotFoundError(
            f"drawer_latch geometry missing: {', '.join(missing)}; "
            "run python benchmark/run/make_drawer_latch_assets.py --headless"
        )


def _case(prim: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "case.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CASE_XY, DRAWER_Z), rot=DRAWER_QUAT),
    )


def _drawer(prim: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "drawer.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=0.2,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0015),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CASE_XY, DRAWER_Z), rot=DRAWER_QUAT),
    )


@configclass
class DrawerLatchSceneCfg(BimanualTactileSceneCfg):
    """Standard workstation with a spring-return drawer."""

    manipuland = None

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.75, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.66, 0.55)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.0, TABLE_TOP_Z - 0.02)),
    )

    case = _case("{ENV_REGEX_NS}/case")
    drawer = _drawer("{ENV_REGEX_NS}/drawer")

    def __post_init__(self):
        ensure_assets()
        self.robot.init_state.joint_pos = start_joint_pos(START_JOINT_POS, START_JAW_WIDTH)
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        point_pads_at(self, PAD_TARGET)


PREVIEW = PreviewSpec(
    sdf_prims=("drawer",),
    work_target=(CASE_XY[0], CASE_XY[1], TABLE_TOP_Z + HANDLE_CENTER_Z),
)


def make_preview_cfg() -> DrawerLatchSceneCfg:
    return DrawerLatchSceneCfg(num_envs=1, env_spacing=3.0)
