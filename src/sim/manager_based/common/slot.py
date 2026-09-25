"""Per-slot expert views over a shared scene; only the scheduler advances time."""

import torch

from sim.harness import CTRL_FPS, Harness


class SlotData:
    def __init__(self, data, slot, origin):
        self._data, self._slot, self._origin = data, slot, origin

    def __getattr__(self, name):
        value = getattr(self._data, name)
        if isinstance(value, dict):
            return {key: tensor[self._slot : self._slot + 1] for key, tensor in value.items()}
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            return value
        value = value[self._slot : self._slot + 1]
        if name in ("root_pos_w", "body_pos_w", "tactile_points_pos_w"):
            return value - self._origin
        if name in ("root_state_w", "body_state_w"):
            value = value.clone()
            value[..., :3] -= self._origin
        return value


class SlotPhysics:
    def __init__(self, view, slot):
        self._view, self._slot = view, slot

    def get_jacobians(self):
        return self._view.get_jacobians()[self._slot : self._slot + 1]


class SlotAsset:
    def __init__(self, asset, slot, origin):
        self._asset, self._slot, self._origin = asset, slot, origin

    @property
    def data(self):
        return SlotData(self._asset.data, self._slot, self._origin)

    @property
    def root_physx_view(self):
        return SlotPhysics(self._asset.root_physx_view, self._slot)

    def __getattr__(self, name):
        if name not in (
            "joint_names",
            "body_names",
            "num_joints",
            "num_bodies",
            "find_joints",
            "find_bodies",
            "cfg",
        ):
            raise AttributeError(f"Slot asset does not expose {name}")
        return getattr(self._asset, name)

    def set_joint_position_target(self, target, joint_ids=None):
        self._asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=[self._slot])

    def write_data_to_sim(self):
        # The scheduler flushes the shared scene once per control tick.
        pass


class SlotScene:
    def __init__(self, scene, slot):
        self._scene, self._slot = scene, slot

    def __getitem__(self, name):
        return SlotAsset(self._scene[name], self._slot, self._scene.env_origins[self._slot])

    def keys(self):
        return self._scene.keys()


class SlotHarness(Harness):
    control_dt = 1 / CTRL_FPS

    def __init__(self, env, slot, grid):
        self.env, self.slot = env, slot
        self.sim, self.device = env.sim, env.device
        self.rows, self.cols = grid
        self.scene = SlotScene(env.scene, slot)
        self.cams = {name: self.scene[f"{name}_camera"] for name in ("base", "left_wrist", "right_wrist")}
        self.scene_cam = self.scene["review_camera"]
        self._hold, self._joint_idx = None, None
        self._posture, self._chains, self.ch = {}, {}, {}
        self._bind_scene()

    def step(self, render=True):
        raise RuntimeError("Only the vector scheduler may advance physics")

    def hold_default(self, nsteps):
        raise RuntimeError("Settling must be scheduled across the shared scene")

    def update_cams(self, include_scene_cam=False):
        # Accessing sensor.data refreshes the shared sensor at its current timestamp.
        pass

    def initial_render(self):
        raise RuntimeError("Tactile references are initialized once by the environment")
