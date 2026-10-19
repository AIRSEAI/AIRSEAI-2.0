# SPDX-License-Identifier: Apache-2.0
"""How lerobot builds and reads an AIRSEAI environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from gym_airseai.tasks import TASK_IDS, VARIANTS

# Tactile variants add one image per finger to the camera observations.
CAMERAS = ("base", "right_wrist", "left_wrist")
PADS = ("right_middle", "right_index", "left_middle", "left_index")
CAMERA_HW = (480, 640)
PAD_HW = (288, 384)
STATE_DIM = ACTION_DIM = 16


@EnvConfig.register_subclass("airseai")
@dataclass
class AirseaiEnvConfig(EnvConfig):
    task: str = "stack_cube"
    variant: str = "tacff"
    fps: int = 20
    episode_length: int | None = None  # None uses the task's own horizon
    init_states: bool = True
    init_states_path: str | None = None
    render_scene_cam: bool = False
    repo_root: str | None = None
    sim_python: str | None = None
    device: str = "cuda"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    )
    features_map: dict[str, PolicyFeature] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in TASK_IDS:
            raise ValueError(f"unknown task {self.task!r}; one of {TASK_IDS}")
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; one of {VARIANTS}")
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
        self.features_map = {ACTION: ACTION, "agent_pos": OBS_STATE}
        for name in self.image_names:
            hw = PAD_HW if name.startswith(("tacff_", "tacrgb_")) else CAMERA_HW
            self.features[name] = PolicyFeature(type=FeatureType.VISUAL, shape=(*hw, 3))
            self.features_map[name] = f"{OBS_IMAGES}.{name}"

    @property
    def image_names(self) -> tuple[str, ...]:
        if self.variant == "vision":
            return CAMERAS
        return CAMERAS + tuple(f"{self.variant}_{p}" for p in PADS)

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task": self.task,
            "variant": self.variant,
            "init_states": self.init_states,
            "init_states_path": self.init_states_path,
            "render_scene_cam": self.render_scene_cam,
            "horizon_s": self.episode_length,
            "repo_root": self.repo_root,
            "sim_python": self.sim_python,
            "device": self.device,
        }
