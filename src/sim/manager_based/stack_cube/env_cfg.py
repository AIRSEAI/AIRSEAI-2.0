"""stack_cube uses the shared native-batch environment configuration."""

from sim.envs import stack_cube as scene
from sim.manager_based.common.env_cfg import build_cfg


def make_cfg(**kwargs):
    return build_cfg(scene, **kwargs)
