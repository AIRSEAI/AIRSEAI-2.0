# SPDX-License-Identifier: Apache-2.0
"""Build independent box-transfer scenes in one simulation."""

from sim.envs import loose_parts_box_transfer as scene
from sim.manager_based.common.env_cfg import build_cfg


def make_cfg(**kwargs):
    return build_cfg(scene, **kwargs)
