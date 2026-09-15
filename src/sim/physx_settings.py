"""PhysX GPU buffer sizing for single-robot scenes on a modest card."""

from __future__ import annotations

import isaaclab.sim as sim_utils

__all__ = ["small_scene_physx_cfg"]


def small_scene_physx_cfg() -> sim_utils.PhysxCfg:
    """PhysX GPU buffers sized for a 10 GB card.

    The stock defaults make PhysX attempt a 1 GB contiguous allocation, which
    fails on a 10 GB card alongside the RTX renderer. These scenes hold one
    bimanual robot and a handful of objects.
    """
    return sim_utils.PhysxCfg(
        gpu_max_rigid_contact_count=2**18,
        gpu_max_rigid_patch_count=2**14,
        gpu_found_lost_pairs_capacity=2**16,
        gpu_collision_stack_size=2**22,
        gpu_heap_capacity=2**22,
        gpu_temp_buffer_capacity=2**20,
    )
