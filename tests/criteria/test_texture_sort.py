"""Cup placement uses the original XY radius with sustained completion."""

import numpy as np
import pytest

from sim.criteria.texture_sort import verdict
from sim.dims.texture_sort import CUP_FLOOR, OBJECT_RADIUS


@pytest.mark.parametrize(
    "case, success",
    [
        ("seated", True),
        ("overhead", True),
        ("wrong", False),
        ("rim", True),
        ("held", True),
        ("closed", True),
        ("brief", False),
        ("boundary", False),
        ("inside", True),
    ],
)
def test_placement(held_trace, right_pads, case, success):
    tr = held_trace(n=80)
    pos = tr.object_pos.pop("manipuland")
    pos[:] = [0, 0, CUP_FLOOR]
    tr.object_pos["target"] = pos
    tr.extra["gripper_gap_m"] = np.full(80, 2 * OBJECT_RADIUS + 0.004)
    for area in tr.pad_area.values():
        area[20:] = 0
    if case == "overhead":
        pos[:, 2] += 0.1
    elif case == "wrong":
        pos[:, 0] = 1
    elif case == "rim":
        pos[:, 0] = 0.03
    elif case == "held":
        for area in tr.pad_area.values():
            area[:] = 20
    elif case == "closed":
        tr.extra["gripper_gap_m"][:] = 0.02
    elif case == "brief":
        pos[19:, 0] += 0.1
    elif case == "boundary":
        pos[:, 0] = 0.04
    elif case == "inside":
        pos[:, 0] = 0.0399
    result = verdict(tr, bin_xy=np.array([[0, 0], [1, 0]]), correct_bin=0, pads=right_pads)
    assert result.success is success
