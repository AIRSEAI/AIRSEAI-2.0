"""stack_cube: the final pose of cube A on cube B decides."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from sim.criteria.stack_cube import verdict
from sim.criteria.trace import FailureMode
from sim.dims.stack_cube import CUBE_SIZE


@pytest.fixture
def cube_trace(held_trace):
    """-> make(dxy, dz): cube A ends `dxy` beside and `dz` above cube B."""
    def make(dxy=0.0, dz=CUBE_SIZE, n=100, area=20.0):
        tr = held_trace(n=n, area=area)
        b = tr.object_pos.pop("manipuland")
        a = b.copy()
        a[-1, 0] += dxy
        a[-1, 2] += dz
        tr.object_pos["cube_a"] = a
        tr.object_pos["cube_b"] = b
        return tr
    return make


def test_passes_when_stacked(cube_trace, right_pads):
    v = verdict(cube_trace(), pads=right_pads)
    assert v.success and v.metrics["z_err_m"] == pytest.approx(0.0)


def test_reports_unfinished_when_beside(cube_trace, right_pads):
    v = verdict(cube_trace(dxy=CUBE_SIZE, dz=0.0), pads=right_pads)
    assert v.failure is FailureMode.UNFINISHED


def test_reports_unfinished_when_off_centre(cube_trace, right_pads):
    v = verdict(cube_trace(dxy=0.6 * CUBE_SIZE), pads=right_pads)
    assert v.failure is FailureMode.UNFINISHED


def test_never_held_is_a_grasp_failure(cube_trace, right_pads):
    assert verdict(cube_trace(area=0.0), pads=right_pads).failure is FailureMode.GRASP_FAILURE


def test_threshold_defaults():
    defaults = {k: p.default for k, p in inspect.signature(verdict).parameters.items()}
    assert defaults["cube_size"] == CUBE_SIZE
    assert np.isclose(defaults["area_mm2"], 5.0)
