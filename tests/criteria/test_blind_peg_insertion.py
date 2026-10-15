# SPDX-License-Identifier: Apache-2.0
"""blind_peg_insertion: the final depth decides; tilt names a lost grasp."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from sim.criteria.blind_peg_insertion import verdict
from sim.criteria.trace import FailureMode


@pytest.fixture
def peg_trace(held_trace):
    """-> make(depth_mm, tilt_deg): a held peg whose depth (scalar or per-step)
    and tilt are given."""
    def make(depth_mm, tilt_deg=2.0, n=100, area=20.0):
        tr = held_trace(n=n, area=area)
        depth = np.asarray(depth_mm, dtype=float) * 1e-3
        tr.extra["insertion_depth_m"] = np.full(n, float(depth)) if depth.ndim == 0 else depth
        tr.extra["peg_tilt_deg"] = np.full(n, tilt_deg)
        return tr
    return make


def test_passes_when_inserted(peg_trace, right_pads):
    v = verdict(peg_trace(30.0), pads=right_pads)
    assert v.success and v.metrics["depth_m"] == pytest.approx(0.030)


def test_reports_unfinished_when_shallow(peg_trace, right_pads):
    assert verdict(peg_trace(10.0), pads=right_pads).failure is FailureMode.UNFINISHED


def test_only_the_final_depth_counts(peg_trace, right_pads):
    depth = np.full(100, 30.0)
    depth[-5:] = 0.0                   # went in, came back out
    v = verdict(peg_trace(depth), pads=right_pads)
    assert v.failure is FailureMode.UNFINISHED and v.metrics["peak_depth_m"] == pytest.approx(0.030)


def test_a_tilted_peg_is_slip(peg_trace, right_pads):
    assert verdict(peg_trace(30.0, tilt_deg=25.0), pads=right_pads).failure is FailureMode.SLIP


def test_never_held_is_a_grasp_failure(peg_trace, right_pads):
    assert verdict(peg_trace(30.0, area=0.0), pads=right_pads).failure is FailureMode.GRASP_FAILURE


def test_threshold_defaults():
    defaults = {k: p.default for k, p in inspect.signature(verdict).parameters.items()}
    assert defaults["min_depth_m"] == 0.025
    assert defaults["max_tilt_deg"] == 20.0
