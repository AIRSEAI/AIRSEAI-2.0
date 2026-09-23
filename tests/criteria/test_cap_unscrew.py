"""cap_unscrew: where the cap ends decides; the bottle's spin names a failure."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from sim.criteria.cap_unscrew import verdict
from sim.criteria.trace import FailureMode
from sim.dims.cap_unscrew import BOTTLE_H, CAP_BORE_DEPTH, TRAY_FLOOR, TRAY_INNER_R

BOTTLE = (0.22, 0.0, 0.2)
TRAY = (0.24, -0.17, 0.2)
UPRIGHT = (1.0, 0.0, 0.0, 0.0)
ON_NECK = (BOTTLE[0], BOTTLE[1], BOTTLE[2] + BOTTLE_H - CAP_BORE_DEPTH)
IN_TRAY = (TRAY[0], TRAY[1], TRAY[2] + TRAY_FLOOR)


@pytest.fixture
def cap_trace(held_trace, right_pads, left_pads):
    """-> make(...): both hands hold, the cap is turned off and set in the tray,
    both hands let go."""
    def make(n=100, dt=0.05, end_xyz=IN_TRAY, end_quat=UPRIGHT, release_step=90,
             thread_deg=130.0, bottle_deg=3.0):
        tr = held_trace(n=n, dt=dt)
        tr.object_pos.pop("manipuland")
        for p in right_pads + left_pads:
            area = np.full(n, 20.0)
            area[release_step:] = 0.0
            tr.pad_area[p] = area
        cap = np.tile(np.asarray(ON_NECK, dtype=float), (n, 1))
        cap[60:] = end_xyz
        tr.object_pos["cap"] = cap
        tr.object_quat["cap"] = np.tile(np.asarray(end_quat, dtype=float), (n, 1))
        tr.object_pos["bottle"] = np.tile(np.asarray(BOTTLE, dtype=float), (n, 1))
        tr.extra["thread_deg"] = np.linspace(0.0, thread_deg, n)
        tr.extra["bottle_deg"] = np.linspace(0.0, bottle_deg, n)
        return tr
    return make


@pytest.fixture
def judge(right_pads, left_pads):
    """-> judge(trace, **thresholds): the right hand turns, the left holds."""
    def call(trace, **kw):
        return verdict(trace, tray_pos=TRAY, turner_pads=right_pads,
                       holder_pads=left_pads, **kw)
    return call


def test_accepts_a_cap_in_the_tray(cap_trace, judge):
    v = judge(cap_trace())
    assert v.success, v.metrics


def test_a_cap_on_its_side_in_the_tray_still_counts(cap_trace, judge):
    side = (0.7071068, 0.7071068, 0.0, 0.0)
    v = judge(cap_trace(end_quat=side))
    assert v.success, v.metrics


def test_a_cap_still_held_is_not_done(cap_trace, judge):
    assert not judge(cap_trace(release_step=100)).success


def test_a_cap_left_on_the_bottle_is_unfinished(cap_trace, judge):
    tr = cap_trace(end_xyz=ON_NECK, thread_deg=60.0)
    assert judge(tr).failure is FailureMode.UNFINISHED


def test_a_bottle_that_spun_with_its_cap_is_coordination(cap_trace, judge):
    tr = cap_trace(end_xyz=ON_NECK, thread_deg=10.0, bottle_deg=90.0)
    assert judge(tr).failure is FailureMode.COORDINATION


def test_a_cap_dropped_beside_the_tray_is_slip(cap_trace, judge):
    beside = (TRAY[0], TRAY[1] + TRAY_INNER_R + 0.02, TRAY[2])
    assert judge(cap_trace(end_xyz=beside)).failure is FailureMode.SLIP


def test_a_turning_hand_that_never_held_is_grasp_failure(cap_trace, judge, right_pads):
    tr = cap_trace(end_xyz=ON_NECK)
    for p in right_pads:
        tr.pad_area[p] = np.zeros(len(tr.t))
    assert judge(tr).failure is FailureMode.GRASP_FAILURE


def test_tray_radius_defaults_to_the_tray_bore():
    assert inspect.signature(verdict).parameters["tray_r"].default == TRAY_INNER_R


def test_the_verdict_survives_decimation(cap_trace, judge, decimate):
    tr = cap_trace()
    tr_d = decimate(tr, 3)
    tr_d.object_quat = {k: v[::3] for k, v in tr.object_quat.items()}
    assert judge(tr_d).success == judge(tr).success
