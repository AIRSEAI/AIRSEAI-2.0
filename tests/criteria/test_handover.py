"""handover: the final seat decides; the fall and the gap window name a failure."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from sim.criteria.handover import verdict
from sim.criteria.trace import FailureMode


@pytest.fixture
def handover_trace(held_trace, right_pads, left_pads):
    """-> make(...): the giver holds, the receiver takes over, the sleeve ends
    seated and let go."""
    def make(n=100, dt=0.05, release_step=60, catch_step=50, seat_step=80, gap_mm=4.0):
        tr = held_trace(n=n, dt=dt)
        giver = np.zeros(n)
        giver[:release_step] = 20.0             # the giver holds, then lets go
        receiver = np.zeros(n)
        receiver[catch_step:seat_step] = 20.0   # the receiver takes over, then sets it down
        for p in left_pads:
            tr.pad_area[p] = giver
        for p in right_pads:
            tr.pad_area[p] = receiver
        tr.object_pos["sleeve"] = tr.object_pos.pop("manipuland")
        tr.extra["gap_m"] = np.full(n, gap_mm * 1e-3)
        over = np.zeros(n)
        over[seat_step - 10:] = 1.0             # above the target for the last stretch
        tr.extra["over_target"] = over
        seat = np.zeros(n)
        seat[seat_step:] = 0.020                # down over the key once set down
        tr.extra["seat_depth_m"] = seat
        return tr
    return make


@pytest.fixture
def judge(right_pads, left_pads):
    """-> judge(trace, **thresholds): the giver is the left arm."""
    def call(trace, **kw):
        return verdict(trace, giver_pads=left_pads, receiver_pads=right_pads, **kw)
    return call


def test_accepts_a_seated_sleeve(handover_trace, judge):
    v = judge(handover_trace())
    assert v.success, v.metrics


def test_a_fall_is_coordination(handover_trace, judge):
    tr = handover_trace()
    pos = tr.object_pos["sleeve"].copy()
    pos[70:, 2] -= 0.06                # onto the table
    tr.object_pos["sleeve"] = pos
    tr.extra["seat_depth_m"] = np.zeros(len(tr.t))     # and stayed there
    assert judge(tr).failure is FailureMode.COORDINATION


def test_a_fall_does_not_overrule_a_seat(handover_trace, judge):
    tr = handover_trace()
    pos = tr.object_pos["sleeve"].copy()
    pos[30:40, 2] -= 0.06              # dropped, picked up again, seated
    tr.object_pos["sleeve"] = pos
    v = judge(tr)
    assert v.success and v.metrics["sleeve_drop_m"] == pytest.approx(0.06)


def test_needs_the_giver_to_have_held(handover_trace, judge, left_pads):
    tr = handover_trace()
    for p in left_pads:
        tr.pad_area[p] = np.zeros(len(tr.t))
    assert judge(tr).failure is FailureMode.GRASP_FAILURE


def test_window_too_open_is_coordination(handover_trace, judge):
    tr = handover_trace(gap_mm=7.0)    # wider than the 42 mm bore admits
    tr.extra["seat_depth_m"] = np.zeros(len(tr.t))     # resting on the rim
    v = judge(tr)
    assert v.failure is FailureMode.COORDINATION
    assert v.metrics["gap_at_target_m"] > v.metrics["gap_hi_m"]


def test_window_too_closed_is_overforce(handover_trace, judge):
    tr = handover_trace(gap_mm=1.0)    # squeezed below the key
    tr.extra["seat_depth_m"] = np.zeros(len(tr.t))     # resting on the key
    v = judge(tr)
    assert v.failure is FailureMode.OVERFORCE
    assert v.metrics["gap_at_target_m"] < v.metrics["gap_lo_m"]


def test_a_gap_outside_the_window_does_not_overrule_a_seat(handover_trace, judge):
    tr = handover_trace(gap_mm=2.0)    # wedged open over the key and seated anyway
    v = judge(tr)
    assert v.success
    assert v.metrics["gap_at_target_m"] < v.metrics["gap_lo_m"]


def test_too_open_and_still_held_is_coordination(handover_trace, judge):
    tr = handover_trace(gap_mm=7.0, seat_step=100)     # never went in, never let go
    tr.extra["over_target"][-10:] = 1.0
    assert judge(tr).failure is FailureMode.COORDINATION


def test_is_unfinished_until_over_the_target_and_let_go(handover_trace, judge):
    tr = handover_trace()
    tr.extra["over_target"] = np.zeros(len(tr.t))
    tr.extra["seat_depth_m"] = np.zeros(len(tr.t))     # never reached the target
    v = judge(tr)
    assert v.failure is FailureMode.UNFINISHED and np.isnan(v.metrics["gap_at_target_m"])
    tr = handover_trace(seat_step=100)                 # seated but still gripped
    tr.extra["seat_depth_m"] = np.full(len(tr.t), 0.020)
    tr.extra["over_target"][-10:] = 1.0
    v = judge(tr)
    assert v.failure is FailureMode.UNFINISHED and v.metrics["released"] == 0.0


def test_reads_the_gap_while_the_receiver_holds(handover_trace, judge):
    tr = handover_trace(seat_step=85, gap_mm=4.0)
    tr.extra["seat_depth_m"] = np.zeros(len(tr.t))     # never went in
    gap = tr.extra["gap_m"].copy()
    gap[85:] = 0.010                   # sprang open once let go over the target
    tr.extra["gap_m"] = gap
    v = judge(tr)
    assert v.metrics["gap_at_target_m"] == pytest.approx(0.004)
    assert v.failure is FailureMode.UNFINISHED


def test_judges_the_final_seat(handover_trace, judge):
    tr = handover_trace()
    seat = tr.extra["seat_depth_m"].copy()
    seat[-1] = 0.0                     # pulled back out at the very end
    tr.extra["seat_depth_m"] = seat
    assert judge(tr).failure is FailureMode.UNFINISHED


def test_reports_but_does_not_judge_the_release_timing(handover_trace, judge):
    tr = handover_trace(catch_step=20, release_step=95, seat_step=96)  # both held 3.75 s
    v = judge(tr)
    assert v.success and v.metrics["gap_s"] == pytest.approx(3.70)


def test_threshold_defaults():
    """Pins the bore, the seat depth and the fall height the task card asks for."""
    defaults = {k: v.default for k, v in inspect.signature(verdict).parameters.items()}
    assert defaults["bore_m"] == 0.042
    assert defaults["min_seat_m"] == 0.010
    assert defaults["drop_height_m"] == 0.05
