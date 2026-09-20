"""sleeve_on_shaft: seat depth, bore margin and their precedence."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from sim.criteria.sleeve_on_shaft import verdict
from sim.criteria.trace import FailureMode


@pytest.fixture
def sleeve_trace(held_trace):
    """-> make(seat_mm, margin_mm, ...): a held sleeve whose seat depth and
    bore margin (mm, scalar or per-step) are given."""
    def make(seat_mm, margin_mm=2.0, n=100, dt=0.05, area=20.0):
        tr = held_trace(n=n, dt=dt, area=area)
        seat = np.asarray(seat_mm, dtype=float) * 1e-3
        tr.extra["seat_depth_m"] = np.full(n, float(seat)) if seat.ndim == 0 else seat
        margin = np.asarray(margin_mm, dtype=float) * 1e-3
        tr.extra["bore_margin_m"] = np.full(n, float(margin)) if margin.ndim == 0 else margin
        return tr
    return make


def test_passes_when_seated(sleeve_trace, right_pads):
    v = verdict(sleeve_trace(20.0), pads=right_pads)
    assert v.success and v.metrics["seat_depth_m"] == pytest.approx(0.020)


def test_reports_unfinished_when_shallow(sleeve_trace, right_pads):
    assert verdict(sleeve_trace(5.0), pads=right_pads).failure is FailureMode.UNFINISHED


def test_reports_overforce_when_the_bore_was_shut_and_not_seated(sleeve_trace, right_pads):
    v = verdict(sleeve_trace(0.0, margin_mm=-0.5), pads=right_pads)
    assert v.failure is FailureMode.OVERFORCE
    assert v.metrics["min_bore_margin_m"] == pytest.approx(-0.0005)


def test_seated_after_a_squeeze_off_the_shaft_is_a_success(sleeve_trace, right_pads):
    seat = np.full(100, 20.0)
    seat[:40] = 0.0
    margin = np.full(100, 2.0)
    margin[10:20] = -1.0
    assert verdict(sleeve_trace(seat, margin_mm=margin), pads=right_pads).success


def test_forced_on_through_a_shut_bore_is_overforce_even_if_seated(sleeve_trace, right_pads):
    seat = np.full(100, 20.0)
    seat[:40] = 0.0
    margin = np.full(100, 2.0)
    margin[45:60] = -0.5
    v = verdict(sleeve_trace(seat, margin_mm=margin), pads=right_pads)
    assert v.failure is FailureMode.OVERFORCE
    assert v.metrics["min_bore_margin_on_shaft_m"] == pytest.approx(-0.0005)


def test_shells_jostling_after_a_drop_are_not_overforce(sleeve_trace, right_pads):
    tr = sleeve_trace(0.0)
    tr.pad_area = {p: np.where(np.arange(100) < 50, 20.0, 0.0) for p in tr.pad_area}
    margin = np.full(100, 2.0)
    margin[70:75] = -0.03
    tr.extra["bore_margin_m"] = margin * 1e-3
    assert verdict(tr, pads=right_pads).failure is FailureMode.UNFINISHED


def test_never_held_is_a_grasp_failure(sleeve_trace, right_pads):
    v = verdict(sleeve_trace(20.0, area=0.0), pads=right_pads)
    assert v.failure is FailureMode.GRASP_FAILURE


def test_only_the_final_seat_depth_counts(sleeve_trace, right_pads):
    seat = np.full(100, 20.0)
    seat[-5:] = 0.0
    assert not verdict(sleeve_trace(seat), pads=right_pads).success


def test_says_so_when_no_seat_was_recorded(held_trace, right_pads):
    v = verdict(held_trace(), pads=right_pads)
    assert not v.success and any("seat_depth_m" in n for n in v.notes)


def test_seat_threshold_moves_the_verdict(sleeve_trace, right_pads):
    tr = sleeve_trace(12.0)
    assert not verdict(tr, pads=right_pads).success
    assert verdict(tr, min_seat_m=0.010, pads=right_pads).success


def test_every_failing_verdict_names_a_mode(sleeve_trace, right_pads):
    """The failure-distribution figure has no 'other' bucket by construction."""
    cases = [verdict(sleeve_trace(5.0), pads=right_pads),
             verdict(sleeve_trace(20.0, area=0.0), pads=right_pads)]
    for v in cases:
        assert not v.success
        assert v.failure is not FailureMode.NONE
        assert v.as_dict()["failure"] in {m.value for m in FailureMode}


def test_verdict_survives_decimation_240hz_to_20hz(sleeve_trace, decimate, right_pads):
    """An episode recorded at either rate gets the same verdict."""
    fast = sleeve_trace(20.0, n=1200, dt=1 / 240)
    slow = decimate(fast, 12)
    a = verdict(fast, pads=right_pads)
    b = verdict(slow, pads=right_pads)
    assert a.success == b.success and a.failure is b.failure
    assert a.metrics["seat_depth_m"] == pytest.approx(b.metrics["seat_depth_m"])


def test_threshold_defaults():
    """Pins the seat depth the task card asks for."""
    defaults = {k: v.default for k, v in inspect.signature(verdict).parameters.items()}
    assert defaults["min_seat_m"] == 0.015
