"""Unit tests for the success criteria, runnable without Isaac or a GPU.

Hand-built traces that trigger one failure mode each (pinning precedence),
pinned threshold defaults, and decimation invariance
(240 Hz and 20 Hz give the same verdict).

    python -m pytest tests/test_criteria.py -q
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from sim.criteria import FailureMode, detect_mistiming
from sim.criteria.tasks import (
    bimanual_lift,
    handover,
    hold_and_screw,
    sleeve_on_shaft,
    slip_detect_regrasp,
)
from sim.criteria.trace import EpisodeTrace, Verdict

PAD_R1, PAD_R2 = "right_left_finger_tactile", "right_right_finger_tactile"
PAD_L1, PAD_L2 = "left_left_finger_tactile", "left_right_finger_tactile"


def _trace(n=100, dt=0.05, area=20.0, obj_xyz=(0.3, 0.0, 0.2)):
    """A clean episode: object held steadily at a fixed pose."""
    t = np.arange(n) * dt
    pos = np.tile(np.asarray(obj_xyz, dtype=float), (n, 1))
    return EpisodeTrace(
        t=t,
        pad_area={p: np.full(n, area) for p in (PAD_R1, PAD_R2, PAD_L1, PAD_L2)},
        object_pos={"manipuland": pos},
    )


# -- verdict hygiene --------------------------------------------------------

def test_verdict_rejects_inconsistent_states():
    with pytest.raises(ValueError):
        Verdict(True, FailureMode.SLIP)
    with pytest.raises(ValueError):
        Verdict(False)


# -- slip_detect_regrasp ----------------------------------------------------

def test_slip_is_detected_and_named():
    tr = _trace()
    pos = tr.object_pos["manipuland"].copy()
    pos[50:, 2] -= 0.02               # slid 20 mm down within the grasp
    tr.object_pos["manipuland"] = pos
    v = slip_detect_regrasp(tr)
    assert v.failure is FailureMode.SLIP
    assert v.metrics["slip_m"] == pytest.approx(0.02)


def test_carrying_the_object_is_not_slip():
    # the whole grasp translates: displacement is large, slip is zero.
    tr = _trace()
    pos = tr.object_pos["manipuland"].copy()
    pos[:, 0] += np.linspace(0, 0.5, len(pos))
    tr.object_pos["manipuland"] = pos
    tr.extra["grasp_frame_pos"] = pos.copy()
    v = slip_detect_regrasp(tr)
    assert v.success, v.metrics


# -- bimanual_lift ----------------------------------------------------------

def test_bimanual_lift_succeeds_on_a_steady_carry():
    v = bimanual_lift(_trace(), target_xy=(0.3, 0.0))
    assert v.success


def test_bimanual_lift_fails_when_the_load_falls():
    tr = _trace()
    pos = tr.object_pos["manipuland"].copy()
    pos[60:, 2] -= 0.10               # fell 10 cm
    tr.object_pos["manipuland"] = pos
    v = bimanual_lift(tr, target_xy=(0.3, 0.0))
    assert v.failure is FailureMode.SLIP


# -- handover ---------------------------------------------------------------

def _handover_trace(n=100, dt=0.05, release_step=60, catch_step=50):
    tr = _trace(n=n, dt=dt)
    giver = np.zeros(n)
    giver[:release_step] = 20.0        # left pad holds, then lets go
    receiver = np.zeros(n)
    receiver[catch_step:] = 20.0       # right pad takes over
    tr.pad_area[PAD_L1] = giver
    tr.pad_area[PAD_R1] = receiver
    tr.extra["tilt_deg"] = np.zeros(n)
    return tr


def test_handover_accepts_a_short_overlap():
    # receiver has load at 2.5 s, giver releases at 3.0 s: 0.5 s gap, at the bar
    tr = _handover_trace(catch_step=50, release_step=60)
    v = handover(tr)
    assert v.success, v.metrics


def test_handover_flags_release_before_the_catch():
    tr = _handover_trace(catch_step=70, release_step=50)   # negative gap
    v = handover(tr)
    assert v.failure is FailureMode.MISTIMING
    assert v.metrics["gap_s"] < 0


def test_handover_flags_holding_on_too_long():
    tr = _handover_trace(catch_step=20, release_step=95)   # 3.75 s of tug-of-war
    assert handover(tr).failure is FailureMode.MISTIMING


def test_handover_tilt_is_judged_only_up_to_the_release():
    """A bar that sags after the giver has let go is a single-arm carry."""
    tr = _handover_trace(catch_step=50, release_step=60)
    tilt = np.zeros(len(tr.t))
    tilt[70:] = 40.0                       # sags long after the giver let go
    tr.extra["tilt_deg"] = tilt
    v = handover(tr)
    assert v.success, v.metrics
    # ... and the number is still reported rather than quietly dropped.
    assert v.metrics["peak_tilt_deg"] == 0.0
    assert v.metrics["peak_tilt_after_deg"] == 40.0


def test_handover_still_fails_on_tilt_during_the_transfer():
    """The window is narrower, not absent."""
    tr = _handover_trace(catch_step=50, release_step=60)
    tilt = np.zeros(len(tr.t))
    tilt[40:55] = 40.0                     # swinging while both still hold
    tr.extra["tilt_deg"] = tilt
    v = handover(tr)
    assert v.failure is FailureMode.COORDINATION
    assert v.metrics["peak_tilt_deg"] == 40.0


def test_mistiming_needs_both_contacts_to_exist():
    tr = _handover_trace()
    tr.pad_area[PAD_R1] = np.zeros(len(tr.t))              # nobody caught it
    flagged, m = detect_mistiming(tr, PAD_R1, PAD_L1, 0.5)
    assert flagged and np.isnan(m["gap_s"])


# -- hold_and_screw ---------------------------------------------------------

@pytest.mark.parametrize("final,expected", [
    (0.30, FailureMode.NONE),        # on target
    (0.39, FailureMode.NONE),        # inside the +-0.10 window
    (0.55, FailureMode.OVERFORCE),   # over-tightened
    (0.10, FailureMode.UNFINISHED),  # under-tightened
])
def test_hold_and_screw_torque_window(final, expected):
    tr = _trace()
    n = len(tr.t)
    tr.extra["cap_torque_nm"] = np.linspace(0.0, final, n)
    tr.extra["bottle_tilt_deg"] = np.zeros(n)
    v = hold_and_screw(tr)
    assert v.failure is expected


def test_hold_and_screw_bottle_rotation_outranks_a_good_torque():
    tr = _trace()
    n = len(tr.t)
    tr.extra["cap_torque_nm"] = np.full(n, 0.30)
    tr.extra["bottle_tilt_deg"] = np.full(n, 25.0)   # the bottle turned
    assert hold_and_screw(tr).failure is FailureMode.COORDINATION


# -- cross-cutting properties ----------------------------------------------

def test_verdict_survives_decimation_240hz_to_20hz():
    """Configurations recorded at different rates must get the same verdict."""
    fast = _trace(n=1200, dt=1 / 240)
    slow = EpisodeTrace(
        t=fast.t[::12],
        pad_area={k: v[::12] for k, v in fast.pad_area.items()},
        object_pos={k: v[::12] for k, v in fast.object_pos.items()},
    )
    a = bimanual_lift(fast, target_xy=(0.3, 0.0))
    b = bimanual_lift(slow, target_xy=(0.3, 0.0))
    assert a.success == b.success and a.failure is b.failure
    assert a.metrics["place_err_m"] == pytest.approx(b.metrics["place_err_m"])


def test_threshold_defaults():
    """Pins every criterion's default thresholds."""
    expected = {
        slip_detect_regrasp: {"max_slip_m": 0.005, "hold_s": 5.0},
        bimanual_lift: {"place_tol_m": 0.02, "drop_height_m": 0.05},
        handover: {"max_gap_s": 0.5, "max_tilt_deg": 15.0},
        hold_and_screw: {"torque_target_nm": 0.30, "torque_tol_nm": 0.10,
                         "max_bottle_tilt_deg": 10.0},
        sleeve_on_shaft: {"min_seat_m": 0.015},
    }
    for fn, wanted in expected.items():
        defaults = {k: v.default for k, v in inspect.signature(fn).parameters.items()}
        for name, value in wanted.items():
            assert defaults[name] == value, f"{fn.__name__}.{name}"


def test_no_force_valued_thresholds_remain():
    """Criteria judge geometry, not force."""
    for fn in (slip_detect_regrasp, bimanual_lift, handover,
               hold_and_screw, sleeve_on_shaft):
        for name in inspect.signature(fn).parameters:
            assert "f_max" not in name and "force" not in name, \
                f"{fn.__name__}.{name} looks force-valued"


# -- sleeve_on_shaft ----------------------------------------------------------

def _sleeve(seat_mm, margin_mm=2.0, n=100, area=20.0):
    """A held sleeve whose seat depth (mm, scalar or per-step) and bore margin are given."""
    tr = _trace(n=n, area=area)
    seat = np.asarray(seat_mm, dtype=float) * 1e-3
    tr.extra["seat_depth_m"] = np.full(n, float(seat)) if seat.ndim == 0 else seat
    margin = np.asarray(margin_mm, dtype=float) * 1e-3
    tr.extra["bore_margin_m"] = np.full(n, float(margin)) if margin.ndim == 0 else margin
    return tr


def test_sleeve_passes_when_seated():
    v = sleeve_on_shaft(_sleeve(20.0))
    assert v.success and v.metrics["seat_depth_m"] == pytest.approx(0.020)


def test_sleeve_reports_unfinished_when_shallow():
    assert sleeve_on_shaft(_sleeve(5.0)).failure is FailureMode.UNFINISHED


def test_sleeve_reports_overforce_when_the_bore_was_shut_and_not_seated():
    v = sleeve_on_shaft(_sleeve(0.0, margin_mm=-0.5))
    assert v.failure is FailureMode.OVERFORCE
    assert v.metrics["min_bore_margin_m"] == pytest.approx(-0.0005)


def test_sleeve_seated_after_a_squeeze_off_the_shaft_is_a_success():
    seat = np.full(100, 20.0)
    seat[:40] = 0.0
    margin = np.full(100, 2.0)
    margin[10:20] = -1.0
    assert sleeve_on_shaft(_sleeve(seat, margin_mm=margin)).success


def test_sleeve_forced_on_through_a_shut_bore_is_overforce_even_if_seated():
    seat = np.full(100, 20.0)
    seat[:40] = 0.0
    margin = np.full(100, 2.0)
    margin[45:60] = -0.5
    v = sleeve_on_shaft(_sleeve(seat, margin_mm=margin))
    assert v.failure is FailureMode.OVERFORCE
    assert v.metrics["min_bore_margin_on_shaft_m"] == pytest.approx(-0.0005)


def test_sleeve_shells_jostling_after_a_drop_are_not_overforce():
    tr = _sleeve(0.0)
    tr.pad_area = {p: np.where(np.arange(100) < 50, 20.0, 0.0) for p in tr.pad_area}
    margin = np.full(100, 2.0)
    margin[70:75] = -0.03
    tr.extra["bore_margin_m"] = margin * 1e-3
    assert sleeve_on_shaft(tr).failure is FailureMode.UNFINISHED


def test_sleeve_never_held_is_a_grasp_failure():
    assert sleeve_on_shaft(_sleeve(20.0, area=0.0)).failure is FailureMode.GRASP_FAILURE


def test_only_the_final_seat_depth_counts():
    seat = np.full(100, 20.0)
    seat[-5:] = 0.0
    assert not sleeve_on_shaft(_sleeve(seat)).success


def test_sleeve_says_so_when_no_seat_was_recorded():
    v = sleeve_on_shaft(_trace())
    assert not v.success and any("seat_depth_m" in n for n in v.notes)


def test_sleeve_seat_threshold_moves_the_verdict():
    tr = _sleeve(12.0)
    assert not sleeve_on_shaft(tr).success
    assert sleeve_on_shaft(tr, min_seat_m=0.010).success


def test_every_failing_verdict_names_a_mode():
    """The failure-distribution figure has no 'other' bucket by construction."""
    cases = [sleeve_on_shaft(_sleeve(5.0)),
             sleeve_on_shaft(_sleeve(20.0, area=0.0))]
    tr = _trace()
    pos = tr.object_pos["manipuland"].copy()
    pos[60:, 2] -= 0.10
    tr.object_pos["manipuland"] = pos
    cases.append(bimanual_lift(tr, target_xy=(0.3, 0.0)))
    for v in cases:
        assert not v.success
        assert v.failure is not FailureMode.NONE
        assert v.as_dict()["failure"] in {m.value for m in FailureMode}
