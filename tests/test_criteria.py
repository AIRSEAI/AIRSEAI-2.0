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

def _handover_trace(n=100, dt=0.05, release_step=60, catch_step=50, seat_step=80,
                    gap_mm=4.0):
    """Left pads hold, right pads take over, the sleeve ends seated and let go."""
    tr = _trace(n=n, dt=dt)
    giver = np.zeros(n)
    giver[:release_step] = 20.0        # left pads hold, then let go
    receiver = np.zeros(n)
    receiver[catch_step:seat_step] = 20.0   # right pads take over, then set it down
    for p in (PAD_L1, PAD_L2):
        tr.pad_area[p] = giver
    for p in (PAD_R1, PAD_R2):
        tr.pad_area[p] = receiver
    tr.object_pos["sleeve"] = tr.object_pos.pop("manipuland")
    tr.extra["gap_m"] = np.full(n, gap_mm * 1e-3)
    over = np.zeros(n)
    over[seat_step - 10:] = 1.0        # above the target for the last stretch
    tr.extra["over_target"] = over
    seat = np.zeros(n)
    seat[seat_step:] = 0.020           # down over the key once set down
    tr.extra["seat_depth_m"] = seat
    return tr


def test_handover_accepts_a_seated_sleeve():
    v = handover(_handover_trace())
    assert v.success, v.metrics


def test_handover_fall_outranks_everything():
    tr = _handover_trace()
    pos = tr.object_pos["sleeve"].copy()
    pos[70:, 2] -= 0.06                # onto the table
    tr.object_pos["sleeve"] = pos
    assert handover(tr).failure is FailureMode.COORDINATION


def test_handover_needs_the_giver_to_have_held():
    tr = _handover_trace()
    for p in (PAD_L1, PAD_L2):
        tr.pad_area[p] = np.zeros(len(tr.t))
    assert handover(tr).failure is FailureMode.GRASP_FAILURE


def test_handover_window_too_open_is_coordination():
    tr = _handover_trace(gap_mm=7.0)   # wider than the 42 mm bore admits
    tr.extra["seat_depth_m"] = np.zeros(len(tr.t))     # resting on the rim
    v = handover(tr)
    assert v.failure is FailureMode.COORDINATION
    assert v.metrics["gap_at_target_m"] > v.metrics["gap_hi_m"]


def test_handover_window_too_closed_is_overforce():
    tr = _handover_trace(gap_mm=1.0)   # squeezed below the key
    tr.extra["seat_depth_m"] = np.zeros(len(tr.t))     # resting on the key
    v = handover(tr)
    assert v.failure is FailureMode.OVERFORCE
    assert v.metrics["gap_at_target_m"] < v.metrics["gap_lo_m"]


def test_handover_seated_below_the_window_is_still_overforce():
    tr = _handover_trace(gap_mm=2.0)   # wedged open over the key and seated anyway
    v = handover(tr)
    assert v.failure is FailureMode.OVERFORCE
    assert v.metrics["seat_depth_m"] >= v.metrics["min_seat_m"]


def test_handover_too_open_and_still_held_is_coordination():
    tr = _handover_trace(gap_mm=7.0, seat_step=100)   # never went in, never let go
    tr.extra["over_target"][-10:] = 1.0
    assert handover(tr).failure is FailureMode.COORDINATION


def test_handover_is_unfinished_until_over_the_target_and_let_go():
    tr = _handover_trace()
    tr.extra["over_target"] = np.zeros(len(tr.t))
    assert handover(tr).failure is FailureMode.UNFINISHED
    tr = _handover_trace(seat_step=100)                    # seated but still gripped
    tr.extra["seat_depth_m"] = np.full(len(tr.t), 0.020)
    tr.extra["over_target"][-10:] = 1.0
    v = handover(tr)
    assert v.failure is FailureMode.UNFINISHED and v.metrics["released"] == 0.0


def test_handover_reads_the_gap_while_the_receiver_holds():
    tr = _handover_trace(seat_step=85, gap_mm=4.0)
    tr.extra["seat_depth_m"] = np.zeros(len(tr.t))     # never went in
    gap = tr.extra["gap_m"].copy()
    gap[85:] = 0.010                   # sprang open once let go over the target
    tr.extra["gap_m"] = gap
    v = handover(tr)
    assert v.metrics["gap_at_target_m"] == pytest.approx(0.004)
    assert v.failure is FailureMode.UNFINISHED


def test_handover_judges_the_final_seat():
    tr = _handover_trace()
    seat = tr.extra["seat_depth_m"].copy()
    seat[-1] = 0.0                     # pulled back out at the very end
    tr.extra["seat_depth_m"] = seat
    assert handover(tr).failure is FailureMode.UNFINISHED


def test_handover_reports_but_does_not_judge_the_release_timing():
    tr = _handover_trace(catch_step=20, release_step=95, seat_step=96)   # both held 3.75 s
    v = handover(tr)
    assert v.success and v.metrics["gap_s"] == pytest.approx(3.70)


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
        handover: {"bore_m": 0.042, "min_seat_m": 0.010, "drop_height_m": 0.05},
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
