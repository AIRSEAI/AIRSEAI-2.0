"""Pin the release criterion and the detent's overshoot behaviour."""

import numpy as np

from sim.criteria.drawer_latch import verdict
from sim.criteria.trace import EpisodeTrace, FailureMode
from sim.mechanisms.drawer_latch import DrawerMechanism


def trace(extension, right_pads, release_at=35):
    n = len(extension)
    pads = {p: np.r_[np.full(release_at, 20.0), np.zeros(n - release_at)] for p in right_pads}
    return EpisodeTrace(
        t=np.arange(n) * 0.05,
        pad_area=pads,
        extra={
            "extension_m": np.asarray(extension, dtype=float),
            "hand_clearance_m": np.r_[np.zeros(release_at), np.full(n - release_at, 0.08)],
        },
    )


def test_drawer_stays_open_after_release(right_pads):
    x = np.r_[np.linspace(0, 0.08, 30), np.full(50, 0.08)]
    result = verdict(trace(x, right_pads), detent_m=0.08, pads=right_pads)
    assert result.success
    assert result.metrics["released_hold_s"] >= 1.0


def test_hand_holding_is_not_completion(right_pads):
    x = np.r_[np.linspace(0, 0.08, 30), np.full(50, 0.08)]
    result = verdict(trace(x, right_pads, release_at=80), detent_m=0.08, pads=right_pads)
    assert not result.success
    assert result.failure is FailureMode.UNFINISHED


def test_return_or_wrong_stop_fails(right_pads):
    x = np.r_[np.linspace(0, 0.08, 35), np.linspace(0.08, 0, 45)]
    assert not verdict(trace(x, right_pads), detent_m=0.08, pads=right_pads).success
    x = np.r_[np.linspace(0, 0.065, 35), np.full(45, 0.065)]
    assert not verdict(trace(x, right_pads), detent_m=0.08, pads=right_pads).success


def test_open_hand_near_handle_is_not_completion(right_pads):
    episode = trace(np.full(80, 0.08), right_pads)
    episode.extra["hand_clearance_m"][:] = 0.0
    assert not verdict(episode, detent_m=0.08, pads=right_pads).success


def test_recontact_after_release_is_not_completion(right_pads):
    episode = trace(np.full(80, 0.08), right_pads)
    episode.pad_area[right_pads[0]][-1] = 20.0
    assert not verdict(episode, detent_m=0.08, pads=right_pads).success


def test_late_return_after_release_is_not_completion(right_pads):
    x = np.r_[np.full(70, 0.08), np.linspace(0.08, 0.0, 10)]
    assert not verdict(trace(x, right_pads), detent_m=0.08, pads=right_pads).success


def test_detent_needs_stillness_and_resets_after_overshoot():
    mechanism = DrawerMechanism(0.08)
    for _ in range(10):
        mechanism.update(0.080, 0.02, 0.05)
    assert mechanism.mode == "search"
    for _ in range(4):
        mechanism.update(0.080, 0.0, 0.05)
    assert mechanism.mode == "latched"
    assert mechanism.update(0.080, 0.0, 0.05).target_m == -0.08
    assert mechanism.update(0.091, 0.01, 0.05).target_m == 0.0
    assert mechanism.mode == "overrun"
    mechanism.update(0.080, 0.0, 0.5)
    assert mechanism.mode == "overrun"
    mechanism.update(0.0, 0.0, 0.05)
    assert mechanism.mode == "search"
