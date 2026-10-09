"""Seating feedback must reject spikes, missing sensors, and lost grasps."""

import math

import pytest

from sim.tasks.mug_on_hook_feedback import ShearContact


def detector(scale=1.0):
    d = ShearContact()
    for value in (100, 101, 99, 100, 100, 101, 99):
        d.observe_free(value * scale)
    d.start()
    return d


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_contact_confirmation_is_relative_to_each_grasp(scale):
    d = detector(scale)
    assert not d.update(130 * scale, 0.05)
    assert not d.update(140 * scale, 0.05)
    assert d.update(130 * scale, 0.05)


def test_brief_impact_does_not_release_and_resets_confirmation():
    d = detector()
    assert not d.update(200, 0.05)
    assert not d.update(100, 0.05)
    assert not d.update(130, 0.05)
    assert not d.update(130, 0.05)
    assert d.update(130, 0.05)


@pytest.mark.parametrize("bad", [dict(valid=False), dict(held=False)])
def test_sensor_loss_or_lost_grasp_cannot_confirm_support(bad):
    d = detector()
    assert not d.update(200, 0.10)
    assert not d.update(200, 0.10, **bad)
    assert not d.update(200, 0.10)
    assert d.update(200, 0.05)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_invalid_force_cannot_confirm_support(value):
    d = detector()
    assert not d.update(value, 1.0)
    assert d.elapsed == 0


def test_missing_baseline_is_an_error_without_pose_fallback():
    d = ShearContact()
    with pytest.raises(RuntimeError):
        d.start()
    with pytest.raises(RuntimeError):
        d.update(500, 1.0)


def test_invalid_free_sample_requires_a_new_valid_baseline():
    d = ShearContact()
    for _ in range(7):
        d.observe_free(100)
    d.observe_free(100, valid=False)
    with pytest.raises(RuntimeError):
        d.start()


@pytest.mark.parametrize("dt", [0.01, 0.025, 0.05])
def test_confirmation_uses_time_not_frame_count(dt):
    d = detector()
    n = round(d.hold_s / dt)
    assert not any(d.update(150, dt) for _ in range(n - 1))
    assert d.update(150, dt)


def test_contact_oscillation_does_not_reset_persistent_elevation():
    d = detector()
    assert not d.update(158, 0.05)
    assert not d.update(111, 0.05)
    assert d.update(136, 0.05)


def test_small_elevation_alone_cannot_trigger_contact():
    d = detector()
    assert not any(d.update(115, 0.05) for _ in range(30))


def test_return_to_free_load_clears_hysteresis():
    d = detector()
    assert not d.update(160, 0.05)
    assert not d.update(109, 0.05)
    assert not any(d.update(115, 0.05) for _ in range(30))
