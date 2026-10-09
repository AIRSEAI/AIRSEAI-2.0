"""Placement criteria for a tray carrying three upright cups."""

import numpy as np

from sim.criteria.glassware_tray_transfer import CUP_NAMES, cup_standing_on_tray, verdict
from sim.criteria.trace import EpisodeTrace
from sim.dims.glassware_tray_transfer import CUP_OFFSETS, TRAY_THICKNESS

RIGHT = ("right_left_finger_tactile", "right_right_finger_tactile")
LEFT = ("left_left_finger_tactile", "left_right_finger_tactile")
GOAL_XY = (0.435, 0.0)


def make_trace(*, tipped=None, displaced=None, goal=True):
    t = np.arange(0, 2.05, 0.05)
    tray = np.tile([GOAL_XY[0] if goal else 0.265, 0.0, 0.203], (len(t), 1))
    contact = np.where(t < 0.4, 12.0, 0.0)
    pos = {"tray": tray}
    quat = {"tray": np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1))}
    for name, offset in zip(CUP_NAMES, CUP_OFFSETS, strict=True):
        pos[name] = tray + np.array([*offset, TRAY_THICKNESS])
        quat[name] = np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1))
    if displaced:
        pos[displaced] = pos[displaced] + np.array([0.18, 0.0, 0.0])
    if tipped:
        angle = np.deg2rad(80)
        quat[tipped][:] = [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
    return EpisodeTrace(t=t, pad_area={p: contact for p in RIGHT + LEFT}, object_pos=pos, object_quat=quat)


def judge(trace):
    return verdict(
        trace, goal_xy=GOAL_XY, goal_z=0.203, support_size=(0.26, 0.30), right_pads=RIGHT, left_pads=LEFT
    )


def test_three_standing_cups_at_destination():
    result = judge(make_trace())
    assert result.success
    assert result.metrics["standing_cups"] == 3
    assert not judge(make_trace(goal=False)).success


def test_each_cup_must_remain_on_tray_and_standing():
    for name in CUP_NAMES:
        assert not judge(make_trace(displaced=name)).success
        assert not judge(make_trace(tipped=name)).success
    standing, _ = cup_standing_on_tray([0.0, 0.0, TRAY_THICKNESS], [1, 0, 0, 0], [0, 0, 0], [1, 0, 0, 0])
    assert standing


def test_centered_tray_with_overhanging_corners_is_rejected():
    trace = make_trace()
    angle = np.deg2rad(45)
    trace.object_quat["tray"][:] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    result = judge(trace)
    assert result.metrics["standing_cups"] == 3
    assert result.metrics["tray_goal_xy_m"] == 0
    assert result.metrics["tray_footprint_inside"] == 0
    assert not result.success


def test_cup_tipping_after_a_valid_hold_is_not_success():
    trace = make_trace()
    trace.object_quat["cup_1"][trace.t >= 1.8] = [2**-0.5, 2**-0.5, 0.0, 0.0]
    result = judge(trace)
    assert result.metrics["longest_completion_hold_s"] >= 1.0
    assert result.metrics["completion_hold_s"] == 0.0
    assert not result.success


def test_gripper_release_and_full_hold_are_required():
    for release_time in (1.5, 3.0):
        trace = make_trace()
        trace.pad_area[RIGHT[0]][:] = np.where(trace.t < release_time, 12.0, 0.0)
        assert not judge(trace).success
