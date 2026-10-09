"""Destination and escape checks for loose_parts_box_transfer."""

import numpy as np

from sim.criteria.loose_parts_box_transfer import part_escaped, verdict
from sim.criteria.trace import EpisodeTrace, FailureMode


def trace_at_goal():
    t = np.arange(80) / 20
    pos = np.tile([0.24, -0.13, 0.20], (len(t), 1))
    pos[30:] = [0.45, 0.13, 0.30]
    return EpisodeTrace(t=t, object_pos={"parts_box": pos}, extra={"escaped": np.zeros(len(t))})


def test_arrival_without_drop_succeeds():
    assert verdict(trace_at_goal(), goal_xy=(0.45, 0.13), goal_surface_z=0.30).success


def test_any_earlier_drop_fails():
    trace = trace_at_goal()
    trace.extra["escaped"][20] = 1
    result = verdict(trace, goal_xy=(0.45, 0.13), goal_surface_z=0.30)
    assert result.failure is FailureMode.SLIP


def test_xy_offset_is_diagnostic_only():
    trace = trace_at_goal()
    trace.object_pos["parts_box"][30:, :2] += [0.03, -0.04]
    result = verdict(trace, goal_xy=(0.45, 0.13), goal_surface_z=0.30)
    assert result.success
    assert np.isclose(result.metrics["goal_error_x_m"], 0.03)
    assert np.isclose(result.metrics["goal_error_y_m"], 0.04)


def test_height_hold_does_not_require_rest():
    trace = trace_at_goal()
    trace.object_pos["parts_box"][30:, 0] += np.linspace(0.0, 0.04, 50)
    assert verdict(trace, goal_xy=(0.45, 0.13), goal_surface_z=0.30).success


def test_part_escape_uses_box_frame():
    q = np.array([2**-0.5, 0, 0, 2**-0.5])
    assert part_escaped([0, 0, 0], q, [0, 0.12, 0.01], 0.013)
    assert not part_escaped([0, 0, 0], q, [0, 0.08, 0.01], 0.013)


def test_long_box_escape_boundary():
    q = [1, 0, 0, 0]
    assert not part_escaped([0, 0, 0], q, [0.13, 0, 0.02], 0.01, box_length=0.30)
    assert part_escaped([0, 0, 0], q, [0.17, 0, 0.02], 0.01, box_length=0.30)


def test_table_below_shelf_is_not_success():
    trace = trace_at_goal()
    trace.object_pos["parts_box"][30:, 2] = 0.20
    assert not verdict(trace, goal_xy=(0.45, 0.13), goal_surface_z=0.30).success


def test_leaving_shelf_after_dwell_is_not_success():
    trace = trace_at_goal()
    trace.object_pos["parts_box"][-5:, 2] = 0.20
    assert not verdict(trace, goal_xy=(0.45, 0.13), goal_surface_z=0.30).success


def test_height_tolerance_on_both_sides():
    for error, success in [(-0.0059, True), (0.0059, True), (-0.0061, False), (0.0061, False)]:
        trace = trace_at_goal()
        trace.object_pos["parts_box"][30:, 2] += error
        assert verdict(trace, goal_xy=(0.45, 0.13), goal_surface_z=0.30).success == success
