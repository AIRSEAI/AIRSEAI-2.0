"""Sustained stacking completion and interruption boundaries."""

import numpy as np
import pytest

from sim.criteria.stack_cube import verdict
from sim.criteria.trace import FailureMode
from sim.dims.stack_cube import CUBE_SIZE


@pytest.fixture
def cube_trace(held_trace):
    tr = held_trace(n=80)
    b = tr.object_pos.pop("manipuland")
    a = b.copy()
    a[:, 2] += CUBE_SIZE
    tr.object_pos.update(cube_a=a, cube_b=b)
    tr.extra["gripper_gap_m"] = np.full(80, CUBE_SIZE + 0.004)
    for area in tr.pad_area.values():
        area[20:] = 0
    return tr


def test_stack_succeeds_and_latches(cube_trace, right_pads):
    cube_trace.object_pos["cube_a"][60:, 0] += 1
    result = verdict(cube_trace, pads=right_pads)
    assert result.success
    assert result.metrics["success_time_s"] == pytest.approx(1.0)


@pytest.mark.parametrize("case", ["brief", "misaligned", "interrupted"])
def test_incomplete_stack(cube_trace, right_pads, case):
    if case == "brief":
        cube_trace.object_pos["cube_a"][19:, 0] += 1
    elif case == "interrupted":
        cube_trace.object_pos["cube_a"][::10, 0] += 1
    else:
        cube_trace.object_pos["cube_a"][:, 0] += 1
    assert not verdict(cube_trace, pads=right_pads).success


def test_no_grasp_fails(cube_trace, right_pads):
    for area in cube_trace.pad_area.values():
        area[:] = 0
    assert verdict(cube_trace, pads=right_pads).failure is FailureMode.GRASP_FAILURE


@pytest.mark.parametrize("gap", [None, 0.0, np.nan])
def test_gripper_state_does_not_gate_success(cube_trace, right_pads, gap):
    cube_trace.extra.clear()
    if gap is not None:
        cube_trace.extra["gripper_gap_m"] = np.full(len(cube_trace.t), gap)
    for area in cube_trace.pad_area.values():
        area[:] = 20
    assert verdict(cube_trace, pads=right_pads).success


def test_prefix_matches_full_trace(cube_trace, right_pads):
    from sim.criteria.trace import EpisodeTrace

    first = None
    for n in range(2, len(cube_trace.t) + 1):
        prefix = EpisodeTrace(
            cube_trace.t[:n],
            {k: v[:n] for k, v in cube_trace.pad_area.items()},
            {k: v[:n] for k, v in cube_trace.object_pos.items()},
            extra={k: v[:n] for k, v in cube_trace.extra.items()},
        )
        result = verdict(prefix, pads=right_pads)
        if result.success:
            first = result.metrics["success_time_s"]
            break
    assert first == verdict(cube_trace, pads=right_pads).metrics["success_time_s"]


def test_bad_clock_rejected(cube_trace, right_pads):
    cube_trace.t[2] = cube_trace.t[1]
    with pytest.raises(ValueError, match="timestamps"):
        verdict(cube_trace, pads=right_pads)


def test_policy_hook_keeps_episodes_independent():
    from sim.criteria.trace import Verdict
    from sim.tasks.spec import Task

    class CompletionTask(Task):
        completion_on_success = True

        def build_trace(self, h, ctx, hist):
            assert h == "handle"
            return ctx["success"]

        def verdict(self, trace, ctx, knobs):
            assert knobs == {"threshold": 1}
            return Verdict(True) if trace else Verdict(False, FailureMode.UNFINISHED)

    task = CompletionTask()
    hist = {"t": [0.0, 0.05]}
    for success in (True, False, True, False):
        assert task.policy_should_terminate("handle", {"success": success}, hist, {"threshold": 1}) is success
