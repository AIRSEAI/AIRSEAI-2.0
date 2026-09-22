"""texture_sort: the bin the puck landed in decides."""

from __future__ import annotations

from sim.criteria.texture_sort import verdict
from sim.criteria.trace import FailureMode


def test_passes_in_the_matching_bin(held_trace, right_pads):
    v = verdict(held_trace(), chosen_bin=1, correct_bin=1, pads=right_pads)
    assert v.success and v.metrics["chosen_bin"] == 1.0


def test_the_wrong_bin_is_unfinished(held_trace, right_pads):
    v = verdict(held_trace(), chosen_bin=0, correct_bin=1, pads=right_pads)
    assert v.failure is FailureMode.UNFINISHED


def test_neither_bin_is_unfinished(held_trace, right_pads):
    v = verdict(held_trace(), chosen_bin=-1, correct_bin=1, pads=right_pads)
    assert v.failure is FailureMode.UNFINISHED


def test_never_held_is_a_grasp_failure(held_trace, right_pads):
    v = verdict(held_trace(area=0.0), chosen_bin=1, correct_bin=1, pads=right_pads)
    assert v.failure is FailureMode.GRASP_FAILURE
