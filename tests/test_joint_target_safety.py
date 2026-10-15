"""CPU checks for the opt-in VR joint target limiter; Isaac Sim is not started."""

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

from sim.joint_target_safety import bound_joint_target


def _limits(low=-math.pi, high=math.pi):
    return torch.tensor([[[low, high]] * 7], dtype=torch.float32)


def test_thirty_degree_ik_reversal_is_bounded_to_one_point_five_degrees_per_frame():
    previous = torch.zeros(1, 7)
    max_step = math.radians(30.0) * 0.05
    first, accepted = bound_joint_target(
        torch.full((1, 7), math.radians(30.0)), previous, _limits(), max_step
    )
    assert accepted
    assert torch.allclose(first, torch.full((1, 7), max_step), atol=1e-7)
    reversed_target, accepted = bound_joint_target(
        torch.full((1, 7), math.radians(-30.0)), first, _limits(), max_step
    )
    assert accepted
    assert torch.max(torch.abs(reversed_target - first)).item() <= max_step + 1e-7
    assert torch.allclose(reversed_target, torch.zeros(1, 7), atol=1e-7)


def test_soft_limit_bounds_command_without_touching_other_arm_or_fingers():
    complete_target = torch.zeros(1, 16)
    complete_target[:, 7] = 0.027  # right finger target, metres in the raw articulation
    complete_target[:, 15] = 0.021  # left finger target
    complete_target[:, 0] = math.radians(44.5)
    before = complete_target.clone()
    limited, accepted = bound_joint_target(
        torch.full((1, 7), math.radians(80.0)),
        complete_target[:, :7],
        _limits(-math.radians(45), math.radians(45)),
        math.radians(30.0) * 0.05,
    )
    assert accepted
    complete_target[:, :7] = limited
    assert complete_target[0, 0] <= math.radians(45) + 1e-7
    assert torch.equal(complete_target[:, 7:], before[:, 7:])


def test_nonfinite_ik_or_invalid_previous_command_is_rejected():
    previous = torch.zeros(1, 7)
    bad = torch.ones(1, 7)
    for value in (float("nan"), float("inf"), -float("inf")):
        bad[0, 3] = value
        result, accepted = bound_joint_target(bad, previous, _limits(), math.radians(1.5))
        assert not accepted and torch.equal(result, previous)
    outside = previous.clone()
    outside[0, 3] = math.radians(46)
    result, accepted = bound_joint_target(
        torch.zeros(1, 7), outside, _limits(-math.radians(45), math.radians(45)), math.radians(1.5)
    )
    assert not accepted and torch.equal(result, outside)


def test_slot_harness_uses_twenty_hertz_control_period(monkeypatch):
    fake_harness = types.ModuleType("sim.harness")
    fake_harness.CTRL_FPS = 20
    fake_harness.Harness = type("Harness", (), {"control_dt": 1 / 60})
    monkeypatch.setitem(sys.modules, "sim.harness", fake_harness)
    source = Path(__file__).resolve().parents[1] / "src" / "sim" / "manager_based" / "common" / "slot.py"
    spec = importlib.util.spec_from_file_location("slot_dt_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SlotHarness.control_dt == 0.05
