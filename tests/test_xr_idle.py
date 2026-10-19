"""XR idle renders a finished scene without another simulation or data step."""

import math
from types import SimpleNamespace

import pytest

from sim.xr_idle import render_completed_scene, should_render_completed_scene


class FakeScene:
    def __init__(self):
        self.now = 0.0
        self.renders = 0
        self.physics_steps = 0
        self.recorded_frames = 0

    def render(self):
        self.renders += 1
        self.now += 0.25

    def step(self):
        self.physics_steps += 1

    def record(self):
        self.recorded_frames += 1


def _run(scene, seconds, *, running=lambda: True, stopping=lambda: False):
    messages = []
    reason = render_completed_scene(
        seconds,
        render=scene.render,
        is_running=running,
        stop_requested=stopping,
        log=messages.append,
        clock=lambda: scene.now,
    )
    return reason, messages


def test_deadline_renders_only_and_logs_exit():
    scene = FakeScene()
    reason, messages = _run(scene, 1.0)
    assert reason == "deadline"
    assert scene.renders == 4
    assert scene.physics_steps == scene.recorded_frames == 0
    assert messages == [
        "[AIRSEAI_XR_IDLE] entering seconds=1",
        "[AIRSEAI_XR_IDLE] exited reason=deadline",
    ]


@pytest.mark.parametrize(
    ("running", "stopping", "expected"),
    [
        (lambda scene: lambda: scene.renders < 2, lambda scene: lambda: False, "app_stopped"),
        (lambda scene: lambda: True, lambda scene: lambda: scene.renders >= 2, "stop_requested"),
    ],
)
def test_shutdown_and_stop_end_idle(running, stopping, expected):
    scene = FakeScene()
    reason, messages = _run(scene, 100.0, running=running(scene), stopping=stopping(scene))
    assert reason == expected
    assert scene.renders == 2
    assert scene.physics_steps == scene.recorded_frames == 0
    assert messages[-1] == f"[AIRSEAI_XR_IDLE] exited reason={expected}"


def test_render_failure_exits_without_propagating_after_final_report():
    scene = FakeScene()
    messages = []

    def fail_render():
        raise RuntimeError("renderer stopped")

    reason = render_completed_scene(
        10.0,
        render=fail_render,
        is_running=lambda: True,
        stop_requested=lambda: False,
        log=messages.append,
        clock=lambda: scene.now,
    )
    assert reason == "render_error:RuntimeError: renderer stopped"
    assert messages[-1].endswith("reason=render_error:RuntimeError: renderer stopped")


def test_keyboard_interrupt_during_render_preserves_completed_report_path():
    messages = []

    def interrupt_render():
        raise KeyboardInterrupt

    reason = render_completed_scene(
        10.0,
        render=interrupt_render,
        is_running=lambda: True,
        stop_requested=lambda: False,
        log=messages.append,
        clock=lambda: 0.0,
    )
    assert reason == "render_error:KeyboardInterrupt"
    assert messages[-1] == "[AIRSEAI_XR_IDLE] exited reason=render_error:KeyboardInterrupt"


def test_logging_failure_ends_idle_without_propagating():
    scene = FakeScene()

    def fail_log(_message):
        raise OSError("log unavailable")

    reason = render_completed_scene(
        10.0,
        render=scene.render,
        is_running=lambda: True,
        stop_requested=lambda: False,
        log=fail_log,
        clock=lambda: scene.now,
    )
    assert reason == "enter_log_error:OSError: log unavailable"
    assert scene.renders == 0


def test_disabled_and_invalid_duration():
    scene = FakeScene()
    assert _run(scene, 0.0) == ("disabled", [])
    assert scene.renders == 0
    for duration in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            _run(scene, duration)


def test_opt_in_guard_preserves_default_and_requires_completed_vr_trial():
    enabled = dict(
        status="complete",
        teleop=True,
        args=SimpleNamespace(teleop_device="vr", xr_keep_open_s=1800.0),
    )
    assert should_render_completed_scene(**enabled)
    for change in (
        {"args": SimpleNamespace(teleop_device="vr", xr_keep_open_s=0.0)},
        {"args": SimpleNamespace()},
        {"status": "interrupted"},
        {"teleop": False},
        {"args": SimpleNamespace(teleop_device="keyboard", xr_keep_open_s=1800.0)},
    ):
        assert not should_render_completed_scene(**(enabled | change))
