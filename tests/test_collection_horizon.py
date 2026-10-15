"""Exercise native collection deadlines without starting Isaac Sim.

The constructor and scheduler are real. Only simulator setup and unused
recording services are replaced; configuration uses the real JSON writer.
"""

import importlib.util
import json
import sys
import types

import pytest

from test_collection_native_vr_forward import _load_collection


_OMITTED = object()


class _CpuJointRow:
    def cpu(self):
        return self

    def tolist(self):
        return [0.0]


@pytest.fixture
def collection_module(monkeypatch):
    module = _load_collection(monkeypatch, [])
    spec = importlib.util.spec_from_file_location(
        "collection_horizon_real_io", module.__file__.replace("collection.py", "collection_io.py")
    )
    real_io = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, real_io)
    spec.loader.exec_module(real_io)
    monkeypatch.setattr(module, "write_json", real_io.write_json)
    monkeypatch.setattr(module, "EpisodeWriteQueue", lambda *_: None)
    monkeypatch.setattr(module, "SlotHarness", lambda *_: object())
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(default_joint_pos=[_CpuJointRow()]), joint_names=["joint_0"]
    )
    environment = types.SimpleNamespace(
        scene={"robot": robot}, device="cpu", reset=lambda **_: None
    )
    monkeypatch.setattr(module, "make_env", lambda *_a, **_kw: environment)
    monkeypatch.setattr(module, "version", lambda _: "cpu-test")
    return module


def _make_collector(module, tmp_path, *, override=_OMITTED, task_horizon=0.10):
    args = types.SimpleNamespace(
        num_envs=1,
        write_mode="sync",
        write_queue_size=1,
        episode_write_workers=1,
        seed=0,
        episodes=1,
        task="deadline_fixture",
        dataset="",
        out=str(tmp_path / "report"),
        init_states="",
        grid=(2, 2),
        device="cpu",
        robot_usd="",
        teleop_device="vr",
    )
    if override is not _OMITTED:
        args.horizon_s = override
    task = types.SimpleNamespace(
        name="deadline_fixture",
        horizon_s=task_horizon,
        make_scene_cfg=lambda _: types.SimpleNamespace(),
        signals=lambda *_: {},
    )
    return module.Collector(args, task, {}, object(), teleop=True)


def _attempt(module, tmp_path):
    attempt = module.Attempt(0, 0, {}, tmp_path)
    attempt.controller = types.SimpleNamespace(after_step=lambda _: None)
    return attempt


@pytest.mark.parametrize(
    ("override", "task_horizon", "frames_before", "deadline_s"),
    [
        (0.10, 0.50, 1, 0.10),
        (0.20, 0.10, 3, 0.20),
        (None, 0.10, 1, 0.10),
        (_OMITTED, 0.10, 1, 0.10),
    ],
    ids=["shorter-override", "longer-override", "none-uses-task", "missing-uses-task"],
)
def test_native_deadline_uses_requested_horizon_or_task_default(
    collection_module, tmp_path, override, task_horizon, frames_before, deadline_s
):
    collector = _make_collector(
        collection_module, tmp_path, override=override, task_horizon=task_horizon
    )
    attempt = _attempt(collection_module, tmp_path)
    for _ in range(frames_before):
        collector.after_step(attempt)
        assert attempt.terminal_reason is None
    collector.after_step(attempt)
    assert attempt.terminal_reason == "timeout"
    assert attempt.hist["t"][-1] == pytest.approx(deadline_s)


def test_deadline_preserves_an_already_finished_attempt(collection_module, tmp_path):
    collector = _make_collector(collection_module, tmp_path, override=0.05)
    attempt = _attempt(collection_module, tmp_path)
    attempt.frame = 10
    attempt.terminal_reason = "finished"
    collector.after_step(attempt)
    assert attempt.terminal_reason == "finished"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
@pytest.mark.parametrize("source", ["override", "task-default"])
def test_constructor_rejects_nonfinite_or_nonpositive_effective_horizon(
    collection_module, tmp_path, value, source
):
    options = {"override": value} if source == "override" else {"task_horizon": value}
    with pytest.raises(ValueError, match="horizon"):
        _make_collector(collection_module, tmp_path, **options)


@pytest.mark.parametrize(
    ("override", "expected"),
    [(0.20, 0.20), (None, 0.10), (_OMITTED, 0.10)],
    ids=["override", "none-uses-task", "missing-uses-task"],
)
def test_configuration_records_effective_horizon(
    collection_module, tmp_path, override, expected
):
    _make_collector(collection_module, tmp_path, override=override)
    configuration = json.loads((tmp_path / "report" / "configuration.json").read_text())
    assert configuration["effective_horizon_s"] == expected
