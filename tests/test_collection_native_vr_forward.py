"""Exercise the native Collector device call without starting Isaac Sim."""

import importlib.util
import sys
import types
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src" / "sim" / "collection.py"


def _load_collection(monkeypatch, calls):
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        if name in {"sim", "sim.envs", "sim.manager_based", "sim.manager_based.common", "lerobot_tactile_ext"}:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("lerobot_tactile_ext")
    stub("lerobot_tactile_ext.merge", merge_recordings=lambda *_: None)
    sim = stub("sim")
    sim.init_state = stub("sim.init_state")
    stub("sim.collection_io", EpisodeWrite=object, EpisodeWriteQueue=object, write_json=lambda *_: None)
    stub("sim.envs")
    stub("sim.envs.bimanual_tactile", TACTILE_SENSORS=())
    stub("sim.harness", CTRL_FPS=20)
    stub("sim.manager_based", make_env=lambda *_a, **_kw: None)
    stub("sim.manager_based.common")
    stub("sim.manager_based.common.mdp", hold_action=lambda *_: None)
    stub("sim.manager_based.common.slot", SlotHarness=object)
    stub(
        "sim.manager_based.common.states",
        prepare_restore=lambda *_: None,
        restore_states=lambda *_: None,
        snapshot_slot=lambda *_: None,
    )
    stub("sim.recording", EpisodeRecorder=object)

    class Device:
        def begin_episode(self, ctx):
            self.ctx = ctx

        def help_text(self):
            return "fake input device"

    def make_device(name, task, harness, **kwargs):
        calls.append((name, task, harness, kwargs))
        return Device()

    stub("sim.teleop", make_device=make_device)
    stub("sim.video", Mp4Writer=object, compose_grid_frame=lambda *_a, **_kw: None)
    stub(
        "sim.xr_idle",
        render_completed_scene=lambda *_a, **_kw: None,
        should_render_completed_scene=lambda *_a, **_kw: False,
    )
    spec = importlib.util.spec_from_file_location("collection_native_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "state_hash", lambda _: "sha")
    return module


def test_native_collector_forwards_position_gain_only_for_vr(monkeypatch):
    calls = []
    module = _load_collection(monkeypatch, calls)

    def begin(device_name, gain):
        args = types.SimpleNamespace(
            seed=0, teleop_device=device_name, pos_speed=0.2, rot_speed=0.9,
            jaw_speed=0.02, vr_joint_speed_deg_s=60.0,
        )
        if gain is not None:
            args.vr_position_gain = gain
        task, harness = object(), object()
        collector = types.SimpleNamespace(
            args=args, handles=[harness], task=task, knobs={}, snapshots=[],
            dataset=None, teleop=True, controller=None,
        )
        actual = {"id": {"episode": 0}}

        def snapshot_start(attempt):
            attempt.ctx = {"side": "right"}
            return actual

        collector.snapshot_start = snapshot_start
        attempt = types.SimpleNamespace(
            requested={"meta": {}}, attempt_id=0, env_id=0,
            report={}, folder=Path("."), ctx=None,
        )
        module.Collector.begin(collector, attempt)
        assert attempt.mode == "running"
        assert attempt.controller.ctx == {"side": "right"}
        return task, harness

    vr_task, vr_harness = begin("vr", 2.0)
    assert calls[-1][:3] == ("vr", vr_task, vr_harness)
    assert calls[-1][3]["vr_position_gain"] == 2.0
    assert calls[-1][3]["vr_joint_speed_deg_s"] == 60.0

    begin("vr", None)
    assert calls[-1][3]["vr_position_gain"] == 1.0

    begin("keyboard", 2.0)
    assert "vr_position_gain" not in calls[-1][3]
    assert "vr_joint_speed_deg_s" not in calls[-1][3]
