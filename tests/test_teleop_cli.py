"""Validate CLI deadlines before the expensive simulator launch boundary."""

import runpy
import sys
import types
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmark/teleop/run_teleop.py"


class _LaunchBoundary(Exception):
    pass


def _run_cli(monkeypatch, horizon):
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")
            parser.add_argument("--xr", action="store_true")

        def __init__(self, _args):
            raise _LaunchBoundary

    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    app = types.ModuleType("isaaclab.app")
    app.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    args = [str(SCRIPT), "--task", "stack_cube", "--teleop_device", "vr"]
    if horizon is not None:
        args.append(f"--horizon_s={horizon}")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    runpy.run_path(str(SCRIPT), run_name="teleop_cli_under_test")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_cli_rejects_invalid_deadline_before_simulator_launch(monkeypatch, capsys, value):
    with pytest.raises(SystemExit) as error:
        _run_cli(monkeypatch, value)
    assert error.value.code == 2
    assert "horizon_s" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "0.05", "180"])
def test_cli_accepts_default_or_finite_positive_deadline(monkeypatch, value):
    with pytest.raises(_LaunchBoundary):
        _run_cli(monkeypatch, value)
