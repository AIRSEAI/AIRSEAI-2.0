"""The criteria package's layering, checked mechanically.

A task's criterion must be removable as one file: nothing shared may depend on
it, and it may not reach into another task or into Isaac.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

import sim.criteria
from sim.tasks.registry import TASKS

CRITERIA_DIR = Path(sim.criteria.__file__).parent
SHARED = ("completion", "detectors", "probes", "trace")
TASK_MODULES = sorted(
    p.stem for p in CRITERIA_DIR.glob("*.py") if p.stem not in SHARED and p.stem != "__init__"
)
FORBIDDEN_PREFIXES = ("isaaclab", "isaacsim", "omni", "torch", "sim.envs", "sim.tasks")


def _imported_modules(path: Path) -> list[str]:
    """Every module name the file imports, including inside functions."""
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


@pytest.mark.parametrize("task", TASK_MODULES)
def test_a_task_module_exposes_a_verdict(task):
    fn = importlib.import_module(f"sim.criteria.{task}").verdict
    params = list(inspect.signature(fn).parameters.values())
    assert params[0].name == "trace"
    assert all(p.kind is p.KEYWORD_ONLY for p in params[1:]), (
        "thresholds are keyword arguments, so a trace can be re-judged"
    )


@pytest.mark.parametrize("task", TASK_MODULES)
def test_a_task_module_reaches_no_further_than_its_own_dims(task):
    for name in _imported_modules(CRITERIA_DIR / f"{task}.py"):
        assert not name.startswith(FORBIDDEN_PREFIXES), f"{task} imports {name}"
        if name.startswith("sim.dims."):
            assert name == f"sim.dims.{task}", f"{task} imports {name}"
        if name.startswith("sim.criteria."):
            assert name.rsplit(".", 1)[1] in SHARED, f"{task} imports {name}"


@pytest.mark.parametrize("module", ("__init__",) + SHARED)
def test_the_shared_modules_name_no_task(module):
    for name in _imported_modules(CRITERIA_DIR / f"{module}.py"):
        assert not name.startswith("sim.dims."), f"{module} imports {name}"
        assert name.rsplit(".", 1)[-1] not in TASK_MODULES, f"{module} imports {name}"


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_every_registered_task_has_a_criterion(task_id):
    module = TASKS[task_id].module
    assert module.rsplit(".", 1)[1] in TASK_MODULES


@pytest.mark.parametrize("task", TASK_MODULES)
def test_thresholds_are_geometric(task):
    """Criteria judge geometry, not force: TacSL magnitudes are uncalibrated."""
    fn = importlib.import_module(f"sim.criteria.{task}").verdict
    for name in inspect.signature(fn).parameters:
        assert "f_max" not in name and "force" not in name, f"{task}.{name}"
