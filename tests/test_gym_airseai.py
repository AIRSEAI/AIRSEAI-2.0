# SPDX-License-Identifier: Apache-2.0
"""The lerobot-facing side of the environment, checked without Isaac.

`gym_airseai` is a client: it starts an Isaac process and talks to it over a
socket. These tests cover everything up to that process -- the registered
ids, the observation shape a variant promises, and the cursor that keeps
sibling environments off each other's states.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sim
from gym_airseai.env import AirseaiEnv
from gym_airseai.tasks import TASK_IDS, VARIANTS
from sim.tasks.registry import TASKS

REPO = Path(sim.__file__).resolve().parents[2]
CLIENT = REPO / "src" / "gym_airseai"


def test_the_client_knows_every_registered_task():
    assert set(TASK_IDS) == set(TASKS)


@pytest.mark.parametrize("path", sorted(CLIENT.glob("*.py")))
def test_the_client_never_imports_isaac(path):
    """It runs in the lerobot process, which is a Python that has no Isaac."""
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    for name in names:
        assert not name.startswith(("isaaclab", "isaacsim", "omni", "torch")), f"{path.name} imports {name}"
        if name.startswith("sim"):
            # Only the two simulation modules that are pure Python: the
            # socket transport, and the registry's list of task ids.
            assert name in ("sim.transport", "sim.tasks.registry"), f"{path.name} imports {name}"


def test_a_variant_is_checked_before_a_process_is_started():
    with pytest.raises(ValueError, match="variant"):
        AirseaiEnv("stack_cube", variant="not_a_variant")


def test_a_missing_state_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="stack_cube"):
        AirseaiEnv("stack_cube", init_states=True, init_states_path=str(tmp_path / "absent.json"))


def test_siblings_do_not_run_the_same_state(tmp_path, monkeypatch):
    """Two environments of one run take turns, rather than both starting at
    state 0 and scoring the same episode twice."""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    a = AirseaiEnv.__new__(AirseaiEnv)
    b = AirseaiEnv.__new__(AirseaiEnv)
    for e in (a, b):
        e.init_state_id, e._stride = 0, 2
        e._cursor = AirseaiEnv._cursor_path("stack_cube")
    if a._cursor is None:
        pytest.skip("no /proc on this platform; the local fallback is used")
    taken = [a._next_state_id(), b._next_state_id(), a._next_state_id(), b._next_state_id()]
    assert taken == [0, 1, 2, 3]


def test_without_a_cursor_one_environment_still_walks_the_set():
    e = AirseaiEnv.__new__(AirseaiEnv)
    e.init_state_id, e._stride, e._cursor = 0, 1, None
    assert [e._next_state_id() for _ in range(3)] == [0, 1, 2]


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_variant_promises_the_keys_its_dataset_has(variant):
    """Environment feature names and shapes must match the training dataset."""
    pytest.importorskip("lerobot")
    from lerobot_airseai.config import AirseaiEnvConfig

    cfg = AirseaiEnvConfig(task="stack_cube", variant=variant)
    names = cfg.image_names
    assert names[:3] == ("base", "right_wrist", "left_wrist")
    assert len(names) == (3 if variant == "vision" else 7)
    for name in names:
        assert cfg.features_map[name] == f"observation.images.{name}"
    assert cfg.features["agent_pos"].shape == (16,)
    assert cfg.features["action"].shape == (16,)
