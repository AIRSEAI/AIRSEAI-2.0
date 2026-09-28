"""Save and restore frozen robot, object and task states.

Robot states include measured joint positions and controller targets;
object states contain settled poses, restored with zero velocity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "SCHEMA_VERSION",
    "snapshot",
    "restore",
    "robot_state",
    "apply_robot",
    "object_pose",
    "apply_object",
    "save",
    "load",
    "digest",
]

SCHEMA_VERSION = 1


def snapshot(h, task, ctx: dict, state_id: int) -> dict:
    """The settled scene as one state, taken between `reset` and the first
    move of the expert."""
    return {
        "id": state_id,
        "robot": robot_state(h),
        "objects": {n: object_pose(h.scene[n]) for n in task.init_state_objects()},
        "task_vars": task.snapshot_init_state(h, ctx),
    }


def restore(h, task, state: dict, knobs: dict) -> dict:
    """Put a state back and return the episode ctx. The objects go first, so
    the task's own settling in `restore_init_state` acts on the real scene."""
    clearance = getattr(task, "restore_contact_clearance_m", 0.0)
    if clearance:
        # Advance once out of contact so previous friction anchors cannot
        # pull a teleported body toward its preceding episode's position.
        for index, name in enumerate(task.init_state_objects()):
            lifted = list(state["objects"][name])
            lifted[2] += clearance * (index + 1)
            apply_object(h, h.scene[name], lifted)
        h.step()
    for name in task.init_state_objects():
        apply_object(h, h.scene[name], state["objects"][name])
    apply_robot(h, state["robot"])
    return task.restore_init_state(h, state, knobs)


def robot_state(h) -> dict:
    """Every joint of the arm, plus the target it is being held at."""
    d = h.robot.data
    return {
        "joint_pos": [round(float(v), 9) for v in d.joint_pos[0].tolist()],
        "joint_pos_target": [round(float(v), 9) for v in d.joint_pos_target[0].tolist()],
    }


def apply_robot(h, robot: dict) -> None:
    """Put the arm back, at rest and already commanded where it stands."""
    dev = h.sim.device
    pos = torch.tensor([robot["joint_pos"]], dtype=torch.float32, device=dev)
    h.robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
    h.robot.set_joint_position_target(
        torch.tensor([robot["joint_pos_target"]], dtype=torch.float32, device=dev)
    )
    h.robot.write_data_to_sim()
    h.reset_ik()


def object_pose(obj) -> list[float]:
    """-> [x, y, z, qw, qx, qy, qz], measured after the settle."""
    p = obj.data.root_pos_w[0].detach().cpu().numpy()
    q = obj.data.root_quat_w[0].detach().cpu().numpy()
    return [round(float(v), 9) for v in (*p, *q)]


def apply_object(h, obj, pose) -> None:
    """Restore an object's pose with zero velocity."""
    h.place(obj, (pose[0], pose[1]), pose[2], quat=tuple(pose[3:7]))


def save(path, *, task: str, states: list[dict], rejected: list[dict], generator: dict) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": task,
                "version": SCHEMA_VERSION,
                "generator": generator,
                "states": states,
                "rejected": rejected,
            },
            indent=1,
            default=_plain,
        )
        + "\n"
    )
    return path


def load(path) -> dict:
    doc = json.loads(Path(path).expanduser().read_text())
    if doc.get("version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema version {doc.get('version')}, this build reads {SCHEMA_VERSION}")
    if not doc.get("states"):
        raise ValueError(f"{path}: no states")
    return doc


def digest(path) -> str:
    """Return the first 16 hexadecimal characters of the file's SHA-256."""
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()[:16]


def _plain(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
