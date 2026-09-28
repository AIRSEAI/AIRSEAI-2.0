"""Register AIRSEAI tasks as Gym environments backed by an Isaac Sim subprocess.

import gym_airseai
env = gym.make("gym_airseai/stack_cube", variant="tacff")
"""

from __future__ import annotations

from gymnasium.envs.registration import register

from gym_airseai.env import AirseaiEnv
from gym_airseai.tasks import TASK_IDS

__all__ = ["AirseaiEnv", "TASK_IDS"]

for _task in TASK_IDS:
    register(id=f"gym_airseai/{_task}", entry_point="gym_airseai.env:AirseaiEnv", kwargs={"task": _task})
