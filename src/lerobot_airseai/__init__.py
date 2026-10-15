# SPDX-License-Identifier: Apache-2.0
"""Register AIRSEAI environments with LeRobot.

    lerobot-eval --env.discover_packages_path=lerobot_airseai \
        --env.type=airseai --env.task=stack_cube --policy.path=<checkpoint>
"""

from __future__ import annotations

from lerobot_airseai.config import AirseaiEnvConfig

__all__ = ["AirseaiEnvConfig"]
