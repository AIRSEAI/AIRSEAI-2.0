# SPDX-License-Identifier: Apache-2.0
"""Gym client for an Isaac Sim subprocess using the active Python environment.

Observations contain uint8 camera images and a 16-dimensional robot state.
Actions are joint targets for the right arm, right jaw, left arm and left jaw.
"""

from __future__ import annotations

import fcntl
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from gym_airseai.tasks import VARIANTS
from sim.transport import recv_msg, send_msg

__all__ = ["AirseaiEnv"]

STARTUP_TIMEOUT_S = 300.0  # Isaac start plus SDF cooking, on a cold cache
CALL_TIMEOUT_S = 600.0


def _default_repo_root() -> Path:
    """The checkout this package lives in: src/gym_airseai/env.py -> ../../"""
    return Path(__file__).resolve().parents[2]


def _default_sim_python() -> str:
    """The interpreter that has Isaac Sim. Overridden with AIRSEAI_SIM_PYTHON."""
    env = os.environ.get("AIRSEAI_SIM_PYTHON")
    if env:
        return env
    return sys.executable


class AirseaiEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str,
        variant: str = "tacff",
        *,
        init_states: bool = True,
        init_states_path: str | None = None,
        episode_index: int = 0,
        n_envs: int = 1,
        seed: int = 11,
        render_scene_cam: bool = False,
        horizon_s: float | None = None,
        repo_root: str | None = None,
        sim_python: str | None = None,
        device: str = "cuda",
        knobs: dict | None = None,
        startup_timeout_s: float = STARTUP_TIMEOUT_S,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.task, self.variant = task, variant
        self.render_mode = "rgb_array"
        self._root = Path(repo_root or os.environ.get("AIRSEAI_REPO_ROOT") or _default_repo_root())
        self._python = sim_python or _default_sim_python()
        self._seed = seed
        # Sibling environments share a cursor to consume each initial state once.
        self.init_state_id = episode_index
        self._stride = max(1, n_envs)
        self._cursor = self._cursor_path(task)
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None

        states = ""
        if init_states:
            states = str(init_states_path or self._root / "meta" / "init_states" / f"{task}.json")
            if not Path(states).is_file():
                raise FileNotFoundError(
                    f"no evaluation states for {task}: {states}. Generate them with "
                    f"benchmark/run/make_init_states.py, or pass init_states=False "
                    f"to draw a fresh start each episode."
                )
        self._start(states, render_scene_cam, horizon_s, device, knobs or {}, startup_timeout_s)

        meta = self._call({"op": "meta"})
        self.instruction = meta["instruction"]
        self._keys = meta["keys"]
        self._max_episode_steps = meta["max_episode_steps"]
        self.n_states = meta["n_states"]
        self.metadata["render_fps"] = meta["fps"]
        shapes = meta["shapes"]
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Dict(
                    {
                        self._short(k): gym.spaces.Box(0, 255, tuple(shapes[k]), np.uint8)
                        for k in self._keys
                        if k != "observation.state"
                    }
                ),
                "agent_pos": gym.spaces.Box(-np.inf, np.inf, tuple(shapes["observation.state"]), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (meta["action_dim"],), np.float32)

    # ---- gym ---------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        i = self._next_state_id()
        msg = {"op": "reset", "seed": self._seed if seed is None else seed}
        msg["state_id" if self.n_states else "episode"] = i
        out = self._call(msg)
        return self._observation(out["obs"]), dict(out["info"], task=self.instruction)

    def step(self, action):
        out = self._call({"op": "step", "action": np.asarray(action, np.float32).reshape(-1)})
        info = dict(out["info"], task=self.instruction)
        return (
            self._observation(out["obs"]),
            float(out["reward"]),
            bool(out["terminated"]),
            bool(out["truncated"]),
            info,
        )

    def render(self):
        return self._call({"op": "render"})["frame"]

    def close(self):
        if self._sock is not None:
            try:
                self._call({"op": "close"}, timeout=30.0)
            except Exception:
                pass
            try:
                self._sock.close()
            finally:
                self._sock = None
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            # Stop a simulation process that did not close within the timeout.
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.send_signal(signal.SIGKILL)
                p.wait(timeout=20)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- which state comes next -------------------------------------------

    @staticmethod
    def _cursor_path(task: str) -> Path | None:
        """One cursor per evaluation run, named after the process that owns
        the environments and when it started, so a later run never inherits
        an earlier one's position."""
        try:
            ppid = os.getppid()
            with open(f"/proc/{ppid}/stat") as f:
                started = f.read().rsplit(")", 1)[1].split()[19]
            return Path(tempfile.gettempdir()) / f"airseai-{task}-{ppid}-{started}"
        except Exception:
            return None  # fall back to counting locally

    def _next_state_id(self) -> int:
        """The next state, claimed under a lock so two sibling environments
        never run the same one."""
        if self._cursor is None:
            i, self.init_state_id = self.init_state_id, self.init_state_id + self._stride
            return i
        try:
            with open(self._cursor, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    raw = f.read().strip()
                    i = int(raw) if raw else self.init_state_id
                    f.seek(0)
                    f.truncate()
                    f.write(str(i + 1))
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            return i
        except Exception:
            self._cursor = None
            return self._next_state_id()

    # ---- the simulation process -------------------------------------------

    def _start(self, states, scene_cam, horizon_s, device, knobs, timeout_s) -> None:
        server = self._root / "benchmark" / "run" / "env_server.py"
        if not server.is_file():
            raise FileNotFoundError(f"no env server at {server}; set AIRSEAI_REPO_ROOT")
        cmd = [
            self._python,
            str(server),
            "--task",
            self.task,
            "--variant",
            self.variant,
            "--port",
            "0",
            "--headless",
            "--enable_cameras",
            "--device",
            device,
            "--seed",
            str(self._seed),
        ]
        if states:
            cmd += ["--init_states", states]
        if scene_cam:
            cmd += ["--scene_cam"]
        if horizon_s is not None:
            cmd += ["--horizon_s", str(horizon_s)]
        for k, v in knobs.items():
            cmd += [f"--{k}", str(v)]
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join([str(self._root / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(
                os.pathsep
            ),
            OMNI_KIT_ACCEPT_EULA="YES",
        )
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        port = self._await_port(timeout_s)
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=CALL_TIMEOUT_S)

    def _await_port(self, timeout_s: float) -> int:
        """Read the server's log until it says which port it took."""
        deadline = time.monotonic() + timeout_s
        tail: list[str] = []
        assert self._proc is not None and self._proc.stdout is not None
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                break
            tail.append(line.rstrip())
            del tail[:-30]
            if "ENVSERVER listening on" in line:
                return int(line.rsplit(":", 1)[1])
        self.close()
        raise RuntimeError("the Isaac process did not come up:\n  " + "\n  ".join(tail[-20:]))

    def _call(self, msg: dict, timeout: float | None = None) -> dict:
        if self._sock is None:
            raise RuntimeError("environment is closed")
        if timeout is not None:
            self._sock.settimeout(timeout)
        send_msg(self._sock, msg)
        out = recv_msg(self._sock)
        if timeout is not None:
            self._sock.settimeout(CALL_TIMEOUT_S)
        if out is None:
            raise RuntimeError("the Isaac process closed the connection")
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    # ---- observations ------------------------------------------------------

    @staticmethod
    def _short(key: str) -> str:
        """'observation.images.left_wrist' -> 'left_wrist'."""
        return key.rsplit(".", 1)[-1]

    def _observation(self, obs: dict) -> dict:
        return {
            "pixels": {self._short(k): v for k, v in obs.items() if k != "observation.state"},
            "agent_pos": obs["observation.state"],
        }
