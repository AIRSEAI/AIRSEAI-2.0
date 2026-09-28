#!/usr/bin/env python3
"""Serve a task through a localhost socket in the AIRSEAI Python environment.

The client and server use the same Python 3.11 environment, including the
project's compatible LeRobot source. The process boundary isolates the Kit
lifecycle; it is not required by a Python version mismatch.

Operations expose metadata, reset, step, render and close. Observations use
uint8 HWC images and a 16-dimensional state; actions are joint targets.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

from sim.tasks.registry import TASKS

VARIANTS = ("vision", "tacff", "tacrgb")

parser = argparse.ArgumentParser(description="Serve an AIRSEAI task to lerobot.")
parser.add_argument("--task", required=True, choices=sorted(TASKS))
parser.add_argument("--variant", default="tacff", choices=VARIANTS, help="which observation set to serve")
parser.add_argument("--port", type=int, default=0, help="0 picks a free port and prints it")
parser.add_argument("--init_states", default="", help="frozen state file; without it every reset draws")
parser.add_argument("--seed", type=int, default=11, help="draw seed, used only without --init_states")
parser.add_argument("--horizon_s", type=float, default=None)
parser.add_argument("--robot_usd", default="")
parser.add_argument("--grid", type=int, nargs=2, default=(288, 384))
parser.add_argument(
    "--scene_cam", action="store_true", help="also spawn the third-person camera, for render()"
)
AppLauncher.add_app_launcher_args(parser)
_known, _rest = parser.parse_known_args()
app_launcher = AppLauncher(_known)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import socket

import numpy as np
import torch

from sim import init_state as init_state_io
from sim.envs.bimanual_tactile import TACTILE_SENSORS
from sim.harness import CTRL_FPS, STEPS_PER_CTRL, Harness
from sim.tasks.registry import load_task
from sim.tasks.spec import RetryReset
from sim.transport import recv_msg, send_msg

TASK, PINS = load_task(_known.task)
for kname, knob in TASK.knobs.items():
    parser.add_argument(f"--{kname}", type=knob.type, default=knob.default, help=knob.help)
args_cli = parser.parse_args()
KNOBS = {k: PINS.get(k, getattr(args_cli, k)) for k in TASK.knobs}


def observation_keys(variant: str, h) -> list[str]:
    """The keys this variant serves, in the order the datasets use."""
    keys = [
        "observation.state",
        "observation.images.base",
        "observation.images.right_wrist",
        "observation.images.left_wrist",
    ]
    if variant in ("tacff", "tacrgb"):
        keys += [f"observation.images.{variant}_{h.tactile_key_stem(p)}" for p in TACTILE_SENSORS]
    return keys


def as_uint8(key: str, value: torch.Tensor) -> np.ndarray:
    """A harness observation as it crosses the socket: state as float32,
    images as uint8 HWC with the batch dimension dropped."""
    v = value[0].detach().cpu()
    if key == "observation.state":
        return v.numpy().astype(np.float32)
    return (v.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()


def main() -> int:
    horizon_s = args_cli.horizon_s or TASK.horizon_s
    h = Harness(
        TASK.make_scene_cfg(KNOBS),
        device=args_cli.device,
        grid=tuple(args_cli.grid),
        robot_usd=args_cli.robot_usd,
        scene_cam=TASK.scene_cam if args_cli.scene_cam else None,
    )
    TASK.apply_collision(h)
    h.start()
    keys = observation_keys(args_cli.variant, h)
    frozen = (
        init_state_io.load(os.path.expanduser(args_cli.init_states))["states"]
        if args_cli.init_states
        else None
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args_cli.port))
    port = srv.getsockname()[1]
    srv.listen(1)
    # The client waits for this line to learn the port.
    print(f"ENVSERVER listening on 127.0.0.1:{port}", flush=True)

    state: dict = {"ctx": None, "hist": None, "step": 0, "nominal": False, "episode": 0, "state_id": None}
    n_ctrl = int(horizon_s * CTRL_FPS)

    def observe() -> dict:
        h.update_cams(include_scene_cam=args_cli.scene_cam)
        return {k: as_uint8(k, v) for k, v in h.build_obs(keys).items()}

    def do_reset(msg: dict) -> dict:
        ep = int(msg.get("episode", state["episode"]))
        for _attempt in range(3):
            try:
                if frozen is not None:
                    st = frozen[int(msg.get("state_id", ep)) % len(frozen)]
                    state["state_id"] = st["id"]
                    state["ctx"] = init_state_io.restore(h, TASK, st, KNOBS)
                else:
                    state["state_id"] = None
                    rng = np.random.default_rng(
                        [int(msg.get("seed", args_cli.seed)), TASK.reset_seed_index(ep)]
                    )
                    state["ctx"] = TASK.reset(h, rng, ep, KNOBS)
            except RetryReset as e:
                print(f"ENVSERVER reset retry ({e})", flush=True)
                continue
            break
        else:
            raise RuntimeError("reset failed three times")
        if not state["nominal"]:
            h.initial_render()
            state["nominal"] = True
        state["hist"] = {"t": []}
        state["step"] = 0
        state["episode"] = ep + 1
        return {"obs": observe(), "info": {"init_state_id": state["state_id"], "arm": state["ctx"]["side"]}}

    def do_step(msg: dict) -> dict:
        ctx, hist = state["ctx"], state["hist"]
        sig = TASK.signals(h, ctx)
        hist["t"].append(state["step"] / CTRL_FPS)
        for k, v in sig.items():
            hist.setdefault(k, []).append(v)
        a16 = np.asarray(msg["action"], dtype=np.float32).reshape(-1)
        hist.setdefault("state", []).append(h.state16()[0].cpu().numpy())
        hist.setdefault("tcp_pos", []).append(np.asarray(h.tcp_pos(), dtype=float))
        hist.setdefault("action", []).append(a16.copy())
        for _ in range(STEPS_PER_CTRL):
            h.apply_action16(torch.from_numpy(a16))
            h.step()
        state["step"] += 1

        terminated = bool(TASK.policy_should_terminate(h, ctx, hist, KNOBS))
        truncated = state["step"] >= n_ctrl
        info: dict = {"init_state_id": state["state_id"]}
        reward = 0.0
        if terminated or truncated:
            trace = TASK.build_trace(h, ctx, hist)
            jaw_col = 7 if ctx["side"] == "right" else 15
            trace.extra.setdefault("jaw_command_mm", np.asarray(hist["action"])[:, jaw_col])
            v = TASK.verdict(trace, ctx, KNOBS)
            reward = float(v.success)
            info.update(
                is_success=bool(v.success),
                failure=v.failure.value,
                metrics=v.metrics,
                **TASK.episode_extra(ctx, hist),
            )
        return {
            "obs": observe(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    while True:
        conn, _ = srv.accept()
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                op = msg.get("op")
                try:
                    if op == "meta":
                        send_msg(
                            conn,
                            {
                                "task": TASK.task_id(KNOBS),
                                "variant": args_cli.variant,
                                "instruction": TASK.instruction,
                                "keys": keys,
                                "action_dim": 16,
                                "fps": CTRL_FPS,
                                "max_episode_steps": n_ctrl,
                                "n_states": 0 if frozen is None else len(frozen),
                                "shapes": {k: list(v.shape) for k, v in observe().items()},
                            },
                        )
                    elif op == "reset":
                        send_msg(conn, do_reset(msg))
                    elif op == "step":
                        send_msg(conn, do_step(msg))
                    elif op == "render":
                        h.update_cams(include_scene_cam=args_cli.scene_cam)
                        send_msg(conn, {"frame": h.cam_rgb("scene" if args_cli.scene_cam else "base")})
                    elif op == "close":
                        send_msg(conn, {"ok": True})
                        return 0
                    else:
                        send_msg(conn, {"error": f"unknown op {op!r}"})
                except Exception as e:
                    # A failure here belongs to one call; the client decides
                    # whether to retry or give up, and sees the reason.
                    send_msg(conn, {"error": f"{type(e).__name__}: {e}"})
        finally:
            conn.close()


if __name__ == "__main__":
    rc = main()
    import sys

    sys.stdout.flush()
    os._exit(rc)  # skip Kit's own shutdown, which can hang
