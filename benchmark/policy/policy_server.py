#!/usr/bin/env python3
"""Loads a lerobot checkpoint and serves `select_action` over a localhost
TCP socket.

    lerobot env ->  python benchmark/policy/policy_server.py --policy <ckpt_dir> --port 5555
    isaac env   ->  python benchmark/eval/evaluate.py --task <task> --policy-server 127.0.0.1:5555
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct


def recv_msg(conn) -> dict | None:
    hdr = conn.recv(8, socket.MSG_WAITALL)
    if len(hdr) < 8:
        return None
    (n,) = struct.unpack("<Q", hdr)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def send_msg(conn, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("<Q", len(data)) + data)


class ChunkQueue:
    """Timestep-indexed action queue, run synchronously: the client-side queue
    of lerobot's async robot client with no network or inference latency.
    A new chunk starts at the step about to execute; where it overlaps
    actions already queued, the two are merged with `aggregate`."""

    def __init__(self, actions_per_chunk: int, threshold: float, aggregate):
        self.actions_per_chunk = actions_per_chunk
        self.threshold = threshold
        self.aggregate = aggregate
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.actions: dict = {}

    def wants_chunk(self) -> bool:
        return len(self.actions) / self.actions_per_chunk <= self.threshold

    def merge(self, chunk: list) -> None:
        for i, new in enumerate(chunk):
            t = self.step + i
            self.actions[t] = self.aggregate(self.actions[t], new) if t in self.actions else new

    def pop(self):
        a = self.actions.pop(self.step)
        self.step += 1
        return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--actions_per_chunk", type=int, default=0,
                    help="if > 0, run lerobot async inference's action queue in lock "
                         "step: request a chunk of this many actions whenever the "
                         "queue is at or below --chunk_size_threshold of it, and "
                         "merge overlapping steps with --aggregate_fn_name. 0 uses "
                         "the checkpoint's own select_action")
    ap.add_argument("--chunk_size_threshold", type=float, default=0.5)
    ap.add_argument("--aggregate_fn_name", default="weighted_average")
    args = ap.parse_args()

    import torch
    from lerobot.async_inference.configs import get_aggregate_function
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import (
        get_policy_class,
        make_pre_post_processors,
    )

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    policy = get_policy_class(cfg.type).from_pretrained(args.policy)
    policy.to(args.device)
    policy.eval()
    # lerobot keeps normalisation outside the policy: observations go through
    # the preprocessor and actions through the postprocessor.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": args.device}})
    print(f"SERVER loaded {type(policy).__name__} (+pre/post processors) "
          f"from {args.policy}", flush=True)
    queue = ChunkQueue(args.actions_per_chunk, args.chunk_size_threshold,
                       get_aggregate_function(args.aggregate_fn_name)) \
        if args.actions_per_chunk > 0 else None
    if queue is not None:
        print(f"SERVER chunk queue: {args.actions_per_chunk} actions per chunk, "
              f"threshold {args.chunk_size_threshold}, {args.aggregate_fn_name}",
              flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"SERVER listening on 127.0.0.1:{args.port}", flush=True)

    while True:
        conn, _ = srv.accept()
        print("SERVER client connected", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                op = msg.get("op")
                if op == "ping":
                    send_msg(conn, {"ok": True})
                elif op == "meta":
                    send_msg(conn, {"input_keys": list(cfg.input_features.keys()),
                                    "policy_type": cfg.type})
                elif op == "reset":
                    policy.reset()
                    if queue is not None:
                        queue.reset()
                    send_msg(conn, {"ok": True})
                elif op == "act":
                    obs = {k: torch.as_tensor(v)
                           for k, v in msg["obs"].items()}
                    obs = preprocessor(obs)
                    with torch.inference_mode():
                        if queue is None:
                            a = postprocessor(policy.select_action(obs))
                        else:
                            if queue.wants_chunk():
                                chunk = policy.predict_action_chunk(obs)
                                queue.merge([postprocessor(chunk[:, i])
                                             for i in range(queue.actions_per_chunk)])
                            a = queue.pop()
                    send_msg(conn, {"action": a.detach().cpu().numpy()})
                elif op == "quit":
                    send_msg(conn, {"ok": True})
                    return 0
                else:
                    send_msg(conn, {"error": f"unknown op {op!r}"})
        finally:
            conn.close()
            print("SERVER client disconnected", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
