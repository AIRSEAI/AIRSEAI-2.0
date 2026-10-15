# SPDX-License-Identifier: Apache-2.0
"""Evaluate frozen starts with independent slots on a shared simulation clock."""

import hashlib
import json
import math
import os
import platform
import signal
import time
from pathlib import Path

import numpy as np
import torch

from sim import init_state
from sim.collection import CONTROL_DT, Collector, state_hash
from sim.collection_io import write_json
from sim.manager_based.common.states import snapshot_slot
from sim.vector_policy import VectorACT


class Evaluator(Collector):
    def __init__(self, args, task, knobs, app):
        self.started = args.startup_started
        self.evaluation_started = None
        self.inference_s = 0.0
        self.commands = {}
        action_steps = 100 if args.n_action_steps is None else args.n_action_steps
        self.policy = VectorACT(args.policy, args.device, args.num_envs, action_steps)
        self.checkpoint_hashes = {
            str(path.relative_to(self.policy.checkpoint)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.policy.checkpoint.rglob("*"))
            if path.is_file()
        }
        document = init_state.load(args.init_states)
        if document.get("generator", {}).get("coordinate_frame") != "environment_local":
            raise ValueError("Vector evaluation requires environment-local frozen states")
        if document.get("generator", {}).get("requires_settle", True):
            raise ValueError("Evaluate settled frozen states without resampling")
        if task.handover_phase is not None:
            raise ValueError("Vector evaluation requires a task with no scripted prologue")
        args.record_dataset = ""
        args.init_states_out = ""
        args.write_mode = "sync"
        args.write_queue_size = 1
        args.video = str(Path(args.video_dir) / "mosaic.mp4") if args.video_dir else ""
        super().__init__(args, task, knobs, app)
        self.report_path = self.out / "eval_report.json"
        self.init_digest = hashlib.sha256(Path(args.init_states).expanduser().read_bytes()).hexdigest()
        self.initialization_s = time.perf_counter() - self.started
        self.hardware = dict(
            host=platform.node(),
            device=str(args.device),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        self.versions = json.loads((self.out / "configuration.json").read_text())["versions"]
        source_root = Path(__file__).resolve().parents[2]
        self.source_hashes = {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source_root / "src/sim").rglob("*.py"))
        }
        entry = source_root / "benchmark/eval/evaluate.py"
        self.source_hashes[str(entry.relative_to(source_root))] = hashlib.sha256(
            entry.read_bytes()
        ).hexdigest()
        if str(args.device).startswith("cuda"):
            self.hardware["gpu"] = torch.cuda.get_device_name(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        write_json(self.out / "checkpoint_sha256.json", self.checkpoint_hashes)

    def begin(self, attempt):
        actual = self.snapshot_start(attempt)
        self.policy.reset(attempt.env_id)
        attempt.report["started_at"] = time.perf_counter()
        attempt.report["init_state_hash"] = state_hash(actual)
        write_json(attempt.folder / "initial_state.json", actual)
        self.snapshots.append((attempt.attempt_id, actual))
        mismatches = []
        for section in ("robot", "objects"):
            for key, expected in attempt.requested[section].items():
                if not np.allclose(actual[section][key], expected, atol=1e-5, rtol=0):
                    mismatches.append(f"{section}.{key}")
        if actual["task_vars"] != attempt.requested["task_vars"]:
            mismatches.append("task_vars")
        attempt.report["restore_mismatches"] = mismatches
        if mismatches:
            attempt.terminal_reason = "restore_failed"
            attempt.report["failure"] = "restore_failed"
            # Do not submit an action for a state whose restoration differs.
            # Collector.run() will record it as a terminal failed attempt.
            attempt.mode = "failed"
        else:
            attempt.mode = "running"

    def schedule(self):
        super().schedule()
        observations = {
            attempt.env_id: self.handles[attempt.env_id].build_obs(self.policy.input_keys)
            for attempt in self.active
            if attempt and attempt.mode == "running"
        }
        start = time.perf_counter()
        self.commands = self.policy.actions(observations)
        self.inference_s += time.perf_counter() - start

    def command(self, attempt):
        self.handles[attempt.env_id].apply_action16(self.commands[attempt.env_id])
        return True

    def after_step(self, attempt):
        attempt.frame += 1
        h = self.handles[attempt.env_id]
        sig = self.task.signals(h, attempt.ctx)
        attempt.hist["t"].append(attempt.frame * CONTROL_DT)
        for key, value in sig.items():
            attempt.hist.setdefault(key, []).append(value)
        if self.task.policy_should_terminate(h, attempt.ctx, attempt.hist, self.knobs):
            attempt.terminal_reason = "terminated"
        elif attempt.frame * CONTROL_DT >= (self.args.horizon_s or self.task.horizon_s):
            attempt.terminal_reason = "timeout"

    def finish(self, attempt):
        row = attempt.report
        row.update(status=attempt.terminal_reason, frames=attempt.frame, success=False, saved=False)
        if attempt.frame:
            trace = self.task.build_trace(self.handles[attempt.env_id], attempt.ctx, attempt.hist)
            verdict = self.task.verdict(trace, attempt.ctx, self.knobs)
            row.update(verdict=verdict.as_dict(), failure=verdict.failure.value)
            row["success_time_s"] = verdict.metrics.get("success_time_s")
            row["success"] = bool(verdict.success) and attempt.terminal_reason != "interrupted"
            row.update(self.task.episode_extra(attempt.ctx, attempt.hist))
        row["wall_s"] = time.perf_counter() - row.pop("started_at", time.perf_counter())
        row["simulation_s"] = attempt.frame * CONTROL_DT
        write_json(attempt.folder / "result.json", row)
        write_json(attempt.folder / "signals.json", attempt.hist)
        write_json(
            attempt.folder / "terminal_state.json",
            snapshot_slot(self.env, attempt.env_id, self.task.init_state_objects()),
        )
        self.reports.append(row)
        self.policy.reset(attempt.env_id)
        self.active[attempt.env_id] = None
        self.report("running")

    def video_state(self, slot, attempt):
        return dict(
            env_id=slot,
            attempt_id=attempt.requested["id"] if attempt else None,
            episode_step=attempt.frame if attempt else None,
            phase=attempt.mode if attempt else "idle",
            status=(attempt.terminal_reason or "policy") if attempt else "idle",
        )

    def report(self, status):
        rows = sorted(self.reports, key=lambda row: row["attempt_id"])
        successes = sum(bool(row.get("success")) for row in rows)
        n = self.count
        rate = successes / n
        z = 1.96
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        elapsed = time.perf_counter() - self.evaluation_started if self.evaluation_started else 0.0
        complete = status == "complete" and len(rows) == n
        report = dict(
            status=status if status != "complete" or complete else "incomplete",
            task=self.task.task_id(self.knobs),
            num_envs=self.num_envs,
            policy=str(self.policy.checkpoint),
            checkpoint_sha256=self.checkpoint_hashes,
            policy_execution=dict(
                chunk_size=self.policy.chunk_size,
                n_action_steps=self.policy.steps,
                temporal_ensemble_coeff=None,
            ),
            init_states_sha256=self.init_digest,
            requested_episodes=n,
            n_episodes=len(rows),
            n_success=successes,
            success_rate=rate if complete else None,
            confidence_interval_95=[center - radius, center + radius] if complete else None,
            episodes=rows,
            hardware=self.hardware,
            knobs=self.knobs,
            seed=self.args.seed,
            versions=self.versions,
            source_sha256=self.source_hashes,
            horizon_s=self.args.horizon_s or self.task.horizon_s,
            video_enabled=bool(self.video_path),
            control_hz=1 / CONTROL_DT,
            timing=dict(
                initialization_s=self.initialization_s,
                evaluation_s=elapsed,
                total_s=time.perf_counter() - self.started,
                inference_s=self.inference_s,
                episodes_per_minute=60 * len(rows) / elapsed if elapsed else 0,
            ),
        )
        if str(self.args.device).startswith("cuda"):
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(self.args.device)
        write_json(self.report_path, report)

    def run(self):
        self.evaluation_started = time.perf_counter()
        return super().run()


def run_evaluation(args, task, knobs, app):
    if not args.init_states or not args.policy or args.policy == "hold" or args.policy_server:
        raise ValueError("Vector evaluation requires --init_states and an in-process ACT checkpoint")
    evaluator = Evaluator(args, task, knobs, app)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    evaluator.stop_requested = lambda: stopping
    try:
        return evaluator.run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
