# SPDX-License-Identifier: Apache-2.0
"""Freeze independent candidate starts and optionally select expert-solvable states."""

from __future__ import annotations

import copy
import hashlib
import signal
from collections import Counter
from pathlib import Path

import numpy as np

from sim import init_state
from sim.collection import Collector, state_hash, validate_states
from sim.collection_io import write_json
from sim.manager_based.common.states import prepare_restore


class StateCollector(Collector):
    scene_stream = "[seed, attempt_id, 0]"

    def __init__(self, args, task, knobs, app):
        self.target_count = args.count
        self.selected = []
        self.processed_candidates = set()
        self.rejections = []
        self.invalid = {}
        self.fingerprints = set()
        self.excluded = set()
        self.exclude_digests = {}
        for index, path in enumerate(args.exclude):
            document = init_state.load(path)
            if document["task"] != task.task_id(knobs):
                raise ValueError("Excluded initial states must belong to the same task")
            self.exclude_digests[f"source_{index:03d}"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for state in document["states"]:
                self.excluded.add(state_hash(state))
                if fingerprint := state.get("meta", {}).get("candidate_hash"):
                    self.excluded.add(fingerprint)
        super().__init__(args, task, knobs, app)
        root = Path(__file__).resolve().parents[2]
        sources = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "src").rglob("*.py"))
        }
        entry = root / "benchmark/run/make_init_states.py"
        sources[str(entry.relative_to(root))] = hashlib.sha256(entry.read_bytes()).hexdigest()
        write_json(self.out / "source_manifest.json", sources)

    def sample_state(self, attempt_id):
        entropy = [self.args.seed, attempt_id, 0]
        state = super().sample_state(attempt_id)
        state["id"] = attempt_id
        state.setdefault("meta", {}).update(
            candidate_id=attempt_id,
            scene_seed=entropy,
            expert_seed=[self.args.seed, attempt_id, 1],
            candidate_hash=state_hash(state),
        )
        return state

    def begin(self, attempt):
        if not attempt.frozen:
            try:
                actual = self.snapshot_start(attempt)
                validate_states(
                    dict(task=self.task.task_id(self.knobs), states=[actual]),
                    self.task,
                    self.knobs,
                    len(self.robot_start["joint_pos"]),
                )
                fingerprint = actual["meta"]["candidate_hash"]
                frozen_hash = state_hash(actual)
                if fingerprint in self.excluded or frozen_hash in self.excluded:
                    raise ValueError("State overlaps the excluded set")
                if fingerprint in self.fingerprints:
                    raise ValueError("Duplicate candidate state")
                self.fingerprints.add(fingerprint)
                actual["meta"]["frozen_hash"] = frozen_hash
                attempt.requested = actual
                attempt.frozen = True
                attempt.mode = "clearing"
                write_json(attempt.folder / "frozen_state.json", actual)
                prepare_restore(self.env, [actual], [attempt.env_id])
            except ValueError as error:
                self.invalid[attempt.attempt_id] = str(error)
                attempt.terminal_reason = "invalid_initial_state"
            return

        restored = self.snapshot_start(attempt)
        for section in ("robot", "objects"):
            for key, expected in attempt.requested[section].items():
                if not np.allclose(restored[section][key], expected, atol=1e-5, rtol=0):
                    self.invalid[attempt.attempt_id] = f"Restoration mismatch: {section}.{key}"
                    attempt.terminal_reason = "invalid_initial_state"
                    return
        if self.args.expert:
            super().begin(attempt)
        else:
            self.snapshots.append((attempt.attempt_id, copy.deepcopy(attempt.requested)))
            attempt.report["init_state_hash"] = state_hash(attempt.requested)
            write_json(attempt.folder / "initial_state.json", attempt.requested)
            attempt.mode = "running"
            attempt.terminal_reason = "initialized"

    def schedule(self):
        # Decide membership in candidate order, never in worker completion order.
        if any(self.active):
            return
        self.collect_writes(drain=True)
        self.select_candidates()
        if len(self.selected) >= self.target_count or min(self.next_ids) >= self.stop:
            self.count = len(self.reports)
            return
        super().schedule()

    def select_candidates(self):
        rows = {row["attempt_id"]: row for row in self.reports}
        for candidate, row in sorted(rows.items()):
            if candidate in self.processed_candidates:
                continue
            self.processed_candidates.add(candidate)
            failure = None
            if candidate in self.invalid:
                failure = self.invalid[candidate]
            elif row.get("write_status") != "complete":
                failure = "Candidate output did not complete"
            elif self.args.expert and not row.get("criterion_success"):
                failure = "Expert did not pass the task criterion"
            elif not self.args.expert and row["status"] != "initialized":
                failure = "Initialization was interrupted"
            if not failure and len(self.selected) >= self.target_count:
                failure = "Candidate surplus after reaching target"
            accepted = failure is None
            if accepted:
                self.selected.append(candidate)
            row["selected"] = accepted
            if failure:
                row["selection_reason"] = failure
                self.rejections.append(dict(candidate_id=candidate, reason=failure))
            write_json(self.out / "attempts" / f"{candidate:06d}" / "result.json", row)

    def state_output(self):
        self.select_candidates()
        snapshots = dict(self.snapshots)
        states = []
        for candidate in self.selected:
            state = copy.deepcopy(snapshots[candidate])
            state.setdefault("meta", {}).update(expert_checked=self.args.expert)
            states.append(state)
        return dict(
            task=self.task.task_id(self.knobs),
            states=states,
            rejected=self.rejections,
            generator=dict(
                seed=self.args.seed,
                num_envs=self.num_envs,
                requires_settle=False,
                expert_checked=self.args.expert,
                target_count=self.target_count,
                max_candidates=self.args.max_candidates,
                candidate_count=len(self.reports),
                excluded_files=self.exclude_digests,
                configuration_sha256=hashlib.sha256(
                    (self.out / "configuration.json").read_bytes()
                ).hexdigest(),
                source_manifest_sha256=hashlib.sha256(
                    (self.out / "source_manifest.json").read_bytes()
                ).hexdigest(),
                coordinate_frame="environment_local",
                velocity_rule="zero_on_restore",
                controller_rule="reset_on_restore",
            ),
        )


def generate_states(args, task, knobs, app):
    destination = Path(args.states_out).expanduser().resolve()
    partial = destination.with_suffix(".partial.json")
    if destination.exists() or partial.exists():
        raise FileExistsError("Choose a new initial-state output path")
    collector = StateCollector(args, task, knobs, app)
    stopping = False

    def interrupted(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    collector.stop_requested = lambda: stopping
    try:
        code = collector.run()
        complete = code == 0 and len(collector.selected) == args.count
        payload = collector.state_output()
        output = destination if complete else partial
        write_json(output, dict(version=init_state.SCHEMA_VERSION, **payload))
        write_json(
            collector.out / "manifest.json",
            dict(
                status="complete" if complete else "interrupted" if code else "insufficient_candidates",
                requested=args.count,
                selected=len(collector.selected),
                candidates=len(collector.reports),
                output=str(output),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                generator=payload["generator"],
            ),
        )
        write_json(
            collector.out / "rejection_summary.json",
            dict(Counter(row["reason"] for row in collector.rejections)),
        )
        return code if code else 0 if complete else 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
