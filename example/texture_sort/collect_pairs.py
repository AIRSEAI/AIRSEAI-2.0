"""Generate base states, collect surface pairs, and publish complete successful pairs."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from generate_pairs import expand_pairs

from lerobot_tactile_ext.merge import merge_recordings

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def run(script, *arguments, gpu):
    child = subprocess.Popen(
        [sys.executable, str(ROOT / script), *map(str, arguments)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
    )
    try:
        code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, child.args)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait()


def main():
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Collection interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--states-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--batch-pairs", type=int, default=32)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--write_mode", choices=("background", "sync"), default="background")
    parser.add_argument("--write_queue_size", type=int, default=2)
    args = parser.parse_args()
    if min(args.pairs, args.batch_pairs, args.num_envs, args.write_queue_size) < 1:
        parser.error("counts must be positive")
    if args.seed < 0 or args.gpu < 0 or args.max_candidates < args.pairs * 2 or args.max_candidates % 2:
        parser.error("seed/GPU must be nonnegative; candidate limit must cover whole pairs")
    args.dataset, args.states_out, args.out = (
        path.expanduser().resolve() for path in (args.dataset, args.states_out, args.out)
    )
    destinations = (args.dataset, args.states_out, args.out)
    if any(path.exists() for path in (*destinations, args.states_out.with_suffix(".partial.json"))):
        parser.error("Choose new output paths")
    if any(
        a == b or a in b.parents or b in a.parents
        for index, a in enumerate(destinations)
        for b in destinations[index + 1 :]
    ):
        parser.error("Output paths must not overlap")
    args.out.mkdir(parents=True)
    inputs, states, selections, rejected = [], [], [], []
    attempted_pairs = 0
    successful_episodes = 0
    batch = 0
    while len(states) < args.pairs * 2 and attempted_pairs < args.max_candidates // 2:
        count = min(
            args.batch_pairs, args.pairs - len(states) // 2, args.max_candidates // 2 - attempted_pairs
        )
        folder = args.out / f"batch_{batch:04d}"
        folder.mkdir()
        base = folder / "base_states.json"
        seed = args.seed + batch
        run(
            "benchmark/run/make_init_states.py",
            "--task",
            "texture_sort",
            "--no-expert",
            "--count",
            count,
            "--max-candidates",
            count * 4,
            "--seed",
            seed,
            "--num_envs",
            args.num_envs,
            "--out",
            base,
            "--headless",
            "--enable_cameras",
            gpu=args.gpu,
        )
        payload = base.read_bytes()
        paired = expand_pairs(json.loads(payload), hashlib.sha256(payload).hexdigest(), seed)
        pair_file = folder / "pair_states.json"
        write_json(pair_file, paired)
        report_dir = folder / "collection"
        run(
            "benchmark/run/collect_dataset.py",
            "--task",
            "texture_sort",
            "--init_states",
            pair_file,
            "--dataset",
            folder / "raw",
            "--out",
            report_dir,
            "--num_envs",
            args.num_envs,
            "--gpus",
            args.gpu,
            "--seed",
            seed,
            "--allow_empty",
            "--write_mode",
            args.write_mode,
            "--write_queue_size",
            args.write_queue_size,
            *(["--render"] if args.render else []),
            gpu=args.gpu,
        )
        report = json.loads((report_dir / "texture_sort_scripted.json").read_text())
        rows = {row["attempt_id"]: row for row in report["episodes"]}
        if report["status"] != "complete" or set(rows) != set(range(count * 2)):
            raise RuntimeError(f"Incomplete collection: {report_dir}")
        for index in range(count):
            members = [rows[index * 2 + offset] for offset in range(2)]
            successful_episodes += sum(bool(row.get("criterion_success")) for row in members)
            pair_id = paired["states"][index * 2]["meta"]["pair_id"]
            accepted = all(
                row.get("criterion_success") and row.get("saved") and row.get("write_status") == "complete"
                for row in members
            )
            selections.append(
                dict(
                    pair_id=pair_id,
                    accepted=accepted,
                    batch=batch,
                    attempts=[row["attempt_id"] for row in members],
                )
            )
            if not accepted:
                rejected.append(dict(pair_id=pair_id, reason="At least one member failed or was not saved"))
                continue
            for row in members:
                attempt = report_dir / "attempts" / f"{row['attempt_id']:06d}"
                state = json.loads((attempt / "initial_state.json").read_text())
                expected = paired["states"][row["attempt_id"]]
                if state["id"] != expected["id"] or state["meta"]["pair_id"] != pair_id:
                    raise ValueError("Collected initial state does not match its source pair")
                state["meta"]["expert_checked"] = True
                states.append(state)
                inputs.append(attempt / "raw")
        attempted_pairs += count
        write_json(folder / "selection.json", selections[-count:])
        batch += 1
    complete = len(states) == args.pairs * 2
    document = dict(
        version=1,
        task="texture_sort",
        states=states,
        rejected=rejected,
        generator=dict(
            requires_settle=False,
            expert_checked=True,
            seed=args.seed,
            coordinate_frame="environment_local",
            attempted_pairs=attempted_pairs,
            attempted_episodes=attempted_pairs * 2,
            successful_episodes=successful_episodes,
            expert_success_rate=(successful_episodes / (attempted_pairs * 2) if attempted_pairs else 0.0),
        ),
    )
    output = args.states_out if complete else args.states_out.with_suffix(".partial.json")
    if complete:
        merge_recordings(inputs, args.dataset)
    write_json(output, document)
    write_json(args.out / "selection.json", selections)
    write_json(
        args.out / "manifest.json",
        dict(
            status="complete" if complete else "insufficient_candidates",
            requested_pairs=args.pairs,
            accepted_pairs=len(states) // 2,
            attempted_pairs=attempted_pairs,
            attempted_episodes=attempted_pairs * 2,
            successful_episodes=successful_episodes,
            expert_success_rate=(successful_episodes / (attempted_pairs * 2) if attempted_pairs else 0.0),
            states=str(output),
            states_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            dataset=str(args.dataset) if complete else None,
            episode_states=[
                dict(episode_index=i, state_id=state["id"], pair_id=state["meta"]["pair_id"])
                for i, state in enumerate(states)
            ],
        ),
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
