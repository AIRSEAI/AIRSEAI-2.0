"""Expand texture-sort starts into matching smooth/rough surface variants."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from sim.dims.texture_sort import OBJECT_SPECS


def expand_pairs(document, source_hash, seed):
    if document.get("task") != "texture_sort" or document.get("version") != 1:
        raise ValueError("Expected texture_sort initial states with schema version 1")
    if not document.get("states") or document.get("generator", {}).get("requires_settle", True):
        raise ValueError("Generate frozen base states with make_init_states.py first")
    states = []
    identities = set()
    for index, base in enumerate(document["states"]):
        identity = json.dumps(base["id"], sort_keys=True)
        if identity in identities:
            raise ValueError("Duplicate base state ID")
        identities.add(identity)
        active = base["task_vars"]["target_key"]
        inactive = next(key for key in OBJECT_SPECS if key != active)
        active_pose = base["objects"][OBJECT_SPECS[active].name]
        parked_pose = base["objects"][OBJECT_SPECS[inactive].name]
        for variant in ("smooth", "rough"):
            state = copy.deepcopy(base)
            state["id"] = f"{source_hash}:{base['id']}:{variant}"
            state["task_vars"]["target_key"] = variant
            for key, spec in OBJECT_SPECS.items():
                state["objects"][spec.name] = copy.deepcopy(active_pose if key == variant else parked_pose)
            state["meta"] = dict(
                base_state_id=base["id"],
                base_file_sha256=source_hash,
                pair_id=f"{source_hash}:{base['id']}",
                variant=variant,
                expert_seed=[seed, index, 1],
                expert_checked=False,
            )
            states.append(state)
    return dict(
        version=document["version"],
        task=document["task"],
        states=states,
        rejected=[],
        generator=dict(
            requires_settle=False,
            base_file_sha256=source_hash,
            coordinate_frame="environment_local",
            expert_checked=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    payload = args.src.read_bytes()
    document = expand_pairs(json.loads(payload), hashlib.sha256(payload).hexdigest(), args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
