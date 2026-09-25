#!/usr/bin/env python3
"""Generate paired texture layouts without running an expert or recording data."""

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


def generate(args):
    # Scene and task modules require Kit to be running.
    from sim import init_state
    from sim.envs.texture_sort import sample_layout
    from sim.manager_based import make_env
    from sim.tasks.registry import load_task

    task, knobs = load_task("texture_sort")
    env = make_env(
        "texture_sort", num_envs=1, device=args.device, scene_cfg=task.make_scene_cfg(knobs), seed=args.seed
    )
    try:
        env.reset(seed=args.seed)
        joints = env.scene["robot"].data.default_joint_pos[0].cpu().tolist()
        robot = dict(joint_pos=joints, joint_pos_target=joints.copy())
        states = []
        for pair in range(args.pairs):
            layout = sample_layout(np.random.default_rng([args.seed, pair, 0]))
            for variant in ("smooth", "rough"):
                state = task.layout_init_state(robot, layout, variant, len(states))
                state["meta"] = dict(
                    group_id=f"texture_sort-{args.seed}-{pair:06d}",
                    variant=variant,
                    experiment="surface_only",
                    layout_seed=[args.seed, pair, 0],
                )
                states.append(state)
        init_state.save(
            args.out,
            task=task.name,
            states=states,
            rejected=[],
            generator=dict(
                seed=args.seed, pairs=args.pairs, requires_settle=True, recipe="texture_sort_pairs"
            ),
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.pairs < 1 or args.seed < 0:
        parser.error("pairs must be positive and seed nonnegative")
    if Path(args.out).expanduser().exists():
        parser.error("out already exists; choose a new path")
    args.enable_cameras = True
    launcher = AppLauncher(args)
    try:
        generate(args)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
