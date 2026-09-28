"""Restore frozen episode starts into selected slots, in environment-local coordinates."""

import copy

import torch


def _validate_slots(env, states, env_ids):
    ids = torch.as_tensor(env_ids, device=env.device)
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Environment IDs must be a one-dimensional integer sequence")
    slots = ids.tolist()
    if not slots or len(states) != len(slots) or len(set(slots)) != len(slots):
        raise ValueError("Provide one state per unique environment ID")
    if min(slots) < 0 or max(slots) >= env.num_envs:
        raise ValueError("Environment ID is out of range")
    return ids.long(), slots


def prepare_restore(env, states, env_ids, *, source_origin=(0.0, 0.0, 0.0)):
    """Clear selected contacts; call env.step once before committing the reset."""
    ids, env_ids = _validate_slots(env, states, env_ids)
    if not hasattr(env, "pending_restores"):
        env.pending_restores = {}
    for name in states[0]["objects"]:
        poses = torch.tensor([state["objects"][name] for state in states], device=env.device)
        poses[:, :3] += env.scene.env_origins[ids] - torch.tensor(source_origin, device=env.device)
        # Preserve relative poses of bodies connected by a spring joint.
        poses[:, 2] += 1.0
        env.scene[name].write_root_pose_to_sim(poses, env_ids=ids)
        env.scene[name].write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)
    for slot in env_ids:
        env.pending_restores[slot] = env._sim_step_counter
    env.scene.write_data_to_sim()


def restore_states(env, states, env_ids, *, source_origin=(0.0, 0.0, 0.0)):
    """Commit after prepare_restore and an ordinary recorded control tick."""
    ids, env_ids = _validate_slots(env, states, env_ids)
    if any(
        slot not in getattr(env, "pending_restores", {})
        or env._sim_step_counter <= env.pending_restores[slot]
        for slot in env_ids
    ):
        raise RuntimeError("prepare_restore and one global control tick must precede restore_states")
    if not hasattr(env, "frozen_task_vars"):
        env.frozen_task_vars = [None] * env.num_envs
        env.frozen_state_ids = [None] * env.num_envs
    origins = env.scene.env_origins[ids]
    source = torch.tensor(source_origin, device=env.device)
    robot = env.scene["robot"]
    pos = torch.tensor([s["robot"]["joint_pos"] for s in states], device=env.device)
    targets = torch.tensor([s["robot"]["joint_pos_target"] for s in states], device=env.device)
    env.scene.reset(ids)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=ids)
    robot.set_joint_position_target(targets, env_ids=ids)
    for name in states[0]["objects"]:
        poses = torch.tensor([s["objects"][name] for s in states], device=env.device)
        poses[:, :3] += origins - source
        obj = env.scene[name]
        obj.write_root_pose_to_sim(poses, env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)
    for slot, state in zip(env_ids, states, strict=True):
        env.frozen_task_vars[slot] = copy.deepcopy(state["task_vars"])
        env.frozen_state_ids[slot] = state["id"]
    if env.cfg.restore_task_state is not None:
        env.cfg.restore_task_state(env, states, env_ids)
    env.action_manager.reset(ids)
    env.observation_manager.reset(ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    # Refresh RTX after teleporting without advancing simulation time.
    for _ in range(env.cfg.num_rerenders_on_reset):
        env.sim.render()
    for slot in env_ids:
        del env.pending_restores[slot]


def snapshot_slot(env, slot, names):
    robot = env.scene["robot"].data
    return dict(
        robot=dict(
            joint_pos=robot.joint_pos[slot].cpu().tolist(),
            joint_pos_target=robot.joint_pos_target[slot].cpu().tolist(),
        ),
        objects={
            name: torch.cat(
                (
                    env.scene[name].data.root_pos_w[slot] - env.scene.env_origins[slot],
                    env.scene[name].data.root_quat_w[slot],
                )
            )
            .cpu()
            .tolist()
            for name in names
        },
        task_vars=copy.deepcopy(env.frozen_task_vars[slot]),
    )
