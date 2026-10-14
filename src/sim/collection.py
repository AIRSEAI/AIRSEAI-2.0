"""Collect independent episodes on a shared simulation clock."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from lerobot_tactile_ext.merge import merge_recordings
from sim import init_state
from sim.collection_io import EpisodeWrite, EpisodeWriteQueue, write_json
from sim.envs.bimanual_tactile import TACTILE_SENSORS
from sim.harness import CTRL_FPS
from sim.manager_based import make_env
from sim.manager_based.common.mdp import hold_action
from sim.manager_based.common.slot import SlotHarness
from sim.manager_based.common.states import prepare_restore, restore_states, snapshot_slot
from sim.recording import EpisodeRecorder
from sim.teleop import make_device
from sim.video import Mp4Writer, compose_grid_frame
from sim.xr_idle import render_completed_scene, should_render_completed_scene

CONTROL_DT = 1 / CTRL_FPS


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def state_hash(state):
    payload = {key: state[key] for key in ("robot", "objects", "task_vars")}
    encoded = json.dumps(payload, sort_keys=True, default=plain, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_states(document, task, knobs, joint_count):
    if document["task"] != task.task_id(knobs):
        raise ValueError("Initial-state task does not match the selected task")
    seen = set()
    for state in document["states"]:
        identity = json.dumps(state["id"], sort_keys=True)
        if identity in seen:
            raise ValueError("Initial-state IDs must be unique")
        seen.add(identity)
        if set(state["objects"]) != set(task.init_state_objects()):
            raise ValueError(f"Object names disagree in state {state['id']}")
        for name in ("joint_pos", "joint_pos_target"):
            values = np.asarray(state["robot"][name], dtype=float)
            if values.shape != (joint_count,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid robot {name} in state {state['id']}")
        for pose in state["objects"].values():
            values = np.asarray(pose, dtype=float)
            if (
                values.shape != (7,)
                or not np.isfinite(values).all()
                or not np.isclose(np.linalg.norm(values[3:]), 1, atol=1e-4)
            ):
                raise ValueError(f"Invalid object pose in state {state['id']}")
        if not isinstance(state["task_vars"], dict) or not isinstance(state.get("meta", {}), dict):
            raise ValueError("task_vars and meta must be objects")
        entropy = state.get("meta", {}).get("expert_seed")
        if entropy is not None and (
            not isinstance(entropy, list)
            or len(entropy) != 3
            or any(type(value) is not int or value < 0 for value in entropy)
            or entropy[2] != 1
        ):
            raise ValueError("expert_seed must be [nonnegative seed, nonnegative index, 1]")
        task.validate_init_state(state)
        state_hash(state)


@dataclass
class Attempt:
    attempt_id: int
    env_id: int
    requested: dict
    folder: Path
    mode: str = "clearing"
    settle_ticks: int = 0
    frame: int = 0
    phase: int = 0
    phase_frame: int = 0
    ctx: dict = field(default_factory=dict)
    hist: dict = field(default_factory=lambda: {"t": []})
    trace: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    recorder: object = None
    controller: object = None
    paused: bool = False
    terminal_reason: str | None = None
    marked_success: bool = False
    target: object = None
    frozen: bool = False


class Collector:
    def __init__(self, args, task, knobs, app, *, teleop=False):
        self.args, self.task, self.knobs, self.app = args, task, knobs, app
        self.teleop = teleop
        self.xr_state_only = bool(getattr(args, "xr_state_only", False))
        if self.xr_state_only and (
            not teleop or args.teleop_device != "vr" or args.dataset or getattr(args, "video", "")
        ):
            raise ValueError("XR state-only mode requires VR teleop without dataset or video")
        self.stop_requested = lambda: False
        self.num_envs = args.num_envs
        self.write_mode = getattr(args, "write_mode", None) or "background"
        self.write_queue_size = getattr(args, "write_queue_size", 4)
        self.episode_write_workers = getattr(args, "episode_write_workers", 2)
        if (
            self.write_mode not in ("sync", "background")
            or self.write_queue_size < 1
            or self.episode_write_workers < 1
        ):
            raise ValueError(
                "write_mode must be sync or background; queue size and worker count must be positive"
            )
        if self.num_envs < 1 or (teleop and self.num_envs != 1):
            raise ValueError("Teleoperation requires exactly one environment")
        if args.seed < 0 or args.episodes < 1:
            raise ValueError("seed must be nonnegative and episodes positive")
        if getattr(args, "init_states_n", 0):
            raise ValueError("Vector collection uses a fixed attempt count; omit init_states_n")
        if getattr(args, "video_tactile", False):
            raise ValueError("Vector --video is a scene mosaic; use dataset tactile previews")
        self.start = getattr(args, "episode_start", 0)
        if self.start < 0:
            raise ValueError("episode_start must be nonnegative")
        dataset = getattr(args, "record_dataset", "") if not teleop else args.dataset
        self.dataset = Path(dataset).expanduser().resolve() if dataset else None
        self.out = (
            Path(
                args.out
                or (str(self.dataset) + "_run" if self.dataset else f"outputs/{task.name}_collection")
            )
            .expanduser()
            .resolve()
        )
        if (
            self.dataset
            and self.out != self.dataset / "run"
            and (
                self.dataset == self.out
                or self.dataset in self.out.parents
                or self.out in self.dataset.parents
            )
        ):
            raise ValueError("Dataset and report directories must not overlap")
        for path in (self.out, self.dataset):
            if path is not None and path.exists():
                raise FileExistsError(f"Choose a new output directory: {path}")
        self.document = init_state.load(args.init_states) if args.init_states else None
        if self.document and self.start:
            raise ValueError("episode_start cannot be used with init_states")
        self.count = len(self.document["states"]) if self.document else args.episodes
        self.stop = self.start + self.count
        for option in ("init_states_out", "video"):
            value = getattr(args, option, "")
            if value and Path(value).expanduser().exists():
                raise FileExistsError(f"Choose a new {option} path: {value}")
        video = getattr(args, "video", "")
        if video and Path(video).expanduser().with_suffix(".frames.json").exists():
            raise FileExistsError("Video metadata already exists; choose a new video path")
        cfg = task.make_scene_cfg(knobs)
        for pad in TACTILE_SENSORS:
            getattr(cfg, pad).tactile_array_size = tuple(args.grid)
        self.env = make_env(
            args.task,
            num_envs=self.num_envs,
            device=args.device,
            robot_usd=args.robot_usd,
            scene_cfg=cfg,
            xr_state_only=self.xr_state_only,
            seed=args.seed,
            review_size=getattr(args, "video_size", (960, 720)),
        )
        self.env.reset(seed=args.seed)
        if self.xr_state_only and os.environ.get("AIRSEAI_XR_DIAG") == "1":
            import carb
            from omni.kit.viewport.utility import get_active_viewport

            settings = carb.settings.get_settings()
            viewport = get_active_viewport()
            print(
                "[AIRSEAI_XR_RENDER] "
                f"mode={self.env.sim.render_mode} "
                f"xr_enabled={settings.get('/app/xr/enabled')} "
                f"offscreen={settings.get('/isaaclab/render/offscreen')} "
                f"active_viewport={settings.get('/isaaclab/render/active_viewport')} "
                f"viewport_updates={getattr(viewport, 'updates_enabled', None)}",
                flush=True,
            )
        default = self.env.scene["robot"].data.default_joint_pos[0].cpu().tolist()
        self.robot_start = dict(joint_pos=default, joint_pos_target=default.copy())
        if self.document:
            validate_states(self.document, task, knobs, len(default))
        self.handles = [SlotHarness(self.env, slot, tuple(args.grid)) for slot in range(self.num_envs)]
        self.active = [None] * self.num_envs
        self.next_ids = [self.start + slot for slot in range(self.num_envs)]
        self.reports, self.snapshots = [], []
        self.tick = 0
        self.controller = None
        self.video_path = Path(args.video).expanduser() if getattr(args, "video", "") else None
        self.video = Mp4Writer(self.video_path, CTRL_FPS) if self.video_path else None
        self.video_frames = []
        self.out.mkdir(parents=True, exist_ok=True)
        versions = {}
        for package in ("isaaclab", "isaacsim", "torch", "numpy", "lerobot"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        write_json(
            self.out / "configuration.json",
            dict(
                arguments=vars(args),
                **(
                    {
                        "recording": dict(
                            mode="xr_state_action_only",
                            saved=[
                                "initial_state.json",
                                "trajectory.json:frames[].state (16D)",
                                "trajectory.json:frames[].action (16D)",
                                "trajectory.json:signals",
                                "trajectory.json:frames[].xr_input",
                                "terminal_state.json",
                            ],
                            omitted=[
                                "LeRobot dataset",
                                "RGB camera streams",
                                "tactile RGB images",
                                "tactile depth arrays",
                                "tactile shear arrays",
                            ],
                        )
                    }
                    if self.xr_state_only
                    else {}
                ),
                versions=versions,
                python=sys.version,
                rng="numpy.PCG64",
                scene_stream=getattr(self, "scene_stream", "[seed, attempt_id, 0]"),
                expert_stream="meta.expert_seed when supplied; otherwise [seed, attempt_id, 1]",
                device=str(self.env.device),
                joint_names=self.env.scene["robot"].joint_names,
            ),
        )
        self.report_path = self.out / (task.name + ("_teleop.json" if teleop else "_scripted.json"))
        self.writes = EpisodeWriteQueue(self.write_mode, self.write_queue_size, self.episode_write_workers)

    def report(self, status):
        episodes = sorted(self.reports, key=lambda row: row["attempt_id"])
        successes = sum(bool(row.get("criterion_success")) for row in episodes)
        rate = successes / len(episodes) if episodes else 0.0
        write_json(
            self.report_path,
            dict(
                task=self.task.task_id(self.knobs),
                seed=self.args.seed,
                num_envs=self.num_envs,
                physics_hz=60,
                control_hz=CTRL_FPS,
                fps=CTRL_FPS,
                status=status,
                requested_attempts=self.count,
                n_episodes=len(episodes),
                n_success=successes,
                success_rate=rate,
                checks={
                    "expert_success_rate": {
                        "pass": bool(len(episodes) == self.count and rate > 0.90),
                        "minimum_exclusive": 0.90,
                        "n": successes,
                        "of": len(episodes),
                    }
                },
                episodes=episodes,
                active_attempts=[a.attempt_id for a in self.active if a],
                pending_writes=self.writes.attempt_ids,
                write_mode=self.write_mode,
                write_queue_size=self.write_queue_size,
                episode_write_workers=self.episode_write_workers,
                knobs=self.knobs,
                init_states_sha256=(init_state.digest(self.args.init_states) if self.document else None),
            ),
        )

    def sample_state(self, attempt_id):
        return (
            copy.deepcopy(self.document["states"][attempt_id])
            if self.document
            else self.task.sample_init_state(
                self.robot_start,
                np.random.default_rng([self.args.seed, attempt_id, 0]),
                attempt_id,
                self.knobs,
            )
        )

    def schedule(self):
        for slot, attempt in enumerate(self.active):
            if attempt is None:
                self.assign(slot)

    def assign(self, slot):
        attempt_id = self.next_ids[slot]
        if attempt_id >= self.stop:
            return
        self.next_ids[slot] += self.num_envs
        state = self.sample_state(attempt_id)
        folder = self.out / "attempts" / f"{attempt_id:06d}"
        folder.mkdir(parents=True)
        attempt = Attempt(attempt_id, slot, state, folder)
        attempt.report = dict(
            attempt_id=attempt_id,
            **({"recording_mode": "xr_state_action_only"} if self.xr_state_only else {}),
            episode=attempt_id,
            env_id=slot,
            run_seed=self.args.seed,
            init_state_id=state["id"],
            input_state_hash=state_hash(state),
            init_state_meta=copy.deepcopy(state.get("meta", {})),
            timeouts=[],
        )
        self.active[slot] = attempt
        write_json(folder / "requested_state.json", state)
        prepare_restore(self.env, [state], [slot])

    def snapshot_start(self, attempt):
        h = self.handles[attempt.env_id]
        h.reset_ik()
        h.unlatch()
        attempt.ctx = self.task.begin_restored_episode(h, attempt.requested, self.knobs)
        actual = snapshot_slot(self.env, attempt.env_id, self.task.init_state_objects())
        actual["task_vars"] = self.task.snapshot_init_state(h, attempt.ctx)
        self.env.frozen_task_vars[attempt.env_id] = copy.deepcopy(actual["task_vars"])
        actual.update(id=attempt.requested["id"], meta=copy.deepcopy(attempt.requested.get("meta", {})))
        return actual

    def begin(self, attempt):
        attempt.requested.setdefault("meta", {}).setdefault(
            "expert_seed", [self.args.seed, attempt.attempt_id, 1]
        )
        actual = self.snapshot_start(attempt)
        h = self.handles[attempt.env_id]
        attempt.report["init_state_hash"] = state_hash(actual)
        write_json(attempt.folder / "initial_state.json", actual)
        self.snapshots.append((attempt.attempt_id, actual))
        if self.dataset:
            attempt.recorder = EpisodeRecorder(h, self.task, str(attempt.folder / "raw"), disk_buffer=True)
        if self.teleop:
            if self.controller is None:
                self.controller = make_device(
                    self.args.teleop_device,
                    self.task,
                    h,
                    knobs=self.knobs,
                    pos_speed=self.args.pos_speed,
                    rot_speed=self.args.rot_speed,
                    jaw_speed=self.args.jaw_speed,
                    **(
                        {
                            "vr_joint_speed_deg_s": getattr(self.args, "vr_joint_speed_deg_s", 30.0),
                            "vr_position_gain": getattr(self.args, "vr_position_gain", 1.0),
                        }
                        if self.args.teleop_device == "vr"
                        else {}
                    ),
                )
            attempt.controller = self.controller
            attempt.controller.begin_episode(attempt.ctx)
            attempt.paused = self.args.teleop_device != "scripted"
            print(attempt.controller.help_text(), flush=True)
        if not self.teleop or self.args.teleop_device == "scripted":
            entropy = attempt.requested.get("meta", {}).get(
                "expert_seed", [self.args.seed, attempt.attempt_id, 1]
            )
            attempt.report["expert_seed"] = entropy
            attempt.report["expert"] = self.task.prepare_expert(attempt.ctx, entropy[0], entropy[1])
        attempt.mode = "running"

    def command(self, attempt):
        h, task = self.handles[attempt.env_id], self.task
        if self.teleop:
            cmd = attempt.controller.advance(attempt.frame * CONTROL_DT)
            if cmd.toggle_pause:
                attempt.paused = not attempt.paused
            if cmd.mark_success or cmd.discard or cmd.reset_episode or cmd.finish:
                attempt.marked_success = cmd.mark_success
                attempt.terminal_reason = (
                    "discarded" if cmd.discard else "reset" if cmd.reset_episode else "finished"
                )
                return False
            if attempt.paused or not cmd.tracking_valid:
                return False
            attempt.controller.apply(cmd)
            phase = "teleop"
        else:
            phase, _ = task.phases[attempt.phase]
            sig = task.signals(h, attempt.ctx)
            target, grip, quat = task.phase_target(
                h, attempt.ctx, phase, attempt.phase_frame * CONTROL_DT, sig, self.knobs
            )
            attempt.target = target
            speed, angular = getattr(task, "speed_for", lambda _: (None, None))(phase)
            if phase in task.joint_phases:
                h.step_joint_toward(target, grip)
            else:
                h.step_ik_toward(target, grip, quat, max_speed=speed, max_ang_speed=angular)
        attempt.trace.append(
            dict(
                frame_index=attempt.frame,
                global_tick=self.tick,
                phase=phase,
                state=h.to16(h.robot.data.joint_pos[0]),
                action=h.action16_recorded(),
                **(
                    {"xr_input": attempt.controller.diagnostics_snapshot()}
                    if self.teleop and hasattr(attempt.controller, "diagnostics_snapshot")
                    else {}
                ),
            )
        )
        if attempt.recorder:
            attempt.recorder.add_frame()
        return True

    def after_step(self, attempt):
        h, task = self.handles[attempt.env_id], self.task
        attempt.frame += 1
        attempt.phase_frame += 1
        sig = task.signals(h, attempt.ctx)
        attempt.hist["t"].append(attempt.frame * CONTROL_DT)
        for key, value in sig.items():
            attempt.hist.setdefault(key, []).append(value)
        if self.teleop:
            attempt.controller.after_step(sig)
        else:
            name, timeout = task.phases[attempt.phase]
            elapsed = attempt.phase_frame * CONTROL_DT
            done = task.phase_done(h, attempt.ctx, name, elapsed, sig, attempt.target, self.knobs)
            if done or elapsed >= timeout:
                if not done:
                    attempt.report["timeouts"].append(name)
                jump = task.next_phase(attempt.ctx, name)
                names = [name for name, _ in task.phases]
                attempt.phase = names.index(jump) if jump is not None else attempt.phase + 1
                attempt.phase_frame = 0
                if attempt.phase >= len(task.phases):
                    attempt.terminal_reason = "finished"
        if attempt.frame * CONTROL_DT >= task.horizon_s and not attempt.terminal_reason:
            attempt.terminal_reason = "timeout"

    def finish(self, attempt):
        self.collect_writes()
        while self.writes.full:
            self.collect_writes(block=True)
        h = self.handles[attempt.env_id]
        row = attempt.report
        row.update(
            status=attempt.terminal_reason,
            frames=attempt.frame,
            marked_success=attempt.marked_success,
            criterion_success=False,
            saved=False,
        )
        if attempt.frame:
            trace = self.task.build_trace(h, attempt.ctx, attempt.hist)
            verdict = self.task.verdict(trace, attempt.ctx, self.knobs)
            row.update(verdict=verdict.as_dict(), criterion_success=bool(verdict.success))
            row.update(self.task.episode_extra(attempt.ctx, attempt.hist))
        row["success_source"] = (
            "both"
            if row["criterion_success"] and attempt.marked_success
            else "operator"
            if attempt.marked_success
            else "criterion"
            if row["criterion_success"]
            else "none"
        )
        keep = (
            attempt.frame > 0
            and attempt.terminal_reason not in ("discarded", "reset", "interrupted")
            and (
                row["criterion_success"]
                or attempt.marked_success
                or getattr(self.args, "keep_unclean", False)
            )
        )
        terminal = snapshot_slot(self.env, attempt.env_id, self.task.init_state_objects())
        observation = None
        if attempt.recorder:
            observation = {
                key: np.asarray(value).copy() for key, value in attempt.recorder.capture_frame().items()
            }
        job = EpisodeWrite(
            folder=attempt.folder,
            row=copy.deepcopy(row),
            terminal=copy.deepcopy(dict(terminal, global_tick=self.tick)),
            trajectory=copy.deepcopy(dict(frames=attempt.trace, signals=attempt.hist)),
            observation=observation,
            recorder=attempt.recorder,
            keep=keep,
            intervention=self.teleop and self.args.teleop_device != "scripted",
        )
        # The worker owns the recorder and only receives detached CPU data.
        if job.recorder:
            job.recorder.h = None
        self.writes.submit(job)
        attempt.recorder = None
        self.active[attempt.env_id] = None
        self.collect_writes()
        self.report("running")

    def collect_writes(self, *, block=False, drain=False, raise_errors=True):
        rows = self.writes.collect(block=block, drain=drain)
        self.reports.extend(rows)
        failed = [row for row in rows if row.get("write_status") == "failed"]
        if failed and raise_errors:
            details = "; ".join(f"attempt {row['attempt_id']}: {row['error']}" for row in failed)
            raise RuntimeError(f"Episode writing failed: {details}")

    def video_state(self, slot, attempt):
        if attempt is None:
            return dict(env_id=slot, attempt_id=None, episode_step=None, phase="idle", status="idle")
        if attempt.mode != "running":
            phase = attempt.mode
        elif self.teleop:
            phase = "teleop"
        elif attempt.phase < len(self.task.phases):
            phase = self.task.phases[attempt.phase][0]
        else:
            phase = "complete"
        status = attempt.terminal_reason or attempt.mode
        return dict(
            env_id=attempt.env_id,
            attempt_id=attempt.attempt_id,
            episode_step=attempt.frame,
            phase=phase,
            status=status,
        )

    def add_video_frame(self):
        states = [self.video_state(slot, attempt) for slot, attempt in enumerate(self.active)]
        frames = [handle.cam_rgb("scene") for handle in self.handles]
        captions = []
        for slot, state in enumerate(states):
            attempt = "-" if state["attempt_id"] is None else state["attempt_id"]
            step = "-" if state["episode_step"] is None else state["episode_step"]
            captions.append(
                (
                    f"env {slot} | attempt {attempt} | step {step}",
                    f"phase {state['phase']} | status {state['status']}",
                )
            )
        width = math.ceil(math.sqrt(len(frames)))
        self.video.add(compose_grid_frame(frames, captions, columns=width))
        self.video_frames.append(
            dict(
                frame=len(self.video_frames),
                global_tick=self.tick,
                sim_time_s=self.tick * CONTROL_DT,
                slots=states,
            )
        )

    def run(self):
        status = "interrupted"
        try:
            self.report("running")
            while not self.stop_requested() and self.app.is_running():
                self.collect_writes()
                self.schedule()
                if not any(self.active):
                    status = "complete"
                    break
                moving = []
                clearing = []
                for attempt in self.active:
                    if attempt is None:
                        continue
                    if attempt.mode == "clearing":
                        clearing.append(attempt)
                    elif attempt.mode == "running" and self.command(attempt):
                        moving.append(attempt)
                # Pausing a human recording freezes simulation, so no unrecorded motion occurs.
                if (
                    self.teleop
                    and not moving
                    and not clearing
                    and all(a is None or a.mode == "running" for a in self.active)
                ):
                    self.env.sim.render()
                else:
                    self.env.step(hold_action(self.env).clone())
                    self.tick += 1
                    for attempt in moving:
                        self.after_step(attempt)
                    for attempt in self.active:
                        if attempt and attempt.mode == "settling":
                            attempt.settle_ticks -= 1
                            if attempt.settle_ticks <= 0:
                                self.begin(attempt)
                    for attempt in clearing:
                        restore_states(self.env, [attempt.requested], [attempt.env_id])
                        if attempt.frozen or (
                            self.document
                            and not self.document.get("generator", {}).get("requires_settle", False)
                        ):
                            self.begin(attempt)
                        else:
                            attempt.mode = "settling"
                            attempt.settle_ticks = math.ceil(self.task.settle_seconds / CONTROL_DT)
                if self.video:
                    self.add_video_frame()
                for attempt in list(self.active):
                    if attempt and attempt.terminal_reason:
                        self.finish(attempt)
            for attempt in list(self.active):
                if attempt:
                    attempt.terminal_reason = "interrupted"
                    self.finish(attempt)
            self.collect_writes(drain=True)
            kept = self.saved_rows()
            if kept and self.dataset:
                merge_recordings(
                    [self.out / "attempts" / f"{r['attempt_id']:06d}" / "raw" for r in kept], self.dataset
                )
                for index, row in enumerate(kept):
                    row["dataset_episode_index"] = index
                    write_json(self.out / "attempts" / f"{row['attempt_id']:06d}" / "result.json", row)
                if self.teleop and self.args.previews:
                    script = Path(__file__).resolve().parents[2] / "benchmark/run/render_previews.py"
                    subprocess.run([sys.executable, str(script), "--dataset", str(self.dataset)], check=True)
            elif self.dataset and status == "complete" and not getattr(self.args, "allow_empty", False):
                raise RuntimeError("No successful demonstrations were accepted")
            payload = self.state_output()
            init_state.save(
                self.out / "init_states.json",
                **payload,
            )
            if getattr(self.args, "init_states_out", ""):
                init_state.save(
                    self.args.init_states_out,
                    **payload,
                )
            self.close_recorders()
            self.report(status)
            if should_render_completed_scene(
                status=status,
                teleop=self.teleop,
                args=self.args,
            ):
                render_completed_scene(
                    self.args.xr_keep_open_s,
                    render=self.env.sim.render,
                    is_running=self.app.is_running,
                    stop_requested=self.stop_requested,
                    log=lambda message: print(message, flush=True),
                )
            return 0 if status == "complete" else 130
        except BaseException as error:
            self.collect_writes(drain=True, raise_errors=False)
            completed_ids = {row["attempt_id"] for row in self.reports}
            for attempt in self.active:
                if attempt and attempt.attempt_id not in completed_ids:
                    attempt.report.update(
                        status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                        saved=False,
                        error=str(error),
                    )
                    write_json(attempt.folder / "result.json", attempt.report)
                    self.reports.append(attempt.report)
            self.report("failed")
            raise
        finally:
            self.writes.close()
            try:
                self.close_recorders()
            finally:
                self.env.close()

    def saved_rows(self):
        return sorted((r for r in self.reports if r["saved"]), key=lambda r: r["attempt_id"])

    def state_output(self):
        return dict(
            task=self.task.task_id(self.knobs),
            states=[state for _, state in sorted(self.snapshots)],
            rejected=[],
            generator=dict(seed=self.args.seed, num_envs=self.num_envs, requires_settle=False),
        )

    def close_recorders(self):
        errors = []
        for attempt in self.active:
            if attempt and attempt.recorder:
                try:
                    recorder, attempt.recorder = attempt.recorder, None
                    recorder.abort()
                except Exception as error:
                    errors.append(error)
        if self.video:
            try:
                video, self.video = self.video, None
                stats = video.close()
                write_json(
                    self.video_path.with_suffix(".frames.json"),
                    dict(**stats, frames_metadata=self.video_frames),
                )
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Recording cleanup failed", errors)


def run_collection(args, task, knobs, app, *, teleop=False):
    stopping = False

    def interrupted(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        collector = Collector(args, task, knobs, app, teleop=teleop)
        collector.stop_requested = lambda: stopping
        try:
            return collector.run()
        except KeyboardInterrupt:
            return 130
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
