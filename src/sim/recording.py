# SPDX-License-Identifier: Apache-2.0
"""Record demonstrations and review videos with a shared data schema."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from sim.envs.bimanual_tactile import SENSOR_FEATURE_NAME, TACTILE_SENSORS
from sim.harness import CTRL_FPS

__all__ = [
    "REVIEW_DIRNAME",
    "EpisodeRecorder",
    "features",
]

# Review videos use the saved dataset episode index.
REVIEW_DIRNAME = ("review", "third_person")
CAMERAS = ("base", "left_wrist", "right_wrist")


def features(rows: int, cols: int) -> dict:
    """The recording's columns: three cameras, four tactile pads in three
    encodings each, the 16-dim state and action, and the episode flags."""
    from tactile_codec import PROFILES

    feats = {f"observation.images.{k}": {"dtype": "video", "shape": (480, 640, 3)} for k in CAMERAS}
    for pn in TACTILE_SENSORS:
        sname = SENSOR_FEATURE_NAME[pn]
        feats[f"observation.images.{sname}_tactile"] = {"dtype": "video", "shape": (288, 384, 3)}
        # Native TacSL force-field channels: depth is the normal component
        # (F_n = k_n*d, k_n = 1.0); ff_shear is `tactile_shear_force` x1e3.
        feats[f"observation.depth.{sname}_tactile"] = {
            "dtype": "data",
            "shape": (rows, cols, 1),
            "profile": PROFILES["sim_v1_depth"],
        }
        feats[f"observation.ff_shear.{sname}_tactile"] = {
            "dtype": "data",
            "shape": (rows, cols, 2),
            "profile": PROFILES["sim_v1_shear"],
        }
    names = [f"{a}_joint_{i}.pos" for a in ("right", "left") for i in range(1, 8)]
    names = names[:7] + ["right_gripper.pos"] + names[7:] + ["left_gripper.pos"]
    feats["observation.state"] = {"dtype": "float32", "shape": (16,), "names": names}
    feats["action"] = {"dtype": "float32", "shape": (16,), "names": names}
    for k in ("is_intervention", "control_mode", "is_success"):
        feats[k] = {"dtype": "int64", "shape": (1,)}
    return feats


class EpisodeRecorder:
    """Save or discard episodes together with their review videos."""

    def __init__(self, h, task, root: str, fps: int = CTRL_FPS, *, disk_buffer=False, durable=False):
        from lerobot_tactile_ext.writer import LeRobotTactileWriter

        if not task.instruction:
            raise SystemExit(f"task '{task.name}' has no `instruction`; the dataset needs a language label")
        self.h = h
        self.root = Path(os.path.expanduser(root))
        self.durable = bool(durable)
        self.writer = LeRobotTactileWriter(
            str(self.root),
            fps=fps,
            features=features(h.rows, h.cols),
            task=task.instruction,
            disk_buffer=disk_buffer,
            durable=self.durable,
        )
        self.fps = fps
        self._review_dir = self.root.joinpath(*REVIEW_DIRNAME) if h.scene_cam is not None else None
        self._review = None
        self._review_path = None
        self._annotations = []

    def capture_frame(self) -> dict:
        """Read all modalities from this environment at the current tick."""
        h = self.h
        h.update_cams(include_scene_cam=True)
        fr = {f"observation.images.{k}": h.cam_rgb(k) for k in CAMERAS}
        fr["observation.state"] = h.to16(h.robot.data.joint_pos[0])
        fr["action"] = h.action16_recorded()
        for pn in TACTILE_SENSORS:
            sname = SENSOR_FEATURE_NAME[pn]
            pen, sh = h.tactile_raw(pn)
            fr[f"observation.images.{sname}_tactile"] = h.tacrgb_u8(pn)
            fr[f"observation.depth.{sname}_tactile"] = pen
            fr[f"observation.ff_shear.{sname}_tactile"] = sh
        return fr

    def add_frame(self) -> np.ndarray:
        """Record before the physics step and return the recorded joint state."""
        fr = self.capture_frame()
        self.writer.add_frame(fr)
        h = self.h
        if h.scene_cam is not None:
            self.review_add(h.cam_rgb("scene"))
        return fr["observation.state"]

    def review_add(self, frame: np.ndarray) -> None:
        if self._review_dir is None:
            return
        if self._review is None:
            from sim.video import Mp4Writer

            self._review_path = self._review_dir / "_pending.mp4"
            self._review = Mp4Writer(self._review_path, self.fps)
        self._review.add(frame)

    def review_close(self, ep_index: int | None) -> None:
        """Keep the pending video as `episode_<ep_index>.mp4`, or drop it."""
        if self._review is None:
            return
        review, self._review = self._review, None
        review.close()
        if ep_index is None:
            self._review_path.unlink(missing_ok=True)
        else:
            self._review_path.rename(self._review_dir / f"episode_{ep_index:06d}.mp4")
        self._review = None

    def save_episode(
        self, ep_index: int, success: bool, *, is_intervention: bool = False, annotation: dict | None = None
    ) -> dict:
        stats = self.writer.save_episode(
            scalars={
                "is_intervention": int(is_intervention),
                "control_mode": int(is_intervention),
                "is_success": int(success),
            }
        )
        self.review_close(ep_index)
        if annotation is not None:
            self._annotations.append(dict(annotation, episode_index=ep_index))
            path = self.root / "meta/episode_annotations.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._annotations, indent=2) + "\n")
            temporary.replace(path)
        return stats

    def discard_episode(self) -> int:
        n = self.writer.discard_episode()
        self.review_close(None)
        return n

    def abort(self) -> None:
        """Release encoders without publishing an incomplete dataset."""
        try:
            self.review_close(None)
        finally:
            self.writer.abort()

    def finalize(self) -> dict:
        return self.writer.finalize()

    def checkpoint_recovery(self, *, decision=None, scalars=None, **context):
        if self.durable:
            self.writer.set_recovery_context(**context)
            if decision is not None:
                self.writer.set_recovery_decision(decision, scalars)
