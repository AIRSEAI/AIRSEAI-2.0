# SPDX-License-Identifier: Apache-2.0
"""Share ACT weights while keeping an independent action queue per environment."""

from collections import deque
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


class VectorACT:
    def __init__(self, checkpoint, device, num_envs, n_action_steps=None):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        if cfg.type != "act" or cfg.n_obs_steps != 1:
            raise ValueError("Vector evaluation requires ACT with n_obs_steps=1")
        if cfg.temporal_ensemble_coeff is not None:
            raise ValueError("Vector evaluation does not support temporal ensembling")
        self.model = get_policy_class(cfg.type).from_pretrained(self.checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.input_keys = list(cfg.input_features)
        self.steps = cfg.n_action_steps if n_action_steps is None else n_action_steps
        if not 1 <= self.steps <= cfg.chunk_size:
            raise ValueError("n_action_steps must be between 1 and chunk_size")
        self.chunk_size = cfg.chunk_size
        self.queues = [deque() for _ in range(num_envs)]

    def reset(self, slot):
        self.queues[slot].clear()

    def actions(self, observations):
        needed = [slot for slot in observations if not self.queues[slot]]
        if needed:
            batch = {
                key: torch.cat([observations[slot][key] for slot in needed], dim=0) for key in self.input_keys
            }
            with torch.inference_mode():
                chunks = self.model.predict_action_chunk(self.pre(batch))[:, : self.steps]
                if chunks.ndim != 3 or chunks.shape[1] != self.steps:
                    raise ValueError("Unexpected ACT action chunk shape")
                # Apply the same postprocessor shape used by select_action: batch x action.
                actions = torch.stack([self.post(chunks[:, i]) for i in range(self.steps)], dim=1)
                actions = actions.detach().cpu()
            if actions.shape != (len(needed), self.steps, 16) or not torch.isfinite(actions).all():
                raise ValueError("Expected finite 16-dimensional joint target actions")
            for index, slot in enumerate(needed):
                self.queues[slot].extend(actions[index].unbind(0))
        return {slot: self.queues[slot].popleft() for slot in observations}
