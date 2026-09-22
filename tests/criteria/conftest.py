"""Trace builders shared by the criteria tests.

Hand-built traces, no Isaac and no GPU: every criterion is a pure function of
an `EpisodeTrace`.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.criteria.trace import EpisodeTrace


@pytest.fixture
def right_pads() -> tuple[str, ...]:
    return ("right_left_finger_tactile", "right_right_finger_tactile")


@pytest.fixture
def left_pads() -> tuple[str, ...]:
    return ("left_left_finger_tactile", "left_right_finger_tactile")


@pytest.fixture
def held_trace(right_pads, left_pads):
    """-> make(n, dt, area, obj_xyz): a clean episode, object held steadily."""
    def make(n=100, dt=0.05, area=20.0, obj_xyz=(0.3, 0.0, 0.2)) -> EpisodeTrace:
        return EpisodeTrace(
            t=np.arange(n) * dt,
            pad_area={p: np.full(n, area) for p in right_pads + left_pads},
            object_pos={"manipuland": np.tile(np.asarray(obj_xyz, dtype=float), (n, 1))},
        )
    return make


@pytest.fixture
def decimate():
    """-> make(trace, step): the same episode sampled every `step`th step."""
    def make(trace: EpisodeTrace, step: int) -> EpisodeTrace:
        return EpisodeTrace(
            t=trace.t[::step],
            pad_area={k: v[::step] for k, v in trace.pad_area.items()},
            object_pos={k: v[::step] for k, v in trace.object_pos.items()},
            extra={k: v[::step] for k, v in trace.extra.items()},
        )
    return make
