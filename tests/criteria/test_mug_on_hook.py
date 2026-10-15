# SPDX-License-Identifier: Apache-2.0
"""mug_on_hook: whether the peg threads the handle's eye decides, at any lean."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from sim.criteria.mug_on_hook import eye_margin, verdict
from sim.criteria.probes import axis_from_quat
from sim.criteria.trace import FailureMode
from sim.dims.mug_on_hook import EYE_FLOOR_Z, EYE_HEIGHT, EYE_X, MUG_OUTER_R

# A peg lying along +y at the origin's height, as the rail's does.
PEG_ROOT = (0.265, -0.240, 0.295)
PEG_TIP = (0.265, -0.171, 0.307)
# Where the peg is meant to end up inside the eye: its middle.
EYE_CENTRE = (MUG_OUTER_R + EYE_X / 2, 0.0, EYE_FLOOR_Z + EYE_HEIGHT / 2)


UPRIGHT = (1.0, 0.0, 0.0, 0.0)


def _turn(axis: str, deg: float):
    """(w, x, y, z) for a turn of `deg` about a world axis."""
    h = math.radians(deg) / 2.0
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return (math.cos(h), *(math.sin(h) * c for c in v))


def _hang_pos(quat=UPRIGHT, offset=(0.0, 0.0, 0.0), along=0.5):
    """Mug origin that puts `EYE_CENTRE + offset` on the peg at fraction `along`."""
    peg = np.asarray(PEG_ROOT) + along * (np.asarray(PEG_TIP) - np.asarray(PEG_ROOT))
    local = np.asarray(EYE_CENTRE) + np.asarray(offset)
    return peg - axis_from_quat(quat, local)


@pytest.fixture
def mug_trace(held_trace):
    """-> make(pos, quat, ...): an episode ending with the mug at that pose."""

    def make(pos, quat=UPRIGHT, n=100, area=20.0, end_area=None, slip=0.0):
        tr = held_trace(n=n, area=area)
        tr.object_pos.pop("manipuland")
        p = np.tile(np.asarray(pos, dtype=float), (n, 1))
        # A bump, not a ramp: the mug wanders in the grasp and comes back, so
        # `slip` changes the path without moving the final pose the verdict reads.
        p[:, 0] += slip * np.sin(np.linspace(0.0, np.pi, n))
        tr.object_pos["mug"] = p
        tr.object_quat["mug"] = np.tile(np.asarray(quat, dtype=float), (n, 1))
        if end_area is not None:
            for k in tr.pad_area:
                tr.pad_area[k] = tr.pad_area[k].copy()
                tr.pad_area[k][-1] = end_area
        return tr

    return make


def _v(tr, right_pads):
    return verdict(tr, peg_root=PEG_ROOT, peg_tip=PEG_TIP, pads=right_pads)


# ---- the probe --------------------------------------------------------------
def test_margin_is_the_clearance_at_the_eye_centre():
    m = eye_margin(_hang_pos(UPRIGHT), UPRIGHT, PEG_ROOT, PEG_TIP)
    assert m == pytest.approx(EYE_X / 2, abs=1e-6)


def test_margin_goes_negative_past_the_handles_outer_bar():
    pos = _hang_pos(UPRIGHT, offset=(EYE_X, 0.0, 0.0))
    assert eye_margin(pos, UPRIGHT, PEG_ROOT, PEG_TIP) < 0.0


def test_a_peg_that_never_cuts_the_handle_plane_is_outside():
    """The mug on the table beside the rail: the segment ends short of it."""
    m = eye_margin((0.10, 0.20, 0.22), UPRIGHT, PEG_ROOT, PEG_TIP)
    assert m < 0.0


# ---- the verdict ------------------------------------------------------------
def test_passes_when_the_peg_threads_the_eye(mug_trace, right_pads):
    v = _v(mug_trace(_hang_pos(UPRIGHT), end_area=0.0), right_pads)
    assert v.success and v.metrics["eye_margin_m"] > 0.0


def test_passes_while_the_mug_hangs_leaning(mug_trace, right_pads):
    """A hung mug turns about the peg until its mass is under it; the test is
    topological, so the lean must not change the answer."""
    q = _turn("y", 40.0)
    v = _v(mug_trace(_hang_pos(q), q, end_area=0.0), right_pads)
    assert v.success


def test_passes_when_the_mug_hangs_yawed(mug_trace, right_pads):
    q = _turn("z", 12.0)
    v = _v(mug_trace(_hang_pos(q), q, end_area=0.0), right_pads)
    assert v.success


def test_released_beyond_the_handle_is_mistiming(mug_trace, right_pads):
    pos = _hang_pos(UPRIGHT, offset=(EYE_X, 0.0, 0.0))
    assert _v(mug_trace(pos, end_area=0.0), right_pads).failure is FailureMode.MISTIMING


def test_released_against_the_body_is_mistiming(mug_trace, right_pads):
    pos = _hang_pos(UPRIGHT, offset=(-EYE_X, 0.0, 0.0))
    assert _v(mug_trace(pos, end_area=0.0), right_pads).failure is FailureMode.MISTIMING


def test_resting_over_the_handles_top_is_mistiming(mug_trace, right_pads):
    """Hand let go with the peg above the handle: supported, but not threaded."""
    pos = _hang_pos(UPRIGHT, offset=(0.0, 0.0, -EYE_HEIGHT))
    assert _v(mug_trace(pos, end_area=0.0), right_pads).failure is FailureMode.MISTIMING


def test_fallen_on_the_table_is_mistiming(mug_trace, right_pads):
    q = _turn("y", 90.0)
    assert _v(mug_trace((0.26, -0.20, 0.219), q, end_area=0.0), right_pads).failure is FailureMode.MISTIMING


def test_still_in_the_pads_at_the_end_is_unfinished(mug_trace, right_pads):
    pos = _hang_pos(UPRIGHT, offset=(EYE_X, 0.0, 0.0))
    assert _v(mug_trace(pos), right_pads).failure is FailureMode.UNFINISHED


def test_never_held_is_a_grasp_failure(mug_trace, right_pads):
    assert _v(mug_trace(_hang_pos(UPRIGHT), area=0.0), right_pads).failure is FailureMode.GRASP_FAILURE


def test_slip_is_named_only_when_the_mug_is_not_hung(mug_trace, right_pads):
    pos = _hang_pos(UPRIGHT, offset=(EYE_X, 0.0, 0.0))
    v = _v(mug_trace(pos, end_area=0.0, slip=0.05), right_pads)
    assert v.failure is FailureMode.SLIP


def test_a_hung_mug_that_slipped_still_succeeds(mug_trace, right_pads):
    """Seating drives the hand down against the peg, so the mug rides up in
    the gel. Process quantities name failures; they do not veto a success."""
    v = _v(mug_trace(_hang_pos(UPRIGHT), end_area=0.0, slip=0.05), right_pads)
    assert v.success


def test_threshold_defaults_come_from_dims():
    defaults = {k: p.default for k, p in inspect.signature(verdict).parameters.items()}
    assert defaults["eye_x"] == EYE_X
