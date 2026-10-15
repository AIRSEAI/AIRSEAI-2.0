# SPDX-License-Identifier: Apache-2.0
"""The shared failure detectors, exercised without any task criterion."""

from __future__ import annotations

import numpy as np
import pytest

from sim.criteria.detectors import detect_grasp_failure, detect_mistiming, detect_slip


def test_slip_is_detected_and_named(held_trace, right_pads):
    tr = held_trace()
    pos = tr.object_pos["manipuland"].copy()
    pos[50:, 2] -= 0.02               # slid 20 mm down within the grasp
    tr.object_pos["manipuland"] = pos
    slipped, m = detect_slip(tr, "manipuland", right_pads)
    assert slipped and m["slip_m"] == pytest.approx(0.02)


def test_carrying_the_object_is_not_slip(held_trace, right_pads):
    # the whole grasp translates: displacement is large, slip is zero.
    tr = held_trace()
    pos = tr.object_pos["manipuland"].copy()
    pos[:, 0] += np.linspace(0, 0.5, len(pos))
    tr.object_pos["manipuland"] = pos
    tr.extra["grasp_frame_pos"] = pos.copy()
    slipped, m = detect_slip(tr, "manipuland", right_pads)
    assert not slipped, m


def test_mistiming_needs_both_contacts_to_exist(held_trace, right_pads, left_pads):
    tr = held_trace()
    tr.pad_area[right_pads[0]] = np.zeros(len(tr.t))       # nobody caught it
    flagged, m = detect_mistiming(tr, right_pads[0], left_pads[0], 0.5)
    assert flagged and np.isnan(m["gap_s"])


def test_hold_length_survives_decimation_240hz_to_20hz(held_trace, decimate, right_pads):
    """A hold lasts as long whatever rate it was recorded at."""
    fast = held_trace(n=1200, dt=1 / 240)
    slow = decimate(fast, 12)
    _, m_fast = detect_grasp_failure(fast, right_pads)
    _, m_slow = detect_grasp_failure(slow, right_pads)
    assert m_fast["longest_hold_s"] == pytest.approx(m_slow["longest_hold_s"], abs=0.06)
