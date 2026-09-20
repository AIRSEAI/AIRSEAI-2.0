"""The verdict type's own invariants."""

from __future__ import annotations

import pytest

from sim.criteria.trace import FailureMode, Verdict


def test_verdict_rejects_inconsistent_states():
    with pytest.raises(ValueError):
        Verdict(True, FailureMode.SLIP)
    with pytest.raises(ValueError):
        Verdict(False)
