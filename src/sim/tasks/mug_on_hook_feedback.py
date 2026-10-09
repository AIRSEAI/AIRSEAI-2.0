"""Contact confirmation using gel shear only; no object poses or hook height."""

from __future__ import annotations

import math
from collections import deque
from statistics import median


class ShearContact:
    """Compare shear against a recent free-carry baseline, rejecting spikes."""

    def __init__(self, ratio=1.25, hold_s=0.15):
        if not math.isfinite(ratio) or ratio <= 1.0 or not math.isfinite(hold_s) or hold_s <= 0.0:
            raise ValueError("Contact needs a shear ratio above one and a positive hold time")
        self.threshold = ratio
        # A contact oscillates around the high threshold. Require a strong
        # onset, then continuous elevation above a lower release threshold.
        self.reset_threshold = 1.0 + 0.4 * (ratio - 1.0)
        self.hold_s = hold_s
        self.free = deque(maxlen=7)
        self.baseline = None
        self.elapsed = 0.0
        self.ratio = 1.0

    def observe_free(self, shear, valid=True):
        if valid and math.isfinite(shear) and shear > 0.0:
            self.free.append(float(shear))
        else:
            self.free.clear()

    def start(self):
        if len(self.free) < 5:
            raise RuntimeError("mug_on_hook needs five valid gel shear samples before seating")
        self.baseline = median(self.free)

    def update(self, shear, dt, valid=True, held=True):
        if self.baseline is None:
            raise RuntimeError("Gel shear baseline has not been established")
        if not valid or not held or not math.isfinite(shear):
            self.elapsed = 0.0
            self.ratio = 0.0
            return False
        self.ratio = float(shear) / self.baseline
        threshold = self.reset_threshold if self.elapsed > 0.0 else self.threshold
        self.elapsed = self.elapsed + dt if self.ratio >= threshold else 0.0
        return self.elapsed + 1e-9 >= self.hold_s
