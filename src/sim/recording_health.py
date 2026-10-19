"""Main-loop liveness for VR recording, including explicit CPU-only I/O waits."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import os
from pathlib import Path
import time


class RecordingHeartbeat:
    def __init__(self, path, *, enabled=True, interval=2.0, clock=None):
        self.path = Path(path)
        self.enabled = enabled
        self.interval = interval
        self.clock = clock or time.time
        self.last_write = None
        self.last_phase = None
        self.context = {}

    def emit(self, phase, *, force=False, **context):
        self.context.update(context)
        if not self.enabled:
            return False
        now = self.clock()
        if (not force and phase == self.last_phase and self.last_write is not None
                and now - self.last_write < self.interval):
            return False
        payload = dict(self.context, updated_at=now, phase=phase, pid=os.getpid())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.json.tmp')
        with temporary.open('w', encoding='utf-8') as output:
            json.dump(payload, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.path)
        self.last_write, self.last_phase = now, phase
        return True

    def run_io(self, phase, function):
        """Wait on CPU file/encoder work; never put simulation/GPU calls here."""
        if not self.enabled:
            return function()
        self.emit(phase, force=True)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='recording-io-wait') as pool:
            future = pool.submit(function)
            while True:
                try:
                    return future.result(timeout=.5)
                except TimeoutError:
                    if future.done():
                        # Preserve an actual TimeoutError raised by the worker.
                        return future.result()
                    self.emit(phase)
