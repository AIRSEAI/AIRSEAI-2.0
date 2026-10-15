# SPDX-License-Identifier: Apache-2.0
"""Sample whole-GPU memory, including RTX allocations outside PyTorch."""

import subprocess
import threading


class GPUMemoryMonitor:
    def __init__(self, device):
        self.index = str(device).split(":")[-1] if ":" in str(device) else "0"
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                value = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            self.index,
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=3,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
                self.samples.append(int(value))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(0.5)

    def finish(self):
        self._stop.set()
        self._thread.join(timeout=4)
        return dict(
            peak_mib=max(self.samples) if self.samples else None,
            samples=len(self.samples),
            interval_seconds=0.5,
            scope="whole GPU including RTX and other processes",
        )
