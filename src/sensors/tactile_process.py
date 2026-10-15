# SPDX-License-Identifier: Apache-2.0
"""Publish tactile snapshots through double-buffered shared memory."""

import multiprocessing as mp
import time
import traceback
from multiprocessing import shared_memory

import numpy as np

from .tactile_features import dataset_features, frame_from_sensor


def _layout(names, previews, height, width):
    layout, offset = [], 0
    for key, spec in dataset_features(names, previews, height, width).items():
        if key.startswith("observation.tactile_status."):
            continue
        dtype = np.dtype("uint8" if spec["dtype"] == "video" else "float32")
        layout.append((key, spec["shape"], dtype, offset))
        offset += int(np.prod(spec["shape"])) * dtype.itemsize
        offset = (offset + 7) // 8 * 8
    layout.append(("_status", (len(names), 2), np.dtype("float64"), offset))
    return layout, offset + len(names) * 16


def _views(shm, layout, base=0):
    return {
        key: np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=base + offset)
        for key, shape, dtype, offset in layout
    }


def _worker(shm_name, names, configs, options, locks, active, stop, ready, status, manager_factory):
    shm = manager = None
    try:
        if manager_factory is None:
            from .visuotactile_sensor import VisuotactileSensorManager

            manager_factory = VisuotactileSensorManager
        layout, size = _layout(names, options["previews"], options["height"], options["width"])
        shm = shared_memory.SharedMemory(name=shm_name)
        slots = [_views(shm, layout, i * size) for i in range(2)]
        manager = manager_factory(configs, backend=options["backend"], max_fps=options["sensor_fps"])
        manager.start()
        while not stop.value:
            started = time.monotonic()
            samples = manager.get_latest_data()
            if all(name in samples for name in names):
                frames = {}
                stamps = []
                for name in names:
                    sample = samples[name]
                    if started - sample["timestamp"] > options["max_age_s"]:
                        raise TimeoutError(f"Stale tactile sensor: {name}")
                    frames.update(
                        frame_from_sensor(
                            name,
                            sample,
                            previews=options["previews"],
                            height=options["height"],
                            width=options["width"],
                        )
                    )
                    stamps.append((sample["timestamp"], sample["frame_id"]))
                index = 1 - active.value
                if locks[index].acquire(False):
                    try:
                        for key, value in frames.items():
                            slots[index][key][:] = value
                        slots[index]["_status"][:] = stamps
                        active.value = index
                        ready.value = True
                    finally:
                        locks[index].release()
            time.sleep(max(0, 1 / options["fps"] - (time.monotonic() - started)))
    except BaseException:
        status.send(traceback.format_exc())
        ready.value = True
    finally:
        if manager is not None:
            manager.stop()
        if shm is not None:
            shm.close()
        status.close()


class TactileProcess:
    def __init__(
        self,
        sensor_configs,
        fps=20,
        *,
        previews=False,
        backend="cuda",
        sensor_fps=35,
        max_age_s=0.5,
        height=288,
        width=384,
        manager_factory=None,
    ):
        if not sensor_configs or fps <= 0 or sensor_fps <= 0 or max_age_s <= 0:
            raise ValueError("Tactile sensors and positive rates/timeouts are required")
        self.sensor_names = list(sensor_configs)
        self.configs = dict(sensor_configs)
        self.options = dict(
            fps=fps,
            previews=previews,
            backend=backend,
            sensor_fps=sensor_fps,
            max_age_s=max_age_s,
            height=height,
            width=width,
        )
        self.manager_factory = manager_factory
        self.layout, self.size = _layout(self.sensor_names, previews, height, width)
        self.shm = self.process = self.status = None
        self.last = None

    def start(self, ready_timeout_s=60):
        if self.process is not None:
            raise RuntimeError("Tactile process already started")
        ctx = mp.get_context("spawn")
        self.shm = shared_memory.SharedMemory(create=True, size=self.size * 2)
        self.locks = [ctx.Lock(), ctx.Lock()]
        self.active = ctx.RawValue("i", 0)
        self.stop_event, ready = ctx.RawValue("b", False), ctx.RawValue("b", False)
        self.status, child = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=_worker,
            args=(
                self.shm.name,
                self.sensor_names,
                self.configs,
                self.options,
                self.locks,
                self.active,
                self.stop_event,
                ready,
                child,
                self.manager_factory,
            ),
            name="tactile-process",
        )
        try:
            self.process.start()
            child.close()
            deadline = time.monotonic() + ready_timeout_s
            while not ready.value:
                self.check_health()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Tactile sensors did not produce complete frames before startup timeout"
                    )
                time.sleep(0.05)
            self.check_health()
            self.get_latest()
        except BaseException:
            child.close()
            self.stop()
            raise

    def check_health(self):
        if self.status is not None and self.status.poll():
            try:
                error = self.status.recv()
            except EOFError:
                error = "Tactile worker closed its status channel"
            raise RuntimeError(error)
        if self.process is None or not self.process.is_alive():
            raise RuntimeError("Tactile worker is not running")

    def get_latest(self):
        self.check_health()
        index = self.active.value
        if self.locks[index].acquire(False):
            try:
                self.last = {
                    key: value.copy()
                    for key, value in _views(self.shm, self.layout, index * self.size).items()
                }
            finally:
                self.locks[index].release()
        if self.last is None:
            raise RuntimeError("No tactile snapshot available")
        result = {key: value for key, value in self.last.items() if key != "_status"}
        now = time.monotonic()
        for name, (stamp, fid) in zip(self.sensor_names, self.last["_status"], strict=True):
            age = now - stamp
            if stamp <= 0 or age > self.options["max_age_s"]:
                raise TimeoutError(f"Stale tactile sensor: {name}")
            result[f"observation.tactile_status.{name}"] = np.array([1, age, fid], dtype=np.float32)
        return result

    def stop(self):
        if self.process is not None:
            self.stop_event.value = True
            if self.process.pid is not None:
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)
            self.process = None
        if self.status is not None:
            self.status.close()
            self.status = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
        self.last = None
