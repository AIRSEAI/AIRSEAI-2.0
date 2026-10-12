"""Acquire coherent DM-Tac samples inside the tactile worker."""

import threading
import time

import numpy as np


class VisuotactileSensorManager:
    def __init__(self, sensor_configs, backend="cuda", max_fps=35):
        self.configs = sensor_configs
        self.backend = backend
        self.max_fps = max_fps
        self.sensors = {}
        self.threads = []
        self.samples = {}
        self.errors = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def start(self):
        from dmrobotics import Mode, Sensor, SensorOptions

        try:
            for name, serial in self.configs.items():
                sensor = Sensor(
                    SensorOptions(
                        dev_id=serial,
                        backend=self.backend,
                        mode=Mode.STANDARD,
                        show_fps=False,
                        max_fps=self.max_fps,
                        enable_raw=True,
                        enable_deformation=True,
                        enable_depth=True,
                        enable_shear=True,
                        enable_force=True,
                    )
                )
                self.sensors[name] = sensor
                thread = threading.Thread(target=self._read, args=(name, sensor), daemon=True)
                thread.start()
                self.threads.append(thread)
        except BaseException:
            self.stop()
            raise

    def _read(self, name, sensor):
        fid = -1
        try:
            while not self.stop_event.is_set():
                if sensor.getDevStatus() != 0:
                    self.stop_event.wait(0.01)
                    continue
                if not sensor.wait_for_new(fid, timeout_ms=100):
                    continue
                image_id, image = sensor.getInferImg()
                image = getattr(image, "img", None)
                if image is None:
                    image_id, image = sensor.getRawImg()
                values = {
                    "deformation": sensor.getDeformation2D(),
                    "shear": sensor.getShear(),
                    "depth": sensor.getDepth(),
                    "force": sensor.getForce(),
                }
                fid = values["depth"][0]
                if image is None or image_id != fid:
                    continue
                if any(value is None or frame_id != fid for frame_id, value in values.values()):
                    continue
                sample = {key: np.asarray(value).copy() for key, (_, value) in values.items()}
                sample.update(img=np.asarray(image).copy(), frame_id=int(fid), timestamp=time.monotonic())
                with self.lock:
                    self.samples[name] = sample
        except Exception as exc:
            with self.lock:
                self.errors[name] = str(exc)
        finally:
            sensor.disconnect()

    def get_latest_data(self):
        with self.lock:
            if self.errors:
                raise RuntimeError(f"Tactile acquisition failed: {self.errors}")
            return dict(self.samples)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2)
        self.threads.clear()
