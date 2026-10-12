"""Physical tactile fields and their LeRobot dataset schema."""

import numpy as np

from tactile_codec import PROFILES

TACTILE_H, TACTILE_W = 288, 384


def dataset_features(sensor_names, previews=False, height=TACTILE_H, width=TACTILE_W):
    features = {}
    for name in sensor_names:
        features[f"observation.images.{name}_tactile"] = {
            "dtype": "video",
            "shape": (height, width, 3),
        }
        for field, channels in (("depth", 1), ("deformation", 2), ("shear", 2)):
            features[f"observation.{field}.{name}_tactile"] = {
                "dtype": "data",
                "shape": (height, width, channels),
                "profile": PROFILES[f"daimon_v1_{field}"],
            }
            if previews:
                features[f"observation.images.{field}_{name}"] = {
                    "dtype": "video",
                    "shape": (height, width, 3),
                }
        features[f"observation.state.force_{name}"] = {"dtype": "float32", "shape": (6,)}
        features[f"observation.tactile_status.{name}"] = {
            "dtype": "float32",
            "shape": (3,),
            "names": ["valid", "age_s", "frame_id"],
        }
    return features


def frame_from_sensor(name, sample, *, previews=False, height=TACTILE_H, width=TACTILE_W):
    frame = {}
    img = np.asarray(sample["img"])
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.shape == (height, width, 1):
        img = np.repeat(img, 3, axis=-1)
    if img.shape != (height, width, 3) or img.dtype != np.uint8:
        raise ValueError(f"Invalid tactile image for {name}: {img.shape}, {img.dtype}")
    frame[f"observation.images.{name}_tactile"] = img
    for field, channels in (("depth", 1), ("deformation", 2), ("shear", 2)):
        data = np.asarray(sample[field], dtype=np.float32)
        if channels == 1 and data.shape == (height, width):
            data = data[..., None]
        if data.shape != (height, width, channels) or not np.isfinite(data).all():
            raise ValueError(f"Invalid {field} for {name}")
        frame[f"observation.{field}.{name}_tactile"] = data
        if previews:
            import cv2

            if channels == 1:
                gray = np.clip(data[..., 0] * 255 / 4, 0, 255).astype(np.uint8)
                rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
            else:
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(8, height, 16):
                    for x in range(8, width, 16):
                        dx, dy = np.clip(data[y, x] * 20, -100, 100).astype(int)
                        cv2.arrowedLine(rgb, (x, y), (x + int(dx), y + int(dy)), (0, 255, 0), 1)
            frame[f"observation.images.{field}_{name}"] = rgb
    force = np.asarray(sample["force"], dtype=np.float32).reshape(-1)
    if force.shape != (6,) or not np.isfinite(force).all():
        raise ValueError(f"Invalid force for {name}")
    frame[f"observation.state.force_{name}"] = force
    return frame
