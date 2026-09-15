"""Write an mp4 from a stream of RGB frames, through whatever is installed.

The writer is chosen at runtime from what is actually present:

    imageio-ffmpeg        preferred
    cv2                   next
    lerobot_tactile_ext   last, and only when PyAV is installed alongside it

`Mp4Writer` exposes construct(path, fps), `add(frame)`, `close()` returning
a dict with a `frames` count.

Not a general video library: 8-bit RGB, constant frame rate, one stream.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = ["Mp4Writer", "compose_tactile_frame", "writer_backend"]


def _backend() -> str:
    """Which writer this machine can use.

    Each candidate is probed by importing the module that does the encoding;
    for lerobot that is PyAV, since its codec imports fine without it.
    """
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
        return "imageio"
    except Exception:
        pass
    try:
        import cv2  # noqa: F401
        return "cv2"
    except Exception:
        pass
    try:
        import av  # noqa: F401  -- what lerobot's codec actually needs

        from lerobot_tactile_ext.codec import Mp4Stream  # noqa: F401
        return "lerobot"
    except Exception:
        pass
    raise RuntimeError(
        "no mp4 writer available: install imageio-ffmpeg, or opencv-python, "
        "or make lerobot_tactile_ext importable together with PyAV")


def writer_backend() -> str:
    """Name of the backend that would be used, for logging before a long run."""
    return _backend()


class Mp4Writer:
    """Append RGB frames, get an mp4.

    Parameters
    ----------
    path
        Destination. Parent directories are created.
    fps
        Constant frame rate written into the container.
    quality
        Passed through to ffmpeg as a CRF-like knob where the backend supports
        it. The default is high so millimetre-scale contact stays visible.
    """

    def __init__(self, path: str | os.PathLike, fps: int = 30,
                 quality: int = 8):
        self.path = Path(os.path.expanduser(str(path)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.frames = 0
        self._backend = _backend()
        self._impl = None
        self._quality = quality
        self._size: tuple[int, int] | None = None

    # -- lazily opened, because ffmpeg wants the frame size up front --------
    def _open(self, h: int, w: int) -> None:
        self._size = (h, w)
        if self._backend == "lerobot":
            from lerobot_tactile_ext.codec import Mp4Stream
            self._impl = Mp4Stream(self.path, self.fps)
        elif self._backend == "imageio":
            import imageio
            # `macro_block_size=1` stops imageio silently resizing frames to a
            # multiple of 16, which would move every pixel coordinate a reader
            # might measure against.
            self._impl = imageio.get_writer(
                str(self.path), fps=self.fps, codec="libx264",
                quality=self._quality, macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
        else:
            import cv2
            self._impl = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, (w, h))

    def add(self, frame) -> None:
        """One frame, (H, W, 3) uint8 RGB. Anything else is coerced."""
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        h, w = arr.shape[:2]
        if self._impl is None:
            self._open(h, w)
        elif self._size != (h, w):
            raise ValueError(
                f"frame size changed mid-video: {self._size} -> {(h, w)}")

        if self._backend == "lerobot":
            self._impl.add(arr)
        elif self._backend == "imageio":
            self._impl.append_data(arr)
        else:
            self._impl.write(arr[..., ::-1])      # cv2 wants BGR
        self.frames += 1

    def close(self) -> dict:
        if self._impl is not None:
            if self._backend == "lerobot":
                self._impl.close()
            elif self._backend == "imageio":
                self._impl.close()
            else:
                self._impl.release()
        return {"frames": self.frames, "path": str(self.path),
                "backend": self._backend, "fps": self.fps}


# ---------------------------------------------------------------------------
# Composite frame: third-person view + the four tactile streams, synchronised
# ---------------------------------------------------------------------------
def _put_text(img: np.ndarray, text: str, xy: tuple[int, int],
              scale: float = 0.55, color=(255, 255, 255)) -> None:
    """Best-effort label: cv2 if present, PIL if present, otherwise nothing."""
    try:
        import cv2
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                    cv2.LINE_AA)
        return
    except Exception:
        pass
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((xy[0], xy[1] - 12), text, fill=tuple(color))
        img[:] = np.asarray(im)
    except Exception:
        pass


def compose_tactile_frame(scene: np.ndarray, pads: list, caption: str = "") -> np.ndarray:
    """One review frame: the scene camera on the left, the four tactile RGB
    renders in a 2x2 grid on the right, each labelled with its pad name, the
    deepest penetration (mm) and the contact area (mm^2) at that instant.

    `pads` is a list of (label, rgb_u8 HxWx3, max_depth_mm, area_mm2
    [, shear [, axial]]) in tile order: right-middle, right-index,
    left-middle, left-index. `shear` is the summed tangential force magnitude
    in TacSL's units (x1e3, the dataset's `ff_shear` scale); an axial load on
    a held object shows in shear even when depth and area do not change.
    Each tile is resized to 384x288 by nearest-neighbour.
    """
    scene = np.ascontiguousarray(scene[..., :3]).astype(np.uint8)
    sh, sw = scene.shape[:2]
    tile_w, tile_h, bar = 384, 288, 24
    grid_w, grid_h = 2 * tile_w, 2 * (tile_h + bar)
    H = max(sh, grid_h)
    canvas = np.zeros((H, sw + grid_w, 3), np.uint8)
    canvas[:sh, :sw] = scene
    if caption:
        _put_text(canvas, caption, (12, 28), 0.7)

    def fit(img: np.ndarray) -> np.ndarray:
        img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
        h, w = img.shape[:2]
        if (h, w) == (tile_h, tile_w):
            return img
        ys = (np.arange(tile_h) * h // tile_h).clip(0, h - 1)
        xs = (np.arange(tile_w) * w // tile_w).clip(0, w - 1)
        return img[ys][:, xs]

    for i, tile in enumerate(pads[:4]):
        label, rgb, depth_mm, area_mm2 = tile[:4]
        shear = tile[4] if len(tile) > 4 else 0.0
        axial = tile[5] if len(tile) > 5 else None
        r, c = divmod(i, 2)
        x0 = sw + c * tile_w
        y0 = r * (tile_h + bar)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = fit(rgb)
        # Header bar: dark, with the numbers that prove or disprove contact.
        canvas[y0 + tile_h:y0 + tile_h + bar, x0:x0 + tile_w] = (30, 30, 30)
        hot = depth_mm > 0.02
        # Fz is the pad's signed vertical resultant (x1e3), positive while the
        # gel carries the object's weight.
        # Units: d mm, A mm^2, S and Fz in thousands of the x1e3 scale.
        txt = f"{label} d {depth_mm:4.2f} A {area_mm2:5.1f} S {shear / 1e3:5.1f}k"
        if axial is not None:
            txt += f" Fz {axial / 1e3:+5.1f}k"
        _put_text(canvas, txt, (x0 + 6, y0 + tile_h + 17), 0.46,
                  (80, 255, 120) if hot else (180, 180, 180))
        if hot:
            # Thin green frame on a loaded pad, so contact is visible at a
            # glance even when the Taxim render itself is subtle.
            canvas[y0:y0 + 3, x0:x0 + tile_w] = (80, 255, 120)
            canvas[y0 + tile_h - 3:y0 + tile_h, x0:x0 + tile_w] = (80, 255, 120)
            canvas[y0:y0 + tile_h, x0:x0 + 3] = (80, 255, 120)
            canvas[y0:y0 + tile_h, x0 + tile_w - 3:x0 + tile_w] = (80, 255, 120)
    return canvas
