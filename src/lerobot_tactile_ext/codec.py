"""Lossless 16-bit tactile fields <-> ffv1/mkv, both directions.

LeRobot's `data` dtype stores per-frame uint16 fields (tactile depth, shear,
deformation) as ffv1-compressed mkv streams. This module is the codec:

* encode: (N,H,W[,C]) uint16 arrays -> mkv. 1-channel goes out `gray16le`;
  2-channel is padded to 3 planes (on-disk contract: shape (H,W,3), third
  plane zero, `original_channels = 2` in the feature info); 3-channel goes
  out `gbrp16le`.
* decode: mkv -> uint16 arrays, cut back to `original_channels` when given.
* `Mp4Stream` / `DataStream`: incremental encoders held open across
  episodes, for the v3 layout where every episode of a feature shares one
  file addressed by timestamps.

The pair is held to bit-exactness by `tactile_codec_roundtrip`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["DataStream", "Mp4Stream", "decode_data_frames", "encode_data_frames"]


def _require_av():
    try:
        import av
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "lerobot_tactile_ext needs PyAV (`pip install av`); it ships with "
            "lerobot 0.5 as a dependency") from e
    return av


def _normalise(frames: np.ndarray) -> tuple[np.ndarray, bool, int]:
    """-> (frames as (N,H,W) or (N,H,W,3) uint16, single_channel, original_channels)."""
    a = np.asarray(frames)
    if a.dtype != np.uint16:
        raise TypeError(f"data dtype frames must be uint16, got {a.dtype}; "
                        "quantize first (tactile_codec.quantize)")
    if a.ndim == 3:                                   # (N, H, W)
        return a, True, 1
    if a.ndim != 4:
        raise ValueError(f"expected (N,H,W[,C]) frames, got shape {a.shape}")
    c = a.shape[-1]
    if c == 1:
        return a[..., 0], True, 1
    if c == 2:                                        # pad to the on-disk 3ch contract
        pad = np.zeros((*a.shape[:-1], 3), dtype=np.uint16)
        pad[..., :2] = a
        return pad, False, 2
    if c == 3:
        return a, False, 3
    raise ValueError(f"unsupported channel count {c}")


def encode_data_frames(frames: np.ndarray, video_path: Path | str, fps: int,
                       overwrite: bool = True) -> dict:
    """uint16 frames -> ffv1/mkv. Returns encoding stats for the report.

    Stats include `bytes` and `compression_ratio` (raw/encoded): a ratio
    near 1.0 is an early smell that the data is noise.
    """
    av = _require_av()
    video_path = Path(video_path)
    if video_path.exists() and not overwrite:
        raise FileExistsError(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    arr, single, orig_c = _normalise(frames)
    n = arr.shape[0]
    h, w = arr.shape[1:3]
    pix_fmt = "gray16le" if single else "gbrp16le"

    with av.open(str(video_path), "w") as out:
        stream = out.add_stream("ffv1", fps)
        stream.pix_fmt = pix_fmt
        stream.width = w
        stream.height = h
        for i in range(n):
            if single:
                frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(arr[i]), format="gray16le")
            else:
                frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(arr[i]), format="gbrp16le")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():                # flush
            out.mux(packet)

    raw_bytes = int(arr.nbytes)
    enc_bytes = video_path.stat().st_size
    return {
        "frames": n, "height": h, "width": w,
        "pix_fmt": pix_fmt, "original_channels": orig_c,
        "raw_bytes": raw_bytes, "bytes": enc_bytes,
        "compression_ratio": raw_bytes / max(enc_bytes, 1),
    }


class _Stream:
    """Incremental encoder: one av container held open across episodes.

    v3 aggregates every episode of a feature into the *same* file
    (``videos/{key}/chunk-000/file-000.*``), addressed by per-episode
    ``from_timestamp``/``to_timestamp`` in the episodes metadata. Batch
    encoding would mean holding all episodes in memory (~GBs of uint16
    fields); this streams instead and only `close()` finalises the file.
    """

    def __init__(self, path: Path | str, fps: int, codec: str, pix_fmt: str,
                 options: dict | None = None):
        av = _require_av()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), "w")
        self._stream = self._container.add_stream(codec, fps)
        self._stream.pix_fmt = pix_fmt
        if options:
            self._stream.options = options
        self._av = av
        self.frames = 0
        self.raw_bytes = 0

    def _push(self, arr: np.ndarray, fmt: str) -> None:
        if self.frames == 0:
            self._stream.width = arr.shape[1]
            self._stream.height = arr.shape[0]
        frame = self._av.VideoFrame.from_ndarray(
            np.ascontiguousarray(arr), format=fmt)
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames += 1
        self.raw_bytes += arr.nbytes

    def close(self) -> dict:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        enc = self.path.stat().st_size
        return {"frames": self.frames, "bytes": enc,
                "compression_ratio": self.raw_bytes / max(enc, 1)}


class Mp4Stream(_Stream):
    """uint8 RGB -> h264/mp4, appendable across episodes."""

    def __init__(self, path: Path | str, fps: int):
        super().__init__(path, fps, "libx264", "yuv420p",
                         {"crf": "18", "preset": "medium"})

    def add(self, frame: np.ndarray) -> None:
        self._push(frame.astype(np.uint8), "rgb24")


class DataStream(_Stream):
    """uint16 fields -> ffv1/mkv, appendable across episodes.

    Same channel contract as `encode_data_frames`: (H,W) or (H,W,1) go out
    gray16le; (H,W,2) is padded to 3 planes; (H,W,3) goes out gbrp16le.
    """

    def __init__(self, path: Path | str, fps: int, channels: int):
        single = channels == 1
        super().__init__(path, fps, "ffv1",
                         "gray16le" if single else "gbrp16le")
        self._single = single

    def add(self, frame: np.ndarray) -> None:
        a = np.asarray(frame)
        if a.dtype != np.uint16:
            raise TypeError(f"data frames must be uint16, got {a.dtype}")
        if self._single:
            self._push(a[..., 0] if a.ndim == 3 else a, "gray16le")
            return
        if a.shape[-1] == 2:
            pad = np.zeros((*a.shape[:-1], 3), dtype=np.uint16)
            pad[..., :2] = a
            a = pad
        self._push(a, "gbrp16le")


def decode_data_frames(video_path: Path | str,
                       original_channels: int | None = None) -> np.ndarray:
    """ffv1/mkv -> uint16 frames.

    Returns (N,H,W) for gray16le streams, (N,H,W,C) otherwise, with C cut back
    to `original_channels` when given (recorded 2-channel data is stored as
    3 planes with the third identically zero).
    """
    av = _require_av()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        single = stream.codec_context.pix_fmt == "gray16le"
        fmt = "gray16le" if single else "gbrp16le"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format=fmt))
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")

    out = np.stack(frames)
    if out.dtype != np.uint16:
        # some av builds hand 16-bit planes back as little-endian views
        out = out.astype(np.uint16)
    if out.ndim == 4 and original_channels is not None:
        out = out[..., :original_channels]
    return out
