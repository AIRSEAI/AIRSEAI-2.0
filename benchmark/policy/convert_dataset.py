#!/usr/bin/env python3
"""Convert a raw tactile recording into standard LeRobot v3 datasets, one per
input variant.

Reads the dataset written by `scripts/run_collect.sh` and writes
<dst-root>/<name>_<variant> through `LeRobotDataset.create`, keeping only
successful episodes. Each kept episode's third-person review video, when the
recording has one, is copied to <dst>/review/third_person/ under its new
episode index; it is for looking at, not a dataset feature, so no policy
trains on it. Variants: `vision` (cameras, state, action), `tacff`
(+ each pad's force field as a 3-channel image: R = depth, G/B = shear x/y),
`tacrgb` (+ the raw tactile RGB renders); `both` = vision + tacff.

Each stream is decoded a frame at a time and written straight out, so memory
stays flat however long the recording is.

Usually run through `scripts/run_convert.sh`; directly:
    python benchmark/policy/convert_dataset.py --src <raw_dataset_dir> \
        --dst-root <datasets_dir> --variant tacff
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tactile_codec import PROFILES, dequantize
from tactile_codec.tacff import tacff_image

FPS = 20
# Review videos beside the dataset, not inside its features.
REVIEW_DIR = Path("review") / "third_person"
# Filled in main(): both arms' pad feature names and the camera keys.
PADS: tuple = ()
VISION_KEYS: tuple = ()


def _short(pad: str) -> str:
    """'right_hand_middle_finger' -> 'right_middle': the output key stem."""
    parts = pad.split("_")
    return f"{parts[0]}_{parts[2]}"


def video_shape(path: Path) -> tuple[int, int]:
    """(height, width) of a stream, read from its header."""
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        return st.codec_context.height, st.codec_context.width


def mp4_frames(path: Path):
    """Yield each frame of an mp4 as an (H, W, 3) uint8 array."""
    import av

    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            yield f.to_ndarray(format="rgb24")


def _plane_order(frame: np.ndarray, ref: np.ndarray) -> list[int]:
    """The permutation carrying the ffmpeg CLI's planes onto PyAV's."""
    for perm in itertools.permutations(range(3)):
        if np.array_equal(frame[..., list(perm)], ref):
            return list(perm)
    raise RuntimeError("cannot align ffmpeg/PyAV plane order")


def mkv_u16_frames(path: Path, original_channels: int):
    """Yield each frame of an ffv1/mkv data stream as uint16.

    Single-channel frames come out (H, W), the rest (H, W, C). Frames arrive
    from the ffmpeg CLI, which is faster than decoding them one by one with
    PyAV; the plane order of its gbrp16le output is calibrated against a PyAV
    decode of the first frame, so the values match a PyAV decode.
    """
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        h, w = st.codec_context.height, st.codec_context.width
        single = st.codec_context.pix_fmt == "gray16le"
    fmt = "gray16le" if single else "gbrp16le"
    planes = 1 if single else 3
    nbytes = h * w * planes * 2

    ref, perm = None, None
    if not single:
        with av.open(str(path)) as c:
            ref = next(c.decode(c.streams.video[0])).to_ndarray(format="gbrp16le")
        ref = np.asarray(ref).astype(np.uint16)
        if ref.ndim == 3 and ref.shape[0] == 3:           # (3,H,W) planar
            ref = ref.transpose(1, 2, 0)

    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", fmt, "-"],
        stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            raw = np.frombuffer(buf, np.uint16)
            if single:
                yield raw.reshape(h, w)
                continue
            frame = raw.reshape(planes, h, w).transpose(1, 2, 0)
            if perm is None:
                perm = _plane_order(frame, ref)
            yield frame[..., perm][..., :original_channels]
    finally:
        proc.stdout.close()
        proc.wait()


def copy_review_videos(src_root: Path, dst: Path, keep_eps: list[int],
                       variant: str) -> None:
    """Copy each kept episode's third-person review mp4, renumbered to the
    converted dataset's episode index (kept episodes are saved in order)."""
    have = src_root / REVIEW_DIR
    if not have.is_dir():
        print(f"[{variant}] no review videos in the recording", flush=True)
        return
    out = dst / REVIEW_DIR
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for new_idx, ep in enumerate(keep_eps):
        f = have / f"episode_{ep:06d}.mp4"
        if f.exists():
            shutil.copy2(f, out / f"episode_{new_idx:06d}.mp4")
        else:
            missing.append(ep)
    print(f"[{variant}] review videos: {len(keep_eps) - len(missing)} -> {out}"
          + (f", missing for episodes {missing}" if missing else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw recording root")
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--name", default="",
                    help="output name stem, default the recording's dir name")
    ap.add_argument("--variant", choices=("vision", "tacff", "tacrgb", "both"),
                    default="both",
                    help="tacrgb = vision + tactile RGB; both = vision + tacff")
    ap.add_argument("--vcodec", default="h264",
                    help="video codec: h264, libsvtav1 or h264_nvenc")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Symmetric schema: both arms' pads and both wrist cameras, always, so the
    # keys are the same whichever arm an episode uses.
    global PADS, VISION_KEYS
    PADS = tuple(f"{arm}_hand_{finger}_finger"
                 for arm in ("right", "left") for finger in ("middle", "index"))
    VISION_KEYS = ("observation.images.base",
                   "observation.images.right_wrist",
                   "observation.images.left_wrist")

    src_root = Path(args.src)
    name = args.name or src_root.name.rstrip("/")
    table = pq.read_table(src_root / "data" / "chunk-000" / "file-000.parquet")
    state = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
    action = np.stack(table["action"].to_pylist()).astype(np.float32)
    ep_idx = np.asarray(table["episode_index"])
    success = np.asarray(table["is_success"])
    # The instruction each frame was recorded under, as the collector wrote it.
    task_idx = np.asarray(table["task_index"])
    tasks = pq.read_table(src_root / "meta" / "tasks.parquet")
    task_text = dict(zip(tasks["task_index"].to_pylist(), tasks["task"].to_pylist(),
                         strict=True))

    def stream(key: str) -> Path:
        return src_root / "videos" / key / "chunk-000"

    def mp4(key: str) -> Path:
        return stream(key) / "file-000.mp4"

    variants = (("vision", "tacff") if args.variant == "both"
                else (args.variant,))

    n = len(state)
    keep_eps = sorted({int(e) for e, ok in zip(ep_idx, success, strict=True) if ok})
    dropped = sorted({int(e) for e in ep_idx} - set(keep_eps))
    keep = set(keep_eps)
    print(f"frames {n}; episodes: keep {len(keep_eps)}, drop (failed) {dropped}",
          flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for variant in variants:
        dst = Path(args.dst_root) / f"{name}_{variant}"
        if dst.exists():
            if not args.overwrite:
                raise SystemExit(f"{dst} exists; pass --overwrite")
            shutil.rmtree(dst)

        # The pads this variant carries, and under which key stem.
        pad_key = {"tacff": "tacff", "tacrgb": "tacrgb"}.get(variant)
        pads = PADS if pad_key else ()

        feats = {
            "observation.state": {"dtype": "float32", "shape": (16,),
                                  "names": [f"j{i}" for i in range(16)]},
            "action": {"dtype": "float32", "shape": (16,),
                       "names": [f"j{i}" for i in range(16)]},
        }
        for k in VISION_KEYS:
            h, w = video_shape(mp4(k))
            feats[k] = {"dtype": "video", "shape": (h, w, 3),
                        "names": ["height", "width", "channels"]}
        for pad in pads:
            h, w = video_shape(
                mp4(f"observation.images.{pad}_tactile") if variant == "tacrgb"
                else stream(f"observation.depth.{pad}_tactile") / "file-000.mkv")
            feats[f"observation.images.{pad_key}_{_short(pad)}"] = {
                "dtype": "video", "shape": (h, w, 3),
                "names": ["height", "width", "channels"]}

        ds = LeRobotDataset.create(
            repo_id=f"local/{name}_{variant}", fps=FPS, features=feats,
            root=dst, robot_type="openarm_bimanual_tactile",
            vcodec=args.vcodec)

        # One decoder per stream, read in step with the parquet rows: a frame
        # of a dropped episode is still pulled, so the streams stay aligned.
        reads = {k: mp4_frames(mp4(k)) for k in VISION_KEYS}
        if variant == "tacrgb":
            for pad in pads:
                reads[f"{pad_key}_{_short(pad)}"] = mp4_frames(
                    mp4(f"observation.images.{pad}_tactile"))
        elif variant == "tacff":
            for pad in pads:
                reads[f"depth_{_short(pad)}"] = mkv_u16_frames(
                    stream(f"observation.depth.{pad}_tactile") / "file-000.mkv", 1)
                reads[f"shear_{_short(pad)}"] = mkv_u16_frames(
                    stream(f"observation.ff_shear.{pad}_tactile") / "file-000.mkv", 2)

        nan_px = 0
        for i in range(n):
            try:
                frames = {k: next(it) for k, it in reads.items()}
            except StopIteration:
                raise SystemExit(f"stream ended at frame {i} of {n}") from None
            ep = int(ep_idx[i])
            last_of_ep = i + 1 == n or int(ep_idx[i + 1]) != ep
            if ep in keep:
                frame = {"observation.state": state[i], "action": action[i],
                         "task": task_text[int(task_idx[i])]}
                for k in VISION_KEYS:
                    frame[k] = frames[k]
                for pad in pads:
                    key = f"observation.images.{pad_key}_{_short(pad)}"
                    if variant == "tacrgb":
                        frame[key] = frames[f"{pad_key}_{_short(pad)}"]
                        continue
                    # raw == 0 is quantised NaN (shear is NaN off-contact).
                    # Map it to the offset (= shear 0), as the evaluator does.
                    s_u16 = frames[f"shear_{_short(pad)}"]
                    nan_px += int((s_u16 == 0).sum())
                    frame[key] = tacff_image(
                        dequantize(frames[f"depth_{_short(pad)}"],
                                   PROFILES["sim_v1_depth"]),
                        dequantize(np.where(s_u16 == 0, 32768, s_u16),
                                   PROFILES["sim_v1_shear"]))
                ds.add_frame(frame)
                if last_of_ep:
                    ds.save_episode()
                    print(f"[{variant}] episode {ep} saved", flush=True)
        for k, it in reads.items():
            if next(it, None) is not None:
                raise SystemExit(f"{k}: more frames than the {n} parquet rows")
        if nan_px:
            print(f"[{variant}] ff_shear: {nan_px} quantised-NaN px -> offset",
                  flush=True)
        copy_review_videos(src_root, dst, keep_eps, variant)
        print(f"[{variant}] done -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
