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

Usually run through `scripts/run_convert.sh`; directly:
    python benchmark/policy/convert_dataset.py --src <raw_dataset_dir> \
        --dst-root <datasets_dir> --variant tacff
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_tactile_ext.codec import encode_data_frames
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


def decode_mp4(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24"))
    return np.stack(frames)


def _gbrp_plane_order() -> list[int]:
    """Plane indices that put the ffmpeg CLI's gbrp16le output back into the
    channel order the frames were written in, found on a probe frame whose
    three channels differ."""
    probe = np.zeros((1, 2, 2, 3), np.uint16)
    probe[..., 0], probe[..., 1], probe[..., 2] = 1000, 2000, 3000
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "probe.mkv"
        encode_data_frames(probe, p, fps=FPS)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p),
             "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
            stdout=subprocess.PIPE, check=True).stdout
    planes = np.frombuffer(raw, np.uint16).reshape(3, 2, 2)[:, 0, 0]
    return [int(np.flatnonzero(planes == v)[0]) for v in (1000, 2000, 3000)]


def fast_decode_mkv(path: Path, original_channels: int) -> np.ndarray:
    """ffv1/mkv -> uint16 frames via the ffmpeg CLI (faster than per-frame PyAV),
    channels in the order they were written."""
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        h, w = st.codec_context.height, st.codec_context.width
        single = st.codec_context.pix_fmt == "gray16le"

    fmt = "gray16le" if single else "gbrp16le"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", fmt, "-"],
        stdout=subprocess.PIPE, check=True).stdout

    if single:
        # (N,H,W)
        return np.frombuffer(raw, np.uint16).reshape(-1, h, w)
    planes = np.frombuffer(raw, np.uint16).reshape(-1, 3, h, w)
    out = planes.transpose(0, 2, 3, 1)[..., _gbrp_plane_order()]
    return out[..., :original_channels]


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
                    help="output name stem (default: the recording's directory name)")
    ap.add_argument("--variant", choices=("vision", "tacff", "tacrgb", "both"),
                    default="both",
                    help="'tacrgb' = vision + raw tactile RGB; 'both' = vision + tacff")
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

    src = Path(args.src)
    src_root = src
    name = args.name or src.name.rstrip("/")
    table = pq.read_table(src / "data" / "chunk-000" / "file-000.parquet")
    state = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
    action = np.stack(table["action"].to_pylist()).astype(np.float32)
    ep_idx = np.asarray(table["episode_index"])
    success = np.asarray(table["is_success"])
    # The instruction each frame was recorded under, as the collector wrote it.
    task_idx = np.asarray(table["task_index"])
    tasks = pq.read_table(src / "meta" / "tasks.parquet")
    task_text = dict(zip(tasks["task_index"].to_pylist(), tasks["task"].to_pylist(),
                         strict=True))

    def stream(key: str) -> Path:
        return src / "videos" / key / "chunk-000"

    variants = (("vision", "tacff") if args.variant == "both"
                else (args.variant,))

    print("decoding video streams ...", flush=True)
    vids = {k: decode_mp4(stream(k) / "file-000.mp4") for k in VISION_KEYS}
    if "tacrgb" in variants:
        for pad in PADS:
            k = f"observation.images.{pad}_tactile"
            vids[k] = decode_mp4(stream(k) / "file-000.mp4")
            print(f"  {pad} tactile RGB decoded {vids[k].shape}", flush=True)

    tacff = {}
    if "tacff" in variants:
        print("decoding force-field streams ...", flush=True)
    for pad in (PADS if "tacff" in variants else ()):
        d_u16 = fast_decode_mkv(
            stream(f"observation.depth.{pad}_tactile") / "file-000.mkv",
            original_channels=1)
        print(f"  {pad} depth decoded {d_u16.shape}", flush=True)
        s_u16 = fast_decode_mkv(
            stream(f"observation.ff_shear.{pad}_tactile") / "file-000.mkv",
            original_channels=2)
        print(f"  {pad} ff_shear decoded {s_u16.shape}", flush=True)
        # raw==0 is quantised NaN (shear is NaN off-contact). Map it to the
        # offset (= shear 0), the same mapping the evaluator applies.
        nan_px = int((s_u16 == 0).sum())
        if nan_px:
            print(f"  {pad} ff_shear: {nan_px} quantised-NaN px -> offset",
                  flush=True)
            s_u16 = np.where(s_u16 == 0, 32768, s_u16)
        depth = dequantize(d_u16, PROFILES["sim_v1_depth"])
        shear = dequantize(s_u16, PROFILES["sim_v1_shear"])
        if depth.ndim == 4:          # (N,H,W,1) -> (N,H,W)
            depth = depth[..., 0]
        tacff[pad] = (depth, shear)

    n = len(state)
    for k, v in vids.items():
        assert len(v) == n, f"{k}: {len(v)} frames vs {n} parquet rows"
    for pad, (d, s) in tacff.items():
        assert len(d) == n and len(s) == n, f"{pad}: stream length mismatch"

    keep_eps = sorted({int(e) for e, ok in zip(ep_idx, success, strict=True) if ok})
    dropped = sorted({int(e) for e in ep_idx} - set(keep_eps))
    print(f"episodes: keep {len(keep_eps)}, drop (failed) {dropped}", flush=True)

    for variant in variants:
        dst = Path(args.dst_root) / f"{name}_{variant}"
        if dst.exists():
            if not args.overwrite:
                raise SystemExit(f"{dst} exists; pass --overwrite")
            shutil.rmtree(dst)

        feats = {
            "observation.state": {"dtype": "float32", "shape": (16,),
                                  "names": [f"j{i}" for i in range(16)]},
            "action": {"dtype": "float32", "shape": (16,),
                       "names": [f"j{i}" for i in range(16)]},
        }
        img_keys = list(VISION_KEYS)
        if variant == "tacff":
            img_keys += [f"observation.images.tacff_{_short(p)}"
                         for p in PADS]   # tacff_right_middle, tacff_left_index, ...
        elif variant == "tacrgb":
            # raw Taxim renders, re-keyed tacrgb_middle / tacrgb_index
            img_keys += [f"observation.images.tacrgb_{_short(p)}"
                         for p in PADS]
        def src_frames(key: str):
            """Frame source for an output image key."""
            if key in vids:
                return vids[key]
            if key.startswith("observation.images.tacrgb_"):
                short = key[len("observation.images.tacrgb_"):]
                pad = next(p for p in PADS if _short(p) == short)
                return vids[f"observation.images.{pad}_tactile"]
            return None                      # tacff: synthesised per-frame

        for k in img_keys:
            src = src_frames(k)
            h, w = (src.shape[1:3] if src is not None
                    else tacff[PADS[0]][0].shape[1:3])
            feats[k] = {"dtype": "video", "shape": (h, w, 3),
                        "names": ["height", "width", "channels"]}

        ds = LeRobotDataset.create(
            repo_id=f"local/{name}_{variant}", fps=FPS, features=feats,
            root=dst, robot_type="openarm_bimanual_tactile",
            vcodec=args.vcodec)

        for ep in keep_eps:
            sel = np.where(ep_idx == ep)[0]
            for i in sel:
                frame = {"observation.state": state[i], "action": action[i],
                         "task": task_text[int(task_idx[i])]}
                for k in VISION_KEYS:
                    frame[k] = vids[k][i]
                if variant == "tacff":
                    for pad in PADS:
                        d, s = tacff[pad]
                        key = f"observation.images.tacff_{_short(pad)}"
                        frame[key] = tacff_image(d[i], s[i])
                elif variant == "tacrgb":
                    for pad in PADS:
                        key = f"observation.images.tacrgb_{_short(pad)}"
                        frame[key] = vids[
                            f"observation.images.{pad}_tactile"][i]
                ds.add_frame(frame)
            ds.save_episode()
            print(f"[{variant}] episode {ep} saved ({len(sel)} frames)",
                  flush=True)
        copy_review_videos(src_root, dst, keep_eps, variant)
        print(f"[{variant}] done -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
