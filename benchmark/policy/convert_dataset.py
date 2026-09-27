#!/usr/bin/env python3
"""Convert successful raw episodes into vision, tacrgb or tacff LeRobot datasets.

Preserve episode metadata and copy review videos outside policy features.

Each stream is decoded a frame at a time and written straight out, so memory
stays flat however long the recording is.

Usually run through `scripts/run_convert.sh`; directly:
    python benchmark/policy/convert_dataset.py --src <raw_dataset_dir> \
        --dst-root <datasets_dir> --variant tacff
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot_tactile_ext.merge import inspect_recording, merge_recordings
from tactile_codec import QuantProfile, dequantize
from tactile_codec.tacff import tacff_image

# Review videos beside the dataset, not inside its features.
REVIEW_DIR = Path("review") / "third_person"
PADS = tuple(f"{arm}_hand_{finger}_finger" for arm in ("right", "left") for finger in ("middle", "index"))
VISION_KEYS = (
    "observation.images.base",
    "observation.images.right_wrist",
    "observation.images.left_wrist",
)


def _short(pad: str) -> str:
    """'right_hand_middle_finger' -> 'right_middle': the output key stem."""
    parts = pad.split("_")
    return f"{parts[0]}_{parts[2]}"


def decode_mp4(path: Path, frames: int, output: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        height, width = stream.codec_context.height, stream.codec_context.width
        decoded = np.lib.format.open_memmap(
            output, mode="w+", dtype=np.uint8, shape=(frames, height, width, 3)
        )
        count = 0
        for count, frame in enumerate(c.decode(stream), start=1):
            if count > frames:
                raise ValueError(f"{path}: more than {frames} frames")
            decoded[count - 1] = frame.to_ndarray(format="rgb24")
    if count != frames:
        raise ValueError(f"{path}: {count} frames vs {frames} parquet rows")
    decoded.flush()
    return decoded


class PlanarFrames:
    def __init__(self, planes, order, channels):
        self.planes = planes
        self.order = order[:channels]
        self.shape = (len(planes), planes.shape[2], planes.shape[3], channels)

    def __len__(self):
        return len(self.planes)

    def __getitem__(self, index):
        return np.asarray(self.planes[index, list(self.order)]).transpose(1, 2, 0)


def fast_decode_mkv(path: Path, original_channels: int, frames: int, output: Path):
    """Decode uint16 frames to disk and align channel order with PyAV."""

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        h, w = st.codec_context.height, st.codec_context.width
        single = st.codec_context.pix_fmt == "gray16le"
    fmt = "gray16le" if single else "gbrp16le"
    with output.open("xb") as raw:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", fmt, "-"],
            stdout=raw,
            check=True,
        )

    if single:
        expected = frames * h * w * np.dtype(np.uint16).itemsize
        if output.stat().st_size != expected:
            raise ValueError(f"{path}: decoded byte count does not match {frames} frames")
        return np.memmap(output, mode="r", dtype=np.uint16, shape=(frames, h, w))
    expected = frames * 3 * h * w * np.dtype(np.uint16).itemsize
    if output.stat().st_size != expected:
        raise ValueError(f"{path}: decoded byte count does not match {frames} frames")
    planes = np.memmap(output, mode="r", dtype=np.uint16, shape=(frames, 3, h, w))

    # calibrate plane order against PyAV on the first frame
    with av.open(str(path)) as c:
        ref = None
        for f in c.decode(c.streams.video[0]):
            ref = f.to_ndarray(format="gbrp16le")
            break
    ref = np.asarray(ref).astype(np.uint16)
    if ref.ndim == 3 and ref.shape[0] == 3:  # (3,H,W) planar
        ref = ref.transpose(1, 2, 0)
    order = tuple(range(3))
    first = np.asarray(planes[0]).transpose(1, 2, 0)
    if not np.array_equal(first, ref):
        for perm in itertools.permutations(range(3)):
            if np.array_equal(first[..., list(perm)], ref):
                order = perm
                break
        else:
            raise RuntimeError(f"{path}: cannot align ffmpeg/PyAV plane order")
    return PlanarFrames(planes, order, original_channels)


def copy_review_videos(src_root: Path, dst: Path, keep_eps: list[int], variant: str) -> None:
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
    print(
        f"[{variant}] review videos: {len(keep_eps) - len(missing)} -> {out}"
        + (f", missing for episodes {missing}" if missing else ""),
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="raw recording roots")
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--name", default="", help="output name stem (default: the recording's directory name)")
    ap.add_argument(
        "--variant",
        choices=("vision", "tacff", "tacrgb", "both"),
        default="both",
        help="'tacrgb' = vision + raw tactile RGB; 'both' = vision + tacff",
    )
    ap.add_argument("--vcodec", default="h264", help="video codec: h264, libsvtav1 or h264_nvenc")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    roots = [Path(p).expanduser().resolve() for p in args.src]
    if len(roots) > 1 and not args.name:
        ap.error("--name is required for multiple inputs")
    name = args.name or roots[0].name
    if name in (".", "..") or Path(name).name != name:
        ap.error("--name must be a directory name")
    dst_root = Path(args.dst_root).expanduser().resolve()
    if any(dst_root == p or p in dst_root.parents for p in roots):
        ap.error("output directory must be outside input recordings")
    dst_root.mkdir(parents=True, exist_ok=True)
    merge_dir = tempfile.TemporaryDirectory(prefix=".raw-merge-", dir=dst_root)
    src = merge_recordings(roots, Path(merge_dir.name) / "raw") if len(roots) > 1 else roots[0]
    inspected = inspect_recording(src)
    annotations = {row["episode_index"]: row for row in inspected["annotations"]}
    src_root = src
    info = inspected["info"]
    fps = info["fps"]
    if isinstance(fps, float) and fps.is_integer():
        fps = int(fps)
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"LeRobot video output requires a positive integer FPS, got {fps!r}")
    source_map_path = src / "meta/source_episodes.json"
    source_map = (
        json.loads(source_map_path.read_text())
        if source_map_path.exists()
        else [
            dict(source="source_000", source_episode_index=i, source_sha256=inspected["digest"])
            for i in range(info["total_episodes"])
        ]
    )
    table = pq.read_table(src / "data" / "chunk-000" / "file-000.parquet")
    state = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
    action = np.stack(table["action"].to_pylist()).astype(np.float32)
    ep_idx = np.asarray(table["episode_index"])
    success = np.asarray(table["is_success"])
    # The instruction each frame was recorded under, as the collector wrote it.
    task_idx = np.asarray(table["task_index"])
    tasks = pq.read_table(src / "meta" / "tasks.parquet")
    task_text = dict(zip(tasks["task_index"].to_pylist(), tasks["task"].to_pylist(), strict=True))

    def stream(key: str) -> Path:
        return src_root / "videos" / key / "chunk-000"

    variants = ("vision", "tacff") if args.variant == "both" else (args.variant,)
    n = len(state)
    decode_root = Path(merge_dir.name) / "decoded"
    decode_root.mkdir()

    print("decoding video streams ...", flush=True)
    vids = {
        key: decode_mp4(stream(key) / "file-000.mp4", n, decode_root / f"vision_{index}.npy")
        for index, key in enumerate(VISION_KEYS)
    }
    if "tacrgb" in variants:
        for index, pad in enumerate(PADS):
            k = f"observation.images.{pad}_tactile"
            vids[k] = decode_mp4(stream(k) / "file-000.mp4", n, decode_root / f"tacrgb_{index}.npy")
            print(f"  {pad} tactile RGB decoded {vids[k].shape}", flush=True)

    tacff = {}
    if "tacff" in variants:
        print("decoding force-field streams ...", flush=True)
    for index, pad in enumerate(PADS if "tacff" in variants else ()):
        d_u16 = fast_decode_mkv(
            stream(f"observation.depth.{pad}_tactile") / "file-000.mkv",
            original_channels=1,
            frames=n,
            output=decode_root / f"depth_{index}.raw",
        )
        print(f"  {pad} depth decoded {d_u16.shape}", flush=True)
        s_u16 = fast_decode_mkv(
            stream(f"observation.ff_shear.{pad}_tactile") / "file-000.mkv",
            original_channels=2,
            frames=n,
            output=decode_root / f"shear_{index}.raw",
        )
        print(f"  {pad} ff_shear decoded {s_u16.shape}", flush=True)
        depth_profile = QuantProfile.from_dict(
            info["features"][f"observation.depth.{pad}_tactile"]["info"]["quant"]
        )
        shear_profile = QuantProfile.from_dict(
            info["features"][f"observation.ff_shear.{pad}_tactile"]["info"]["quant"]
        )
        tacff[pad] = (d_u16, s_u16, depth_profile, shear_profile)

    for k, v in vids.items():
        assert len(v) == n, f"{k}: {len(v)} frames vs {n} parquet rows"
    for pad, (d, s, _, _) in tacff.items():
        assert len(d) == n and len(s) == n, f"{pad}: stream length mismatch"

    keep_eps = sorted({int(e) for e, ok in zip(ep_idx, success, strict=True) if ok})
    dropped = sorted({int(e) for e in ep_idx} - set(keep_eps))
    print(f"frames {n}; episodes: keep {len(keep_eps)}, drop (failed) {dropped}", flush=True)

    if not keep_eps:
        raise ValueError("No accepted episodes to convert")

    for variant in variants:
        final_dst = dst_root / f"{name}_{variant}"
        if any(final_dst == p or final_dst in p.parents or p in final_dst.parents for p in roots):
            raise ValueError("Output and input directories must not overlap")
        conversion = dict(
            variant=variant,
            vcodec=args.vcodec,
            sources=source_map,
            source_sha256=inspected["digest"],
            format_version=1,
        )
        if final_dst.exists() and not args.overwrite:
            marker = final_dst / "meta/conversion.json"
            if marker.is_file() and json.loads(marker.read_text()) == conversion:
                meta = LeRobotDatasetMetadata(f"local/{name}_{variant}", root=final_dst)
                if meta.total_episodes == len(keep_eps):
                    print(f"[{variant}] verified existing output: {final_dst}", flush=True)
                    continue
            raise FileExistsError(f"{final_dst} is incomplete or differs; pass --overwrite")
        staging = tempfile.TemporaryDirectory(prefix=f".{name}_{variant}-", dir=dst_root)
        dst = Path(staging.name) / "dataset"

        feats = {
            "observation.state": {"dtype": "float32", "shape": (16,), "names": [f"j{i}" for i in range(16)]},
            "action": {"dtype": "float32", "shape": (16,), "names": [f"j{i}" for i in range(16)]},
        }
        img_keys = list(VISION_KEYS)
        if variant == "tacff":
            img_keys += [
                f"observation.images.tacff_{_short(p)}" for p in PADS
            ]  # tacff_right_middle, tacff_left_index, ...
        elif variant == "tacrgb":
            img_keys += [f"observation.images.tacrgb_{_short(p)}" for p in PADS]

        def src_frames(key: str):
            """Frame source for an output image key."""
            if key in vids:
                return vids[key]
            if key.startswith("observation.images.tacrgb_"):
                short = key[len("observation.images.tacrgb_") :]
                pad = next(p for p in PADS if _short(p) == short)
                return vids[f"observation.images.{pad}_tactile"]
            return None  # tacff: synthesised per-frame

        for k in img_keys:
            src = src_frames(k)
            h, w = src.shape[1:3] if src is not None else tacff[PADS[0]][0].shape[1:3]
            feats[k] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}

        ds = LeRobotDataset.create(
            repo_id=f"local/{name}_{variant}",
            fps=fps,
            features=feats,
            root=dst,
            robot_type="openarm_bimanual_tactile",
            vcodec=args.vcodec,
        )

        for ep in keep_eps:
            sel = np.where(ep_idx == ep)[0]
            for i in sel:
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task_text[int(task_idx[i])],
                }
                for k in VISION_KEYS:
                    frame[k] = vids[k][i]
                if variant == "tacff":
                    for pad in PADS:
                        d_u16, s_u16, depth_profile, shear_profile = tacff[pad]
                        depth = dequantize(d_u16[i], depth_profile)
                        if depth.ndim == 3:
                            depth = depth[..., 0]
                        shear_raw = np.asarray(s_u16[i])
                        shear_raw = np.where(shear_raw == 0, shear_profile.offset, shear_raw)
                        shear = dequantize(shear_raw, shear_profile)
                        key = f"observation.images.tacff_{_short(pad)}"
                        frame[key] = tacff_image(depth, shear)
                elif variant == "tacrgb":
                    for pad in PADS:
                        key = f"observation.images.tacrgb_{_short(pad)}"
                        frame[key] = vids[f"observation.images.{pad}_tactile"][i]
                ds.add_frame(frame)
            ds.save_episode()
            print(f"[{variant}] episode {ep} saved ({len(sel)} frames)", flush=True)
        ds.finalize()
        copy_review_videos(src_root, dst, keep_eps, variant)
        (dst / "meta/source_episodes.json").write_text(
            json.dumps([dict(source_map[ep], episode_index=i) for i, ep in enumerate(keep_eps)], indent=2)
            + "\n"
        )
        if annotations:
            (dst / "meta/episode_annotations.json").write_text(
                json.dumps(
                    [
                        dict(annotations[ep], episode_index=i)
                        for i, ep in enumerate(keep_eps)
                        if ep in annotations
                    ],
                    indent=2,
                )
                + "\n"
            )
        (dst / "meta/conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
        backup = None
        if final_dst.exists():
            backup = dst_root / f".{final_dst.name}-backup-{uuid.uuid4().hex}"
            final_dst.rename(backup)
        try:
            dst.rename(final_dst)
        except BaseException:
            if backup is not None:
                backup.rename(final_dst)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        staging.cleanup()
        print(f"[{variant}] done -> {final_dst}", flush=True)
    for frames in vids.values():
        mapping = getattr(frames, "_mmap", None)
        if mapping is not None:
            mapping.close()
    for depth, shear, _, _ in tacff.values():
        depth_mapping = getattr(depth, "_mmap", None)
        if depth_mapping is not None:
            depth_mapping.close()
        shear_planes = getattr(shear, "planes", shear)
        shear_mapping = getattr(shear_planes, "_mmap", None)
        if shear_mapping is not None:
            shear_mapping.close()
    merge_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
