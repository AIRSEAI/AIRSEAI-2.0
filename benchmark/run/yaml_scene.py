#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone native-LUT scene preview; original project entry points are unchanged."""

import argparse
import json
import os
from pathlib import Path

from sim.yaml_scene import load_startup_environment


def create_scene(spec, robot_usd, device, lut_directory, cfg=None):
    """Compose the original Harness and calibrated sensors before physics reset."""
    from sim.envs.bimanual_tactile import make_preview_cfg
    from sim.harness import Harness
    from sim.tactile import apply_sdf_collision

    h = Harness(
        cfg or make_preview_cfg(),
        device=device,
        grid=(36, 48),
        robot_usd=robot_usd,
        environment_yaml=spec["source"],
    )
    if not apply_sdf_collision("/World/envs/env_0/manipuland", sdf_resolution=256):
        raise RuntimeError("Failed to configure preview SDF collider")
    return h, h.environment_receipt


def main():
    from isaaclab.app import AppLauncher

    from sim.runtime import install_error_handler

    install_error_handler()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-yaml", required=True)
    parser.add_argument("--robot_usd", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=90)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    spec = load_startup_environment(args.environment_yaml)
    args.enable_cameras = True
    app = AppLauncher(args).app
    if not app.is_running():
        raise RuntimeError("Isaac app did not start")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    h, receipt = create_scene(spec, args.robot_usd, args.device, out / "lut")
    h.start()
    h.initial_render()
    h.hold_default(args.steps)
    from PIL import Image

    for name, camera in h.cams.items():
        camera.update(1 / 60, force_recompute=True)
        # The renderer's RGB is already final; no remap, resize or second distortion.
        Image.fromarray(h.cam_rgb(name)).save(out / f"{name}.png")
    receipt["robot_usd"] = args.robot_usd
    receipt["isaac_executed"] = True
    (out / "environment-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Native LUT images and receipt: {out}", flush=True)
    # The original standalone scripts also avoid Kit sensor teardown hangs.
    os._exit(0)


if __name__ == "__main__":
    main()
