#!/usr/bin/env python3
"""(Re)build the simulation robot USD.

Every entry point builds this automatically on first use, so this command is
only needed to force a rebuild -- after changing finger or camera placement,
for instance.

    python benchmark/run/build_asset.py
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build the tactile robot USD.")
parser.add_argument("--robot", default="",
                    help="stock OpenArm USD (default: the copy under assets/usds/)")
parser.add_argument("--gel", default="",
                    help="GelSight finger USD (default: the Isaac Lab Nucleus "
                         "TacSL asset)")
parser.add_argument("--out", default="",
                    help="output USD (default: beside the stock asset)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from sim.assets.build_robot_asset import graft
from sim.assets.provision import ROBOT_USD, ensure_robot_asset, ensure_stock_asset


def main() -> int:
    if not (args_cli.robot or args_cli.gel or args_cli.out):
        print(f"[build_asset] wrote {ensure_robot_asset(rebuild=True)}",
              flush=True)
        return 0

    robot = os.path.expanduser(args_cli.robot) if args_cli.robot else str(
        ensure_stock_asset())
    if not os.path.exists(robot):
        raise SystemExit(f"stock robot USD not found: {robot}")
    gel = args_cli.gel or (
        f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/"
        "gelsight_r15_finger.usd")
    out = os.path.expanduser(args_cli.out) or str(ROBOT_USD)
    if (os.path.dirname(os.path.abspath(out))
            != os.path.dirname(os.path.abspath(robot))):
        raise SystemExit("--out must sit beside the stock USD: its payload "
                         "references are relative")
    graft(robot_usd=robot, gel_usd=gel, out_usd=out)
    print(f"[build_asset] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    import sys
    sys.stdout.flush()
    os._exit(rc)
