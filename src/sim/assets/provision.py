"""Locate or build the tactile robot USD after Isaac Sim has started.

Vendor geometry is downloaded from openarm_isaac_lab and is not redistributed.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ROBOT_USD", "STOCK_USD", "ensure_robot_asset", "ensure_stock_asset"]


def _find_repo() -> Path:
    """The checkout this module lives in (editable install or source tree)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


REPO = _find_repo()
# Generated robot and task assets are stored outside Git.
USDS_ROOT = REPO / "assets" / "usds"
ROBOT_ASSET_DIR = USDS_ROOT / "openarm_bimanual"


def task_asset_dir(task: str) -> Path:
    """Folder holding the generated USDs for one task: `assets/usds/<task>/`."""
    return USDS_ROOT / task


UPSTREAM_REPO = "https://github.com/enactic/openarm_isaac_lab"
UPSTREAM_SUBDIR = "source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/openarm_bimanual"
STOCK_USD = ROBOT_ASSET_DIR / "openarm_bimanual.usd"
ROBOT_USD = ROBOT_ASSET_DIR / "openarm_tactile.usd"

_MISSING = f"""vendor robot geometry not found at {ROBOT_ASSET_DIR}

Get it from enactic's OpenArm Isaac Lab (Apache-2.0):

    git clone --depth 1 {UPSTREAM_REPO} /tmp/openarm_isaac_lab
    mkdir -p {ROBOT_ASSET_DIR.parent}
    cp -r /tmp/openarm_isaac_lab/{UPSTREAM_SUBDIR} {ROBOT_ASSET_DIR.parent}/

then run the command again. Alternatively pass --robot_usd with a path to a
robot asset you built yourself."""


def ensure_stock_asset() -> Path:
    """The stock OpenArm USD; an error with instructions if it is absent."""
    if not STOCK_USD.exists():
        raise SystemExit(_MISSING)
    return STOCK_USD


def ensure_robot_asset(path: str | Path | None = None, rebuild: bool = False) -> str:
    """The tactile robot USD, built on demand.

    `path` overrides the default location; a caller-supplied path that does
    not exist is an error rather than something to build, because it is
    probably a typo.
    """
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise SystemExit(f"robot asset not found: {p}")
        return str(p)
    if ROBOT_USD.exists() and not rebuild:
        return str(ROBOT_USD)

    from .build_robot_asset import graft

    # AIRSEAI_GEL_USD: a local GelSight finger USD, used instead of the
    # Nucleus download.
    gel_usd = os.environ.get("AIRSEAI_GEL_USD", "")
    if not gel_usd:
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

        gel_usd = f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd"

    stock = ensure_stock_asset()
    print(f"[asset] building the tactile robot USD (gel: {gel_usd})", flush=True)
    graft(robot_usd=str(stock), gel_usd=gel_usd, out_usd=str(ROBOT_USD))
    return str(ROBOT_USD)
