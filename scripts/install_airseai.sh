#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install AIRSEAI and save its output root in the selected conda environment.
# AIRSEAI_USE_DEFAULT=1 accepts defaults without prompting.
# OUTPUT_ROOT overrides the default <repo dir>/outputs.
set -u

# ---------------- defaults ----------------
ENV_NAME="${AIRSEAI_ENV_NAME:-airseai}"
# Locate conda from the environment, then from the common install paths.
default_conda_root() {
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null && return 0
    fi
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
             /opt/conda /usr/local/miniconda3; do
        [ -x "$c/bin/conda" ] && { echo "$c"; return 0; }
    done
    echo "$HOME/miniconda3"
}
CONDA_ROOT="${AIRSEAI_CONDA_ROOT:-$(default_conda_root)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${AIRSEAI_REPO_DIR:-$(dirname "$SCRIPT_DIR")}"
LEROBOT_DIR="${AIRSEAI_LEROBOT_DIR:-$HOME/lerobot}"
ISAACLAB_DIR="${AIRSEAI_ISAACLAB_DIR:-$HOME/IsaacLab}"
# Clone sources: github.com first, the regional mirror only after the clone
# itself fails.
ISAACLAB_URL="${AIRSEAI_ISAACLAB_URL:-https://github.com/isaac-sim/IsaacLab.git}"
LEROBOT_URL="${AIRSEAI_LEROBOT_URL:-https://github.com/huggingface/lerobot}"
OPENARM_URL="${AIRSEAI_OPENARM_URL:-https://github.com/enactic/openarm_isaac_lab}"
# The mirror is derived by rewriting the github.com prefix, so one knob covers
# every clone. Empty (or AIRSEAI_NO_MIRROR_FALLBACK=1) disables the fallback.
GIT_MIRROR="${AIRSEAI_GIT_MIRROR:-https://gitclone.com/github.com/}"
GIT_TRIES="${AIRSEAI_GIT_TRIES:-2}"

configure_output_root() {
    case "$REPO_DIR" in
        '~') REPO_DIR="$HOME" ;;
        '~/'*) REPO_DIR="$HOME/${REPO_DIR#\~/}" ;;
    esac
    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR%/}/outputs}"
    if [ -t 0 ] && [ "${AIRSEAI_USE_DEFAULT:-0}" != "1" ]; then
        read -r -p "output root   [${OUTPUT_ROOT}]: " ans && [ -n "$ans" ] && OUTPUT_ROOT="$ans"
    fi
    case "$OUTPUT_ROOT" in
        '~') OUTPUT_ROOT="$HOME" ;;
        '~/'*) OUTPUT_ROOT="$HOME/${OUTPUT_ROOT#\~/}" ;;
    esac
    mkdir -p "$OUTPUT_ROOT" || return 1
    OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd -P)" || return 1
}

save_output_root() {
    conda env config vars set --prefix "$CONDA_PREFIX" "OUTPUT_ROOT=$1" || return 1
    export OUTPUT_ROOT="$1"
}

# ---------------- interactive parameters ----------------
if [ -t 0 ] && [ "${AIRSEAI_USE_DEFAULT:-0}" != "1" ]; then
    read -r -p "conda env name [${ENV_NAME}]: " ans && [ -n "$ans" ] && ENV_NAME="$ans"
    read -r -p "conda root    [${CONDA_ROOT}]: " ans && [ -n "$ans" ] && CONDA_ROOT="$ans"
    read -r -p "repo dir      [${REPO_DIR}]: " ans && [ -n "$ans" ] && REPO_DIR="$ans"
fi

configure_output_root || exit 1
mkdir -p "$OUTPUT_ROOT/output" || exit 1
LOG="$OUTPUT_ROOT/output/install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "### install_airseai.sh $(date '+%F %T')"
echo "### env=$ENV_NAME conda=$CONDA_ROOT repo=$REPO_DIR log=$LOG"

# ---------------- step machinery ----------------
declare -a RESULTS=()
FAILED=0
step() {   # step <critical|soft> <name> <command...>
    local kind="$1" name="$2"; shift 2
    echo; echo "=== [STEP] $name ==="
    local t0=$SECONDS
    if "$@"; then
        RESULTS+=("OK    $name ($((SECONDS-t0))s)"); return 0
    fi
    RESULTS+=("FAIL  $name ($((SECONDS-t0))s)")
    if [ "$kind" = critical ]; then
        FAILED=1; summary; echo "ABORTED at: $name"; exit 1
    fi
    echo "(non-critical, continuing)"; return 0
}
summary() {
    echo; echo "=== SUMMARY ==="
    printf '%s\n' "${RESULTS[@]}"
    echo "log: $LOG"
}

# ---------------- package sources ----------------
# International sources are the default; each has a regional mirror that is
# selected automatically after $PROBE_RETRIES failed reachability probes.
# Force either side explicitly with AIRSEAI_PIP_INDEX / AIRSEAI_TORCH_WHEELS,
# or set AIRSEAI_NO_MIRROR_FALLBACK=1 to fail instead of falling back.
PIP_INDEX="${AIRSEAI_PIP_INDEX:-https://pypi.org/simple/}"
PIP_INDEX_CN="${AIRSEAI_PIP_INDEX_CN:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_WHEELS="${AIRSEAI_TORCH_WHEELS:-https://download.pytorch.org/whl/cu128}"
TORCH_WHEELS_CN="${AIRSEAI_TORCH_WHEELS_CN:-https://mirrors.aliyun.com/pytorch-wheels/cu128/}"
PROBE_RETRIES="${AIRSEAI_PROBE_RETRIES:-3}"
PIP_RETRIES="${AIRSEAI_PIP_RETRIES:-3}"

reachable() {   # reachable <url> — $PROBE_RETRIES attempts, 15s each
    local url="$1" i
    for i in $(seq 1 "$PROBE_RETRIES"); do
        curl -fsSI --max-time 15 -o /dev/null "$url" && return 0
        # stderr: pick_source captures stdout, where this would corrupt the URL
        echo "  probe $i/$PROBE_RETRIES failed: $url" >&2
        sleep 2
    done
    return 1
}

pick_source() {  # pick_source <label> <primary> <cn-mirror> -> echoes the winner
    local label="$1" primary="$2" mirror="$3"
    if reachable "$primary"; then echo "$primary"; return 0; fi
    if [ "${AIRSEAI_NO_MIRROR_FALLBACK:-0}" = "1" ] || [ -z "$mirror" ]; then
        echo "$primary"; return 0
    fi
    echo "!! $label unreachable after $PROBE_RETRIES tries — falling back to $mirror" >&2
    echo "$mirror"
}

select_sources() {
    PIP_INDEX="$(pick_source "pip index" "$PIP_INDEX" "$PIP_INDEX_CN")"
    TORCH_WHEELS="$(pick_source "torch wheels" "$TORCH_WHEELS" "$TORCH_WHEELS_CN")"
    # Exported so every pip in this process tree (including isaaclab.sh's)
    # uses the same index.
    export PIP_INDEX_URL="$PIP_INDEX"
    echo "pip index   : $PIP_INDEX"
    echo "torch wheels: $TORCH_WHEELS"
}

pip_install() {  # pip install, retrying once through the regional mirror
    pip install --retries "$PIP_RETRIES" "$@" && return 0
    [ "${AIRSEAI_NO_MIRROR_FALLBACK:-0}" != "1" ] && [ -n "$PIP_INDEX_CN" ] \
        && [ "$PIP_INDEX" != "$PIP_INDEX_CN" ] || return 1
    echo "!! pip failed on $PIP_INDEX — retrying via $PIP_INDEX_CN"
    pip install --retries "$PIP_RETRIES" --index-url "$PIP_INDEX_CN" "$@"
}

preflight() {
    grep PRETTY /etc/os-release
    # Captured, not piped, so a missing nvidia-smi fails the check.
    local smi drv
    if ! smi=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1); then
        echo "nvidia-smi is not usable — install the NVIDIA driver first:"
        echo "$smi"
        return 1
    fi
    echo "$smi" | head -2
    drv=$(echo "$smi" | head -1 | awk -F', *' '{print $2}' | cut -d. -f1)
    case "$drv" in
        "" | *[!0-9]*)
            echo "WARNING: cannot parse a driver version out of: $smi" ;;
        *)
            [ "$drv" -lt 580 ] \
                && echo "WARNING: driver $drv < 580 required by README; continuing anyway" ;;
    esac
    df -h "$HOME" | tail -1
    [ -x "$CONDA_ROOT/bin/conda" ] || { echo "conda not found at $CONDA_ROOT"; return 1; }
    [ -f "$REPO_DIR/pyproject.toml" ] || { echo "repo not found at $REPO_DIR"; return 1; }
}

make_env() {
    local selected_output_root="$OUTPUT_ROOT"
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        if [ "${AIRSEAI_RECREATE:-0}" = "1" ]; then
            conda env remove -n "$ENV_NAME" -y
        else
            echo "env '$ENV_NAME' exists — reusing (AIRSEAI_RECREATE=1 to wipe)"
        fi
    fi
    # conda-forge only: the Anaconda default channels refuse to serve until
    # their Terms of Service are accepted, and nothing past the interpreter
    # comes from conda anyway.
    conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
        || conda create -n "$ENV_NAME" python=3.11 -y \
                        -c conda-forge --override-channels || return 1
    conda activate "$ENV_NAME" || return 1
    save_output_root "$selected_output_root" || return 1
    python -V | grep -q "3\.11" || { echo "env python is not 3.11"; return 1; }
    # stdout is a pipe (everything is tee'd), so pip hides its progress bar.
    # "raw" prints byte counts, which survive a pipe. Older pip rejects the
    # value, hence the probe.
    if python -m pip install --progress-bar raw --help >/dev/null 2>&1; then
        export PIP_PROGRESS_BAR="${AIRSEAI_PIP_PROGRESS:-raw}"
    fi
}

configure_runtime() {
    local library="$CONDA_PREFIX/lib/libstdc++.so.6"
    if [ -f "$library" ]; then
        local preload saved_vars
        saved_vars=$(conda env config vars list --prefix "$CONDA_PREFIX" --json) || return 1
        preload=$(AIRSEAI_SAVED_VARS="$saved_vars" python - "$library" <<'PYEOF'
import json
import os
import re
import sys

saved = json.loads(os.environ["AIRSEAI_SAVED_VARS"])
entries = [sys.argv[1]]
for value in (saved.get("LD_PRELOAD", ""), os.environ.get("LD_PRELOAD", "")):
    entries.extend(item for item in re.split(r"[:\s]+", value) if item)
print(":".join(dict.fromkeys(entries)))
PYEOF
) || return 1
        conda env config vars set --prefix "$CONDA_PREFIX" "LD_PRELOAD=$preload" || return 1
        export LD_PRELOAD="$preload"
    fi
}

install_torch() {
    if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.7.0') else 1)" 2>/dev/null; then
        echo "torch 2.7.0 already present — skip"; return 0
    fi
    pip install --retries "$PIP_RETRIES" \
        "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" --index-url "$TORCH_WHEELS" && return 0
    [ "${AIRSEAI_NO_MIRROR_FALLBACK:-0}" != "1" ] && [ "$TORCH_WHEELS" != "$TORCH_WHEELS_CN" ] || return 1
    echo "!! torch wheels failed on $TORCH_WHEELS — retrying via $TORCH_WHEELS_CN"
    pip install --retries "$PIP_RETRIES" \
        "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" -f "$TORCH_WHEELS_CN"
}

git_clone_retry() {  # git_clone_retry <url> <dir> [tag] [sparse-path]
    local url="$1" dir="$2" ref="${3:-}" sparse="${4:-}" src try n=0
    local -a refarg=(); [ -n "$ref" ] && refarg=(--branch "$ref")
    # Sparse + blob-filtered: only the requested paths are downloaded. Needs
    # server-side partial-clone support, so callers must handle failure.
    local -a filt=(); [ -n "$sparse" ] && filt=(--filter=blob:none --sparse)
    local -a sources=("$url")
    if [ "${AIRSEAI_NO_MIRROR_FALLBACK:-0}" != "1" ] && [ -n "$GIT_MIRROR" ]; then
        local m="${url/https:\/\/github.com\//$GIT_MIRROR}"
        [ "$m" != "$url" ] && sources+=("$m")
    fi
    for src in "${sources[@]}"; do
        [ "$src" != "$url" ] && echo "!! $url unreachable — falling back to $src" >&2
        for try in $(seq 1 "$GIT_TRIES"); do
            # only wipe what one of our own attempts left behind
            n=$((n+1)); [ "$n" -gt 1 ] && rm -rf "$dir"
            # Per-invocation options, not `git config --global`: HTTP/1.1 and
            # a shallow fetch survive the aborts a flaky link produces.
            if git -c http.version=HTTP/1.1 \
                   -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 \
                   clone --depth 1 "${refarg[@]}" "${filt[@]}" "$src" "$dir"; then
                if [ -n "$sparse" ] && ! git -C "$dir" sparse-checkout set "$sparse"; then
                    echo "  sparse-checkout failed: $sparse" >&2
                    sleep 1; continue
                fi
                # the mirror is a download path, not the repo's identity
                [ "$src" != "$url" ] && git -C "$dir" remote set-url origin "$url"
                return 0
            fi
            echo "  clone $try/$GIT_TRIES failed: $src" >&2
            sleep 5
        done
    done
    return 1
}

install_isaacsim() {
    if pip show isaacsim >/dev/null 2>&1; then echo "isaacsim already installed — skip"; return 0; fi
    # isaacsim 5.1.0.0 (incl. extscache) resolves from the selected index.
    pip_install "isaacsim[all,extscache]==5.1.0.0"
}

install_isaaclab() {
    if python -c "import isaaclab, isaaclab_contrib" 2>/dev/null; then
        echo "isaaclab + isaaclab_contrib already importable — skip"; return 0
    fi
    if [ ! -d "$ISAACLAB_DIR/.git" ]; then
        git_clone_retry "$ISAACLAB_URL" "$ISAACLAB_DIR" v2.3.2 || return 1
    fi
    # flatdict's setup.py needs pkg_resources, which setuptools>=81 removed;
    # pin setuptools and pre-build flatdict without build isolation.
    pip_install "setuptools<81" wheel
    local flatdict
    # Not `grep ... || echo default`: the exit status of that pipeline is
    # head's, so a failed grep yields an empty argument instead of the default.
    flatdict=$(grep -o 'flatdict[=<>~!]*[0-9.]*' \
                    "$ISAACLAB_DIR/source/isaaclab/setup.py" 2>/dev/null | head -1)
    [ -n "$flatdict" ] || flatdict="flatdict==4.0.1"
    pip_install --no-build-isolation "$flatdict"
    # rl-games' download can fail; only isaaclab + isaaclab_contrib are needed.
    ( cd "$ISAACLAB_DIR" && ./isaaclab.sh --install ) || true
    # ensure the core package made it in even if an RL extra aborted the loop
    python -c "import isaaclab" 2>/dev/null || pip_install -e "$ISAACLAB_DIR/source/isaaclab"
    python -c "import isaaclab, isaaclab_contrib; print('isaaclab', isaaclab.__version__)"
}

install_repo() {
    ( cd "$REPO_DIR" && pip_install -e '.[codec,dataset,dev]' ) || return 1
    # Every repo package must import from REPO_DIR/src.
    ( cd / && python -c '
import sys
import re
from pathlib import Path
want = (Path(sys.argv[1]) / "src").resolve()
bad = 0
for name in ("sim", "tactile_codec", "lerobot_tactile_ext"):
    got = Path(__import__(name).__file__).resolve()
    ok = want in got.parents
    bad += not ok
    print(("ok  " if ok else "BAD ") + name + " -> " + str(got))
raise SystemExit(bad)
' "$REPO_DIR" )
}

install_lerobot() {
    if [ ! -d "$LEROBOT_DIR/.git" ]; then
        git_clone_retry "$LEROBOT_URL" "$LEROBOT_DIR" v0.5.1 || return 1
    else
        echo "lerobot dir exists: $(git -C "$LEROBOT_DIR" describe --tags --always 2>/dev/null)"
    fi
    pip_install datasets pandas huggingface_hub
    conda env config vars set --prefix "$CONDA_PREFIX" "PYTHONPATH=$LEROBOT_DIR/src" || return 1
    rm -f "$CONDA_PREFIX/etc/conda/activate.d/lerobot_pythonpath.sh"
    export PYTHONPATH="$LEROBOT_DIR/src"
    python -c "import lerobot; print('lerobot', lerobot.__version__)"
}

patch_lerobot_py311() {
    # lerobot 0.5 uses py3.12-only PEP 695 syntax, while Isaac Sim 5.1 pins
    # cp311. Rewrite it to 3.11 equivalents: generic class/def parameters in
    # three files, plus "type X = ..." aliases anywhere. Idempotent.
    python - <<PY
from pathlib import Path
base = Path("$LEROBOT_DIR/src/lerobot")

# "type X = ...": a plain assignment is equivalent at runtime. Tree-wide rather
# than by filename, since a point release can add more anywhere.
import re
_alias = re.compile(r"^(\s*)type\s+([A-Za-z_]\w*)\s*=\s*(.+)")
for _f in sorted(base.rglob("*.py")):
    _out, _hit = [], False
    for _ln in _f.read_text().split("\n"):
        _m = _alias.match(_ln)
        if _m:
            _ln = _m.group(1) + _m.group(2) + " = " + _m.group(3)
            _hit = True
        _out.append(_ln)
    if _hit:
        _f.write_text("\n".join(_out))
        print("alias-patched:", _f.relative_to(base))

def patch(rel, old, new):
    p = base / rel
    s = p.read_text()
    if old in s:
        p.write_text(s.replace(old, new, 1)); print("patched:", rel)
    else:
        print("ok (already patched or absent):", rel)

patch("utils/io_utils.py",
"def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:",
"""from typing import TypeVar as _TypeVar
T = _TypeVar("T", bound="JsonLike")


def deserialize_json_into_object(fpath: Path, obj: T) -> T:""")

patch("datasets/streaming_dataset.py",
"class Backtrackable[T]:",
"""from typing import Generic as _Generic, TypeVar as _TypeVar
T = _TypeVar("T")


class Backtrackable(_Generic[T]):""")

# pipeline.py: the class is decorated, so the helper defs must go above the
# decorator line.
p = base / "processor/pipeline.py"
s = p.read_text()
old = "class DataProcessorPipeline[TInput, TOutput](HubMixin):"
if old in s:
    lines = s.split("\\n")
    for i, ln in enumerate(lines):
        if ln.startswith(old):
            j = i
            while j > 0 and lines[j-1].lstrip().startswith("@"):
                j -= 1
            ins = ["from typing import Generic as _Generic, TypeVar as _TypeVar",
                   "TInput = _TypeVar(\\"TInput\\")",
                   "TOutput = _TypeVar(\\"TOutput\\")", "", ""]
            lines[i] = "class DataProcessorPipeline(HubMixin, _Generic[TInput, TOutput]):"
            lines[j:j] = ins
            break
    p.write_text("\\n".join(lines)); print("patched: processor/pipeline.py")
else:
    print("ok (already patched or absent): processor/pipeline.py")
PY
    if ! python -m compileall -q "$LEROBOT_DIR/src/lerobot"; then
        echo "!! lerobot does not compile under python 3.11 — see the error above"
        return 1
    fi
    return 0
}

install_policy_deps() {
    local tf
    tf=$(grep -o 'transformers[=<>~!]*[0-9.]*' \
              "$ISAACLAB_DIR/source/isaaclab/setup.py" 2>/dev/null | head -1)
    [ -n "$tf" ] || tf="transformers"
    pip_install "draccus==0.10.0" "$tf" accelerate einops diffusers peft \
                pyserial deepdiff jsonlines termcolor wandb
}

install_video_deps() {
    pip_install imageio imageio-ffmpeg "opencv-python-headless==4.11.0.86"
}

verify_policy_io() {
    # Stage 4 writes a checkpoint that stage 5 loads back. Both ends break
    # silently on the wrong dependency version, so check them here instead of
    # after a training run.
    python - <<'PY'
import io
import json

import draccus
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import get_policy_class

buf = io.StringIO()
with draccus.config_type("json"):
    draccus.dump(ACTConfig(), buf)
if "type" not in json.loads(buf.getvalue()):
    raise SystemExit("draccus writes no 'type' key: stage 5 cannot load a "
                     "stage 4 checkpoint. Needs draccus==0.10.0.")
get_policy_class("act")
print("policy save/load ok")
PY
}

fetch_robot_asset() {
    if [ -d "$REPO_DIR/assets/usds/openarm_bimanual" ]; then
        echo "openarm_bimanual USD already present — skip"; return 0
    fi
    # Only the bimanual USD directory is needed: sparse-fetch that path, with a
    # full clone as the fallback.
    local usds=source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds
    local asset_source
    asset_source=$(mktemp -d "$OUTPUT_ROOT/openarm-assets.XXXXXX") || return 1
    git_clone_retry "$OPENARM_URL" "$asset_source" "" "$usds/openarm_bimanual" || {
        echo "!! sparse fetch failed — retrying as a full clone" >&2
        rm -rf "$asset_source"
        git_clone_retry "$OPENARM_URL" "$asset_source" || return 1
    }
    [ -d "${asset_source}/$usds/openarm_bimanual" ] || {
        echo "openarm_bimanual not found in the clone — upstream layout changed?"; return 1
    }
    mkdir -p "$REPO_DIR/assets/usds"
    cp -r "${asset_source}/$usds/openarm_bimanual" "$REPO_DIR/assets/usds/"
}

make_task_assets() {
    # Regenerate every task's USDs under assets/usds/ from its generator,
    # overwriting whatever is there. The task list comes from the registry,
    # so a new task needs no edit here.
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    cd "$REPO_DIR" || return 1
    local list name script
    list=$(PYTHONPATH="$REPO_DIR/src" python - <<'PYEOF'
from sim.tasks.registry import TASKS
for name, script in sorted({(e.name, e.assets) for e in TASKS.values() if e.assets}):
    print(name, script)
PYEOF
) || return 1
    while read -r name script; do
        [ -n "$name" ] || continue
        rm -rf "assets/usds/$name"
        python "$script" --headless || return 1
    done <<< "$list"
    ls assets/usds/
}

install_ffmpeg() {
    # Stage 3 decodes the ffv1 tactile streams with the ffmpeg CLI.
    if command -v ffmpeg >/dev/null 2>&1; then
        echo "ffmpeg present: $(ffmpeg -version 2>/dev/null | head -1)"; return 0
    fi
    if ! sudo -v; then
        echo "!! ffmpeg is missing and sudo is unavailable — install it with:"
        echo "!!     sudo apt-get install -y ffmpeg"
        return 1
    fi
    sudo apt-get update && sudo apt-get install -y ffmpeg || return 1
    ffmpeg -hide_banner -decoders 2>/dev/null | grep -q ffv1 \
        || { echo "!! installed ffmpeg has no ffv1 decoder"; return 1; }
}

verify() {
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    python - "$REPO_DIR" <<'PYEOF'
import importlib
import sys
import tempfile
from pathlib import Path

import av
import cv2
import gymnasium as gym
import imageio_ffmpeg
import numpy as np

root = (Path(sys.argv[1]) / "src").resolve()
for name in ("sim", "tactile_codec", "lerobot_tactile_ext", "gym_airseai", "lerobot_airseai"):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if root not in path.parents:
        raise RuntimeError(f"{name} loaded from unexpected path: {path}")
    print(f"ok {name}: {path}")

from gym_airseai import TASK_IDS
from lerobot_airseai import AirseaiEnvConfig

from sim.video import Mp4Writer

dataset = importlib.import_module("lerobot.datasets.lerobot_dataset")
for name in ("LeRobotDataset", "LeRobotDatasetMetadata"):
    getattr(dataset, name)
if not Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file():
    raise RuntimeError("Recording ffmpeg executable is missing")

for task in TASK_IDS:
    gym.spec(f"gym_airseai/{task}")
    for variant in ("vision", "tacrgb", "tacff"):
        AirseaiEnvConfig(task=task, variant=variant)
print("Gym registration, LeRobot configurations and dataset imports ok")

with tempfile.TemporaryDirectory(prefix="airseai-video-") as folder:
    path = Path(folder) / "recording.mp4"
    writer = Mp4Writer(path, fps=20)
    try:
        for value in (0, 64, 128, 255):
            frame = np.full((64, 96, 3), value, dtype=np.uint8)
            cv2.putText(frame, "test", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
            writer.add(frame)
    finally:
        writer.close()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != 20:
            raise RuntimeError("Recording frame rate mismatch")
        frames = list(container.decode(video=0))
    if len(frames) != 4 or any((frame.width, frame.height) != (96, 64) for frame in frames):
        raise RuntimeError("Recording frame count or dimensions mismatch")
print("Video encode/decode ok")
PYEOF
    [ "$?" -eq 0 ] || return 1
    ( cd "$REPO_DIR" && python benchmark/run/verify_codec.py --out "$OUTPUT_ROOT/verify/stage0/codec" ) || return 1
    python -c "import isaaclab, isaaclab_contrib; print('isaac ok')"
}

step critical "package sources"      select_sources
step critical "preflight"            preflight
step critical "ffmpeg (apt)"         install_ffmpeg
step critical "conda env ($ENV_NAME, py3.11)" make_env
step critical "C++ runtime"          configure_runtime
step critical "torch 2.7.0 cu128"    install_torch
step critical "isaacsim 5.1.0.0"     install_isaacsim
step critical "IsaacLab v2.3.2"      install_isaaclab
step critical "repo -e .[codec,dataset,dev]" install_repo
step critical "lerobot 0.5.1 (source, py3.11)" install_lerobot
step critical "lerobot py3.11 compat patches" patch_lerobot_py311
step critical "policy dependencies"  install_policy_deps
step critical "video dependencies"   install_video_deps
step critical "policy save/load"     verify_policy_io
step critical "OpenArm robot asset"  fetch_robot_asset
step critical "task assets (boards, balls, cans)" make_task_assets
step critical "verify"               verify

summary
echo "DONE $(date '+%F %T')"
