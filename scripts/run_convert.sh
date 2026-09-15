#!/bin/bash
# Stage 3 — dataset conversion (airseai env, lerobot from $AIRSEAI_LEROBOT_DIR).
#
# Raw recording (LeRobot-v3 style, ffv1/mkv tactile streams) -> the standard
# LeRobot datasets the trainer reads, one per input variant.
#
#   bash scripts/run_convert.sh <raw_dataset_dir>
#   VARIANTS="tacff" bash scripts/run_convert.sh <raw_dataset_dir>
#   FORCE=1 bash scripts/run_convert.sh <raw_dataset_dir>
#
# Inputs
#   $1          raw dataset dir (what run_collect.sh wrote); required
#   VARIANTS    space-separated subset of: vision tacff tacrgb   default: all
#   FORCE=1     reconvert a variant that already exists
#   BENCH_ROOT  work root       default $HOME/airseai-benchmark
#   BASE        default $BENCH_ROOT/datasets
#
# Output: $BASE/<name>_<variant> for each variant, where <name> is the raw
# dataset's directory name, so the converted sets carry the recording's name.
set -eu
RAW="${1:?usage: run_convert.sh <raw_dataset_dir>}"
RAW="${RAW%/}"
NAME="$(basename "$RAW")"
VARIANTS="${VARIANTS:-vision tacff tacrgb}"
BENCH_ROOT="${BENCH_ROOT:-$HOME/airseai-benchmark}"
BASE="${BASE:-$BENCH_ROOT/datasets}"

for v in $VARIANTS; do
  case "$v" in vision|tacff|tacrgb) ;; *) echo "unknown variant: $v"; exit 64 ;; esac
done

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# lerobot is a source checkout on PYTHONPATH, as install_benchmark.sh sets it up.
LEROBOT_DIR="${AIRSEAI_LEROBOT_DIR:-$HOME/lerobot}"
export PYTHONPATH="$PWD/src:$LEROBOT_DIR/src"
export HF_HUB_OFFLINE=1
CONDA_ENVS="${CONDA_ENVS:-$HOME/miniconda3/envs}"
PY="${SIM_PY:-$CONDA_ENVS/airseai/bin/python}"

[ -d "$RAW" ] || { echo "raw dataset not found: $RAW"; exit 1; }

todo=()
for v in $VARIANTS; do
    if [ -d "$BASE/${NAME}_$v" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "exists, skipping: $BASE/${NAME}_$v  (FORCE=1 to redo)"
    else
        todo+=("$v")
    fi
done
[ ${#todo[@]} -eq 0 ] && { echo "nothing to convert"; exit 0; }

echo "converting ${todo[*]} from $RAW -> $BASE/${NAME}_<variant>"
for v in "${todo[@]}"; do
    "$PY" benchmark/policy/convert_dataset.py \
        --src "$RAW" --dst-root "$BASE" --name "$NAME" --variant "$v" --overwrite
done
echo "done:"; ls -d "$BASE/${NAME}_"* 
