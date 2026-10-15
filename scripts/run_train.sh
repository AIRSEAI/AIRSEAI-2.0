#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Train a LeRobot policy on a converted dataset and save checkpoints and logs.
set -euo pipefail
DATASET="${DATASET:?set DATASET=<raw recording name, e.g. blind_peg_insertion_dataset>}"
VARIANT="${VARIANT:?set VARIANT=vision|tacff|tacrgb}"
POLICY="${POLICY:-act}"
case "$VARIANT" in
    vision|tacrgb|tacff) ;;
    *) echo "VARIANT must be vision, tacrgb or tacff" >&2; exit 64 ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/airseai-benchmark}"
BASE="${BASE:-$OUTPUT_ROOT}"
DS="${DS:-$BASE/datasets/${DATASET}_${VARIANT}}"
STEPS="${STEPS:-50000}"
SEED="${SEED:-1000}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
for value in "$STEPS" "$BATCH" "$SAVE_FREQ" "$LOG_FREQ"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "steps, batch and frequencies must be positive integers" >&2
        exit 64
    fi
done
for value in "$SEED" "$WORKERS"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "seed and workers must be nonnegative integers" >&2
        exit 64
    fi
done
if [ -z "${GPU:-}" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2,2nr | awk -F, 'NR == 1 {gsub(/[[:space:]]/, "", $1); print $1}')
    [ -n "$GPU" ] || { echo "no GPU available; set GPU explicitly" >&2; exit 1; }
fi

case "$BASE" in /*) ;; *) BASE="$PWD/$BASE" ;; esac
case "$DS" in /*) ;; *) DS="$PWD/$DS" ;; esac
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PWD/src${AIRSEAI_LEROBOT_DIR:+:$AIRSEAI_LEROBOT_DIR/src}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
PY="${DEFAULT_PY:-python}"

if [ ! -f "$DS/meta/info.json" ]; then
    echo "converted dataset not found: $DS/meta/info.json" >&2
    exit 1
fi

OUT="$BASE/train/${DATASET}_${VARIANT}_${POLICY}_s${SEED}"
[ ! -e "$OUT" ] || { echo "training output already exists: $OUT" >&2; exit 1; }
mkdir -p "$BASE/logs"
echo "training $POLICY on ${DATASET}_${VARIANT}, GPU $GPU"
echo "  dataset: $DS"
echo "  output : $OUT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m lerobot.scripts.lerobot_train \
  --policy.type="$POLICY" --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="local/${DATASET}_${VARIANT}" \
  --dataset.root="$DS" \
  --output_dir="$OUT" --batch_size="$BATCH" --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" --log_freq="$LOG_FREQ" --seed="$SEED" --num_workers="$WORKERS" \
  --wandb.enable=false "$@" \
  2>&1 | tee "$BASE/logs/train_${DATASET}_${VARIANT}_${POLICY}_s${SEED}.log"
