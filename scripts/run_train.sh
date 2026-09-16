#!/bin/bash
# Stage 4 — BC training (airseai env, lerobot from $AIRSEAI_LEROBOT_DIR).
#
# Trains only. Run stage 3 first:
#
#   bash scripts/run_convert.sh ~/blind_peg_insertion_dataset
#   DATASET=blind_peg_insertion_dataset VARIANT=tacff bash scripts/run_train.sh
#
# The policy is ACT, with lerobot's ImageNet-pretrained ResNet18 as the
# backbone for every image stream, tactile included.
#
# Inputs (env vars; DATASET and VARIANT are required)
#   DATASET  the raw recording's directory name, as stage 3 named its outputs
#   VARIANT  vision | tacff | tacrgb
#            vision : observation.images.base + wrist cameras + state(16)
#            tacff  : + tacff_{right,left}_{middle,index} (force field as 3ch image:
#                     R=depth, G/B=shear x/y, fixed scales)
#            tacrgb : + tacrgb_{right,left}_{middle,index} (raw Taxim tactile RGB)
#            action : 16-dim joint targets (deg / gripper mm)
#   POLICY   lerobot --policy.type, default act
#   BASE     work root, default $BENCH_ROOT
#   DS       converted dataset dir, default $BASE/datasets/${DATASET}_$VARIANT
#   GPU      default: the GPU with the most free memory
#   STEPS    default 50000        SEED  default 1000
#
# Output: $BASE/outputs/${DATASET}_${VARIANT}_${POLICY}_s${SEED}/checkpoints/...
set -eu
DATASET="${DATASET:?set DATASET=<raw recording name, e.g. blind_peg_insertion_dataset>}"
VARIANT="${VARIANT:?set VARIANT=vision|tacff|tacrgb}"
POLICY="${POLICY:-act}"

BENCH_ROOT="${BENCH_ROOT:-$HOME/airseai-benchmark}"
BASE="${BASE:-$BENCH_ROOT}"
DS="${DS:-$BASE/datasets/${DATASET}_${VARIANT}}"
STEPS="${STEPS:-50000}"
SEED="${SEED:-1000}"
if [ -z "${GPU:-}" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# lerobot is a source checkout on PYTHONPATH; its train module is run directly.
LEROBOT_DIR="${AIRSEAI_LEROBOT_DIR:-$HOME/lerobot}"
export PYTHONPATH="$PWD/src:$LEROBOT_DIR/src"
export HF_HUB_OFFLINE=1
CONDA_ENVS="${CONDA_ENVS:-$HOME/miniconda3/envs}"
PY="${SIM_PY:-$CONDA_ENVS/airseai/bin/python}"

if [ ! -d "$DS" ]; then
    echo "converted dataset not found: $DS"
    echo "run stage 3 first:  VARIANTS=$VARIANT bash scripts/run_convert.sh <raw_dataset_dir>"
    exit 1
fi

OUT="$BASE/outputs/${DATASET}_${VARIANT}_${POLICY}_s${SEED}"
rm -rf "$OUT"          # lerobot-train refuses an existing dir
mkdir -p "$BASE/logs"
echo "training $POLICY on ${DATASET}_${VARIANT}, GPU $GPU"
echo "  dataset: $DS"
echo "  output : $OUT"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m lerobot.scripts.lerobot_train \
  --policy.type="$POLICY" --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="local/${DATASET}_${VARIANT}" \
  --dataset.root="$DS" \
  --output_dir="$OUT" --batch_size="${BATCH:-8}" --steps="$STEPS" \
  --save_freq=10000 --log_freq=200 --seed="$SEED" --num_workers=8 \
  --wandb.enable=false \
  2>&1 | tee "$BASE/logs/train_${DATASET}_${VARIANT}_${POLICY}_s${SEED}.log"
