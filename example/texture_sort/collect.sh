#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATES="${1:?usage: collect.sh <init_states.json> <dataset_out>}"
DATASET="${2:?usage: collect.sh <init_states.json> <dataset_out>}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${DEFAULT_PY:-python}" "$REPO_DIR/benchmark/run/collect_dataset.py" \
    --task texture_sort --init_states "$STATES" --dataset "$DATASET" \
    --num_envs "${NUM_ENVS:-4}" --gpus "${GPUS:-0}" --seed "${SEED:-7}" --render
