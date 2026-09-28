#!/bin/bash
# Collect expert attempts; accepted episodes are combined into one dataset.
#   bash scripts/run_collect.sh <task> [episodes] [dataset_out]
# NUM_ENVS, GPUS, SEED, INIT_STATES, RENDER and PREVIEWS configure collection.
# WRITE_MODE and WRITE_QUEUE_SIZE configure episode saving.
set -euo pipefail
TASK="${1:?usage: run_collect.sh <task> [episodes] [dataset_out]}"
EP="${2:-50}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${3:-${OUTPUT_ROOT:-$REPO_DIR/outputs}/raw/${TASK}_expert}"
case "$OUT" in
    /*) ;;
    '~/'*) OUT="$HOME/${OUT#\~/}" ;;
    *) OUT="$PWD/$OUT" ;;
esac

cd "$REPO_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
PY="${DEFAULT_PY:-python}"

ARGS=(--task "$TASK" --episodes "$EP" --dataset "$OUT"
      --num_envs "${NUM_ENVS:-1}" --gpus "${GPUS:-0}" --seed "${SEED:-7}")
[ "${RENDER:-0}"   = 1 ] && ARGS+=(--render)
[ "${PREVIEWS:-0}" = 1 ] && ARGS+=(--previews)
[ -n "${INIT_STATES:-}" ] && ARGS+=(--init_states "$INIT_STATES")
[ -n "${WRITE_MODE:-}" ] && ARGS+=(--write_mode "$WRITE_MODE")
ARGS+=(--write_queue_size "${WRITE_QUEUE_SIZE:-2}")

echo "collect: task=$TASK episodes=$EP -> $OUT"
exec "$PY" benchmark/run/collect_dataset.py "${ARGS[@]}"
