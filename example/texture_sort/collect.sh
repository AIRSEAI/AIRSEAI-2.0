#!/usr/bin/env bash
# Expand surface pairs and collect them through the common entry points.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${1:?usage: collect.sh <dataset_out> [init_states_out]}"
PAIRS="${PAIRS:-250}"
if ! [[ "$PAIRS" =~ ^[1-9][0-9]*$ ]]; then
    echo "PAIRS must be a positive integer" >&2
    exit 64
fi
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
ARGS=(--dataset "$DATASET" --states-out "${2:-${DATASET}_init_states.json}"
      --out "${REPORT_DIR:-${DATASET}_run}" --pairs "$PAIRS"
      --max-candidates "${MAX_CANDIDATES:-$((PAIRS * 8))}"
      --batch-pairs "${BATCH_PAIRS:-32}" --num_envs "${NUM_ENVS:-4}"
      --seed "${SEED:-41}" --gpu "${GPU:-0}"
      --write_mode "${WRITE_MODE:-background}" --write_queue_size "${WRITE_QUEUE_SIZE:-2}")
if [ "${RENDER:-0}" = 1 ]; then
    ARGS+=(--render)
fi
exec "${DEFAULT_PY:-python}" "$REPO_DIR/example/texture_sort/collect_pairs.py" "${ARGS[@]}"
