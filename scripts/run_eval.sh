#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Evaluate a checkpoint from frozen starts on the local simulation host.
set -euo pipefail
TASK="${1:?usage: run_eval.sh <task> <checkpoint_dir> [episodes] [seed]}"
CKPT="${2:?checkpoint directory required}"
EP="${3:-100}"
SEED="${4:-11}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$HOME/outputs/evaluation/$TASK/run_01}"
NUM_ENVS="${NUM_ENVS:-1}"
STATES="${INIT_STATES:-$REPO_DIR/meta/init_states/$TASK.json}"
[ "$STATES" != 1 ] || STATES="$REPO_DIR/meta/init_states/$TASK.json"
case "$CKPT" in hold|/*) ;; *) CKPT="$PWD/$CKPT" ;; esac
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
case "$STATES" in /*) ;; *) STATES="$PWD/$STATES" ;; esac
if [ "$CKPT" != hold ] && [ ! -f "$CKPT/config.json" ]; then
    echo "missing checkpoint config: $CKPT" >&2
    exit 1
fi
[ -f "$STATES" ] || { echo "missing initial states: $STATES" >&2; exit 1; }
[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ENVS must be positive" >&2; exit 64; }
cd "$REPO_DIR"
PY="${DEFAULT_PY:-python}"
if [ -z "${AIRSEAI_LEROBOT_DIR:-}" ] && [ -d "$HOME/lerobot/src" ]; then
    AIRSEAI_LEROBOT_DIR="$HOME/lerobot"
fi
if [ -z "${LD_PRELOAD:-}" ] && [ -x "$PY" ]; then
    PY_ROOT="$(cd "$(dirname "$PY")/.." && pwd)"
    [ -f "$PY_ROOT/lib/libstdc++.so.6" ] && export LD_PRELOAD="$PY_ROOT/lib/libstdc++.so.6"
fi
export PYTHONPATH="$PWD/src${AIRSEAI_LEROBOT_DIR:+:$AIRSEAI_LEROBOT_DIR/src}${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi
COMMON=(--task "$TASK" --headless --enable_cameras --episodes "$EP" --seed "$SEED"
        --out "$OUT" --num_envs "$NUM_ENVS" --init_states "$STATES")
if [ -n "${N_ACTION_STEPS:-}" ]; then
    COMMON+=(--n_action_steps "$N_ACTION_STEPS")
fi
if [ "${VIDEO:-0}" = 1 ]; then
    COMMON+=(--video_dir "$OUT/visual_review")
fi
INPROCESS="${INPROCESS:-1}"
if [ "$CKPT" = hold ] || [ "${INPROCESS:-0}" = 1 ]; then
    exec "$PY" benchmark/eval/evaluate.py "${COMMON[@]}" --policy "$CKPT"
fi
PORT="${PORT:-5555}"
SLOG=$(mktemp /tmp/policy_server.XXXX.log)
"$PY" benchmark/policy/policy_server.py --policy "$CKPT" --port "$PORT" > "$SLOG" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true' EXIT
for ((attempt = 0; attempt < 90; attempt++)); do
    grep -q listening "$SLOG" && break
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
done
grep -q listening "$SLOG" || { echo "policy server failed: $SLOG" >&2; tail -5 "$SLOG"; exit 1; }
"$PY" benchmark/eval/evaluate.py "${COMMON[@]}" --policy-server "127.0.0.1:$PORT"
