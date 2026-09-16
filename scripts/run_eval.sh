#!/bin/bash
# Stage 5 — evaluate a policy checkpoint on a task (Isaac Sim, airseai env).
#
#   bash scripts/run_eval.sh <task> <checkpoint_dir> [episodes] [seed]
#
#   bash scripts/run_eval.sh blind_peg_insertion ~/ckpt/act_tacff 25
#   bash scripts/run_eval.sh stack_cube ~/ckpt/act_stack 10
#   VIDEO=1 bash scripts/run_eval.sh blind_peg_insertion ~/ckpt/... 5
#   bash scripts/run_eval.sh stack_cube hold 3          # loop test, no checkpoint
#
# Inputs
#   $1  task        stack_cube | blind_peg_insertion | blind_peg_dim | texture_sort
#   $2  checkpoint  lerobot pretrained_model dir, or "hold" for the hold-pose
#                   loop test (action = current state; success expected 0)
#   $3  episodes    default 25
#   $4  seed        default 11
#   OUT         report dir, default ~/<task>_eval
#   VIDEO=1     also record base / active-wrist / third-person mp4s
#   INPROCESS=1 load the policy inside the Isaac process instead of serving
#               it from a separate process
#
#   PORT        policy server port, default 5555
#   SIM_PY      python to use, default $CONDA_ENVS/airseai/bin/python
#
# The policy runs in its own process and is reached over localhost; the
# observation keys are read from the checkpoint.
# Report: $OUT/eval_report.json
set -eu
TASK="${1:?usage: run_eval.sh <task> <checkpoint_dir|hold> [episodes] [seed]}"
CKPT="${2:?checkpoint dir (or 'hold') required}"
EP="${3:-25}"
SEED="${4:-11}"
OUT="${OUT:-$HOME/${TASK}_eval}"
VID_ARGS=(); [ "${VIDEO:-0}" = 1 ] && VID_ARGS=(--video_dir "${OUT}_videos")

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_DIR="${AIRSEAI_LEROBOT_DIR:-$HOME/lerobot}"
export PYTHONPATH="$PWD/src:$LEROBOT_DIR/src"
export OMNI_KIT_ACCEPT_EULA=YES
CONDA_ENVS="${CONDA_ENVS:-$HOME/miniconda3/envs}"
PY="${SIM_PY:-$CONDA_ENVS/airseai/bin/python}"
PORT="${PORT:-5555}"

COMMON=(--task "$TASK" --headless --enable_cameras --episodes "$EP" --seed "$SEED" --out "$OUT")

if [ "$CKPT" = hold ] || [ "${INPROCESS:-0}" = 1 ]; then
    exec "$PY" benchmark/eval/evaluate.py "${COMMON[@]}" --policy "$CKPT" \
        ${VID_ARGS[@]+"${VID_ARGS[@]}"}
fi

[ -f "$CKPT/config.json" ] || { echo "not a lerobot checkpoint dir: $CKPT"; exit 1; }
SLOG=$(mktemp /tmp/policy_server.XXXX.log)
"$PY" benchmark/policy/policy_server.py --policy "$CKPT" --port "$PORT" > "$SLOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
for _ in $(seq 1 90); do grep -q listening "$SLOG" && break; sleep 2; done
grep -q listening "$SLOG" || { echo "policy server failed:"; tail -5 "$SLOG"; exit 1; }

"$PY" benchmark/eval/evaluate.py "${COMMON[@]}" --policy-server "127.0.0.1:$PORT" \
    ${VID_ARGS[@]+"${VID_ARGS[@]}"}
echo "report: $OUT/eval_report.json"
