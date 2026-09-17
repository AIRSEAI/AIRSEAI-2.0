#!/bin/bash
# Stage 2 — collect demonstrations (Isaac Sim, airseai env).
#
#   bash scripts/run_collect.sh <task> [episodes] [dataset_out]
#
# Inputs
#   $1  task        one of the TASKS registry in benchmark/run/collect_dataset.py:
#                   stack_cube | blind_peg_insertion | blind_peg_dim |
#                   texture_sort | sleeve_on_shaft | handover
#   $2  episodes    how many demonstrations to record   (default 50)
#   $3  dataset_out where to write the dataset          (default ~/<task>_dataset)
#   RENDER=1        also write the verification video   (<out>_run/<task>.mp4)
#   PREVIEWS=1      also derive the human-viewable tactile preview streams
#
# Output: a LeRobot-v3-style dataset (tactile RGB + native force field) at
# dataset_out, and a run directory beside it (<dataset_out>_run) holding the
# per-episode report and any video.
#
# Task knobs (light level, clearance, ...) are fixed by the task's registry
# entry, not chosen here.
set -eu
TASK="${1:?usage: run_collect.sh <task> [episodes] [dataset_out]}"
EP="${2:-50}"
OUT="${3:-$HOME/${TASK}_dataset}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PWD/src"
export OMNI_KIT_ACCEPT_EULA=YES
CONDA_ENVS="${CONDA_ENVS:-$HOME/miniconda3/envs}"
PY="${SIM_PY:-$CONDA_ENVS/airseai/bin/python}"

ARGS=(--task "$TASK" --episodes "$EP" --dataset "$OUT")
[ "${RENDER:-0}"   = 1 ] && ARGS+=(--render)
[ "${PREVIEWS:-0}" = 1 ] && ARGS+=(--previews)

echo "collect: task=$TASK episodes=$EP -> $OUT"
exec "$PY" benchmark/run/collect_dataset.py "${ARGS[@]}"
