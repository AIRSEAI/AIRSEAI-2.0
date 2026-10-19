#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Convert one or more compatible recordings into LeRobot input variants.
set -euo pipefail
[ "$#" -gt 0 ] || { echo "usage: run_convert.sh <raw_dataset_dir> [...]" >&2; exit 64; }
RAW=()
for input in "$@"; do
    [ -d "$input" ] || { echo "raw dataset not found: $input" >&2; exit 1; }
    RAW+=("$(cd "$input" && pwd -P)")
done
if [ "$#" -gt 1 ] && [ -z "${DATASET:-}" ]; then
    echo "set DATASET for multiple input recordings" >&2; exit 64
fi
NAME="${DATASET:-$(basename "${RAW[0]}")}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs}"
BASE="${BASE:-$OUTPUT_ROOT/datasets}"
export PYTHONPATH="$PWD/src${AIRSEAI_LEROBOT_DIR:+:$AIRSEAI_LEROBOT_DIR/src}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
PY="${DEFAULT_PY:-python}"
VARIANTS="${VARIANTS:-vision tacff tacrgb}"
for variant in $VARIANTS; do
    case "$variant" in vision|tacff|tacrgb) ;; *) echo "unknown variant: $variant" >&2; exit 64 ;; esac
done
ARGS=()
[ "${FORCE:-0}" != 1 ] || ARGS+=(--overwrite)
for variant in $VARIANTS; do
    "$PY" benchmark/policy/convert_dataset.py --src "${RAW[@]}" \
        --dst-root "$BASE" --name "$NAME" --variant "$variant" ${ARGS[@]+"${ARGS[@]}"}
done
