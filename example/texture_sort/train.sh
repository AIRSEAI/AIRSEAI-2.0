#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Train ACT on all episodes of a converted texture-sort dataset.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATASET="${DATASET:-texture_sort_pairs_s41_p250}"
export VARIANT="${VARIANT:?set VARIANT=vision|tacrgb|tacff}"
export POLICY=act
export STEPS="${STEPS:-60000}"
export SEED="${SEED:-1000}"
export BATCH="${BATCH:-8}"
export WORKERS="${WORKERS:-8}"
export SAVE_FREQ="${SAVE_FREQ:-20000}"
export LOG_FREQ="${LOG_FREQ:-200}"

exec bash "$REPO_DIR/scripts/run_train.sh" \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.use_amp=false \
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=3200 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.n_vae_encoder_layers=4 \
  --policy.dropout=0.1 \
  --policy.kl_weight=10.0 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.optimizer_weight_decay=1e-4 "$@"
