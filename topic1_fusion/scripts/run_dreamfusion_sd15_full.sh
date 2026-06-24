#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [gpu_id]"
  exit 1
fi

GPU_ID=${1:-3}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
HF_HOME_DIR=${HF_HOME_DIR:-"$REPO_ROOT/hf_home"}
TRIAL_NAME=${TRIAL_NAME:-dreamfusion-sd15-full}
TRIAL_TAG=${TRIAL_TAG:-hamburger_full_20260603}
WANDB_PROJECT=${WANDB_PROJECT:-hw3-topic1}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-dreamfusion_sd15_full_gpu${GPU_ID}}
SD15_SNAPSHOT=${SD15_SNAPSHOT:-"$HF_HOME_DIR/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"}

export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-online}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

TRIAL_DIR="$REPO_ROOT/topic1_fusion/code/threestudio/outputs/${TRIAL_NAME}/${TRIAL_TAG}"

cd "$REPO_ROOT/topic1_fusion/code/threestudio"

conda run -n env_hw3_gen3d python launch.py \
  --config configs/dreamfusion-sd.yaml \
  --train \
  --gpu 0 \
  use_timestamp=False \
  name="$TRIAL_NAME" \
  tag="$TRIAL_TAG" \
  trainer.max_steps=10000 \
  trainer.val_check_interval=1000 \
  checkpoint.every_n_train_steps=1000 \
  data.width=64 \
  data.height=64 \
  data.batch_size=1 \
  system.prompt_processor.prompt="a hamburger" \
  system.prompt_processor.pretrained_model_name_or_path="$SD15_SNAPSHOT" \
  system.guidance.pretrained_model_name_or_path="$SD15_SNAPSHOT" \
  system.guidance.trainer_max_steps=10000 \
  system.loggers.wandb.enable=true \
  system.loggers.wandb.project="$WANDB_PROJECT" \
  system.loggers.wandb.name="$WANDB_RUN_NAME"

conda run -n env_hw3_gen3d python launch.py \
  --config "$TRIAL_DIR/configs/parsed.yaml" \
  --export \
  --gpu 0 \
  resume="$TRIAL_DIR/ckpts/last.ckpt" \
  system.exporter_type=mesh-exporter \
  system.geometry.isosurface_threshold=10.
