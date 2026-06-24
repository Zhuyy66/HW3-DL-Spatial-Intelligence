#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [gpu_id]"
  exit 1
fi

GPU_ID=${1:-5}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
HF_HOME_DIR=${HF_HOME_DIR:-"$REPO_ROOT/hf_home"}
TRIAL_NAME=${TRIAL_NAME:-stable-zero123-objectC}
TRIAL_TAG=${TRIAL_TAG:-zombie_20260616}
WANDB_PROJECT=${WANDB_PROJECT:-hw3-topic1}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-stable_zero123_object_C_gpu${GPU_ID}}
WANDB_ENABLE=${WANDB_ENABLE:-false}
IMAGE_PATH=${IMAGE_PATH:-"$REPO_ROOT/topic1_fusion/data/object_C/input_rgba.png"}
MAX_STEPS=${MAX_STEPS:-600}
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-100}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-100}
DATA_HEIGHT=${DATA_HEIGHT:-[128,128,128]}
DATA_WIDTH=${DATA_WIDTH:-[128,128,128]}
DATA_RESOLUTION_MILESTONES=${DATA_RESOLUTION_MILESTONES:-[100000,100000]}
RANDOM_CAMERA_BATCH_SIZE=${RANDOM_CAMERA_BATCH_SIZE:-[1,1,1]}
RANDOM_CAMERA_HEIGHT=${RANDOM_CAMERA_HEIGHT:-[64,64,64]}
RANDOM_CAMERA_WIDTH=${RANDOM_CAMERA_WIDTH:-[64,64,64]}
RANDOM_CAMERA_RESOLUTION_MILESTONES=${RANDOM_CAMERA_RESOLUTION_MILESTONES:-[100000,100000]}
NUM_SAMPLES_PER_RAY=${NUM_SAMPLES_PER_RAY:-128}
RUN_EXPORT=${RUN_EXPORT:-true}

export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TRIAL_DIR="$REPO_ROOT/topic1_fusion/code/threestudio/outputs/${TRIAL_NAME}/${TRIAL_TAG}"

PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/env_hw3_gen3d/bin/python}

cd "$REPO_ROOT/topic1_fusion/code/threestudio"

"$PYTHON_BIN" -u launch.py \
  --config configs/stable-zero123.yaml \
  --train \
  --gpu 0 \
  use_timestamp=False \
  name="$TRIAL_NAME" \
  tag="$TRIAL_TAG" \
  data.image_path="$IMAGE_PATH" \
  data.default_elevation_deg=5.0 \
  data.default_azimuth_deg=0.0 \
  data.height="$DATA_HEIGHT" \
  data.width="$DATA_WIDTH" \
  data.resolution_milestones="$DATA_RESOLUTION_MILESTONES" \
  trainer.max_steps="$MAX_STEPS" \
  trainer.val_check_interval="$VAL_CHECK_INTERVAL" \
  checkpoint.every_n_train_steps="$CHECKPOINT_STEPS" \
  data.random_camera.batch_size="$RANDOM_CAMERA_BATCH_SIZE" \
  data.random_camera.height="$RANDOM_CAMERA_HEIGHT" \
  data.random_camera.width="$RANDOM_CAMERA_WIDTH" \
  data.random_camera.resolution_milestones="$RANDOM_CAMERA_RESOLUTION_MILESTONES" \
  system.renderer.num_samples_per_ray="$NUM_SAMPLES_PER_RAY" \
  system.loss.lambda_normal_smooth=0.0 \
  system.loss.lambda_3d_normal_smooth=0.0 \
  system.loggers.wandb.enable="$WANDB_ENABLE" \
  system.loggers.wandb.project="$WANDB_PROJECT" \
  system.loggers.wandb.name="$WANDB_RUN_NAME"

if [[ "$RUN_EXPORT" == "true" ]]; then
  "$PYTHON_BIN" -u launch.py \
    --config "$TRIAL_DIR/configs/parsed.yaml" \
    --export \
    --gpu 0 \
    resume="$TRIAL_DIR/ckpts/last.ckpt" \
    system.exporter_type=mesh-exporter \
    system.geometry.isosurface_threshold=10.
fi
