#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <model_dir> <source_dir> [gpu_id]"
  echo "Example: $0 /root/HW3/topic1_fusion/pretrained/garden/model /root/HW3/topic1_fusion/data/mipnerf360/garden 3"
  exit 1
fi

MODEL_DIR=$1
SOURCE_DIR=$2
GPU_ID=${3:-0}

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR"
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR"
  exit 1
fi

cd /root/HW3/topic1_fusion/code/gaussian-splatting

CUDA_VISIBLE_DEVICES="$GPU_ID" python render.py \
  -m "$MODEL_DIR" \
  -s "$SOURCE_DIR" \
  --iteration -1 \
  --skip_train

echo "Render complete. Check:"
echo "  $MODEL_DIR/test"
