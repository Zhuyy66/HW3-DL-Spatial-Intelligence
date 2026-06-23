#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <counter_source_dir> <output_dir> [gpu_id]"
  echo "Example: $0 /root/HW3/topic1_fusion/data/mipnerf360/counter /root/HW3/topic1_fusion/outputs/counter_7k 3"
  exit 1
fi

SOURCE_DIR=$1
OUTPUT_DIR=$2
GPU_ID=${3:-0}

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR"
  exit 1
fi

if [[ ! -d "$SOURCE_DIR/images" ]]; then
  echo "Expected images directory missing: $SOURCE_DIR/images"
  exit 1
fi

if [[ ! -d "$SOURCE_DIR/sparse" && ! -d "$SOURCE_DIR/distorted/sparse" ]]; then
  echo "Expected COLMAP sparse directory missing under: $SOURCE_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd /root/HW3/topic1_fusion/code/gaussian-splatting

OPTIMIZER_TYPE="default"
if CUDA_VISIBLE_DEVICES="$GPU_ID" python - <<'PY' >/dev/null 2>&1
from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
PY
then
  OPTIMIZER_TYPE="sparse_adam"
fi

echo "Using optimizer_type=$OPTIMIZER_TYPE"

CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
  -s "$SOURCE_DIR" \
  -m "$OUTPUT_DIR" \
  --eval \
  --iterations 7000 \
  --test_iterations 7000 \
  --save_iterations 7000 \
  --optimizer_type "$OPTIMIZER_TYPE" \
  --antialiasing \
  --wandb_project hw3-topic1 \
  --wandb_run_name "counter_7k_gpu${GPU_ID}"

echo "Training complete. Recommended follow-up:"
echo "  CUDA_VISIBLE_DEVICES=$GPU_ID python render.py -m $OUTPUT_DIR --iteration 7000 --skip_train"
