#!/usr/bin/env bash
set -euo pipefail

HF_HOME_DIR=${HF_HOME_DIR:-/root/HW3/hf_home}
ZERO123_DIR=/root/HW3/topic1_fusion/code/threestudio/load/zero123

mkdir -p "$HF_HOME_DIR" "$ZERO123_DIR"

export HF_HOME="$HF_HOME_DIR"

conda run -n env_hw3_gen3d python /tmp/hf_download_threestudio_models.py

echo "Model cache ready under $HF_HOME_DIR"
