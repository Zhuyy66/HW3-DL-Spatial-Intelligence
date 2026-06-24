#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
HF_HOME_DIR=${HF_HOME_DIR:-"$REPO_ROOT/hf_home"}
ZERO123_DIR="$REPO_ROOT/topic1_fusion/code/threestudio/load/zero123"

mkdir -p "$HF_HOME_DIR" "$ZERO123_DIR"

export HF_HOME="$HF_HOME_DIR"

conda run -n env_hw3_gen3d python /tmp/hf_download_threestudio_models.py

echo "Model cache ready under $HF_HOME_DIR"
