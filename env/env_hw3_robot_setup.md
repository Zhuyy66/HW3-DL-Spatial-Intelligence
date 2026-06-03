# env_hw3_robot Setup Notes

This file records the GPU-ready Day 1 server setup for member B. Run these
commands on the Linux GPU server, not on Windows.

## Assumptions

- Server project root: `/root/Test/Zhr/DL/HW3`
- Server driver exposes CUDA 12.4.
- The current broken `env_hw3_robot` should be repaired in place first.
- Course dataset link from the assignment PDF: `huiwon/calvin_task_ABC_D`
- Important foreground commands should save logs with `2>&1 | tee logs/name.log`.

## Environment Variables

Add the shared cache and mirror variables once:

```bash
cat >> ~/.bashrc <<'EOF'

# HW3 DL Spatial Intelligence
export HW3_ROOT=/root/Test/Zhr/DL/HW3
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HW3_ROOT/.hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$HW3_ROOT/.torch
export PIP_CACHE_DIR=$HW3_ROOT/.pip_cache
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
EOF
```

Then prepare cache directories:

```bash
source ~/.bashrc
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$HW3_ROOT/.conda_pkgs" "$HW3_ROOT/logs"
conda config --add pkgs_dirs "$HW3_ROOT/.conda_pkgs"
```

## Repair env_hw3_robot In Place

Use a CUDA 12.4-compatible stack. LeRobot is pinned to `0.4.0` so that it can
run with PyTorch 2.6 on the current server driver. TorchCodec is pinned to the
0.2 series, which the official compatibility table maps to PyTorch 2.6.

The first failed attempt installed the conda-forge CPU PyTorch build. The repair
below keeps the environment and replaces PyTorch with channel-qualified
`pytorch::` packages. It also avoids `set -u`, because conda activation scripts
may reference unset variables.

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs

LOG=logs/day1_repair_env_hw3_robot_$(date +%Y%m%d_%H%M%S).log
{
  set -eo pipefail

  source /opt/conda/etc/profile.d/conda.sh
  conda activate env_hw3_robot

  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  cat > "$CONDA_PREFIX/etc/conda/activate.d/hw3_paths.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

  conda install -y --override-channels \
    -c pytorch -c nvidia -c conda-forge \
    pytorch::pytorch==2.6.0 \
    pytorch::torchvision==0.21.0 \
    pytorch::torchaudio==2.6.0 \
    pytorch::pytorch-cuda=12.4

  conda install -y -c conda-forge ffmpeg=6 libstdcxx-ng libgcc-ng

  python -m pip install -U pip
  python -m pip install \
    "torchcodec==0.2.0" \
    "lerobot==0.4.0" \
    "wandb" "datasets" "pandas" "pyarrow" "huggingface_hub"
} 2>&1 | tee "$LOG"
```

If conda still selects a CPU build or fails to solve, keep the environment
active and install the CUDA 12.4 wheels with pip:

```bash
conda activate env_hw3_robot
cd /root/Test/Zhr/DL/HW3

python -m pip install --force-reinstall \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 \
  2>&1 | tee logs/day1_pip_torch_cu124_$(date +%Y%m%d_%H%M%S).log
```

## Clean Rebuild Fallback

Use this only if the repair path leaves the environment inconsistent. Run it
from `base`, not from inside `env_hw3_robot`.

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs

LOG=logs/day1_clean_rebuild_env_hw3_robot_$(date +%Y%m%d_%H%M%S).log
{
  set -eo pipefail

  source /opt/conda/etc/profile.d/conda.sh
  conda activate base
  conda env remove -n env_hw3_robot -y || true
  conda create -y -n env_hw3_robot python=3.12
  conda activate env_hw3_robot

  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  cat > "$CONDA_PREFIX/etc/conda/activate.d/hw3_paths.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

  conda install -y --override-channels \
    -c pytorch -c nvidia -c conda-forge \
    pytorch::pytorch==2.6.0 \
    pytorch::torchvision==0.21.0 \
    pytorch::torchaudio==2.6.0 \
    pytorch::pytorch-cuda=12.4 \
    ffmpeg=6 libstdcxx-ng libgcc-ng

  python -m pip install -U pip
  python -m pip install \
    "torchcodec==0.2.0" \
    "lerobot==0.4.0" \
    "wandb" "datasets" "pandas" "pyarrow" "huggingface_hub"
} 2>&1 | tee "$LOG"
```

## Verification

```bash
conda activate env_hw3_robot
cd /root/Test/Zhr/DL/HW3

python - <<'PY' 2>&1 | tee logs/day1_torch_build_check_$(date +%Y%m%d_%H%M%S).log
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device0", torch.cuda.get_device_name(0))
PY

python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/day1_verify_act_import_$(date +%Y%m%d_%H%M%S).log

python topic2_act/scripts/probe_calvin_dataset.py \
  --repo-id huiwon/calvin_task_ABC_D \
  --local-dir /root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_probe \
  --max-meta-files 50 \
  2>&1 | tee logs/day1_probe_calvin_dataset_$(date +%Y%m%d_%H%M%S).log
```

Pass criteria:

- `torch.version.cuda` is not `None`
- `torch.cuda.is_available: True`
- `ACTPolicy imported`
- `torchcodec` imports
- `ffmpeg -version` succeeds
- Dataset probe downloads metadata and one parquet sample
