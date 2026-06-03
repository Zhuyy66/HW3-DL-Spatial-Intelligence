# env_hw3_robot Setup Notes

This file records the GPU-ready Day 1 server setup for member B. Run these
commands on the Linux GPU server, not on Windows.

## Assumptions

- Server project root: `/root/Test/Zhr/DL/HW3`
- Server driver exposes CUDA 12.4.
- The current broken `env_hw3_robot` can be removed.
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

## Rebuild env_hw3_robot

Use a CUDA 12.4-compatible stack. LeRobot is pinned to `0.4.0` so that it can
run with PyTorch 2.6 on the current server driver. TorchCodec is pinned to the
0.2 series, which the official compatibility table maps to PyTorch 2.6.

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs

LOG=logs/day1_rebuild_env_hw3_robot_$(date +%Y%m%d_%H%M%S).log
{
  set -euo pipefail
  source ~/.bashrc || true

  conda deactivate || true
  conda env remove -n env_hw3_robot -y || true
  conda create -y -n env_hw3_robot python=3.12
  conda activate env_hw3_robot

  conda install -y -c conda-forge ffmpeg=6 libstdcxx-ng libgcc-ng
  conda install -y pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia

  python -m pip install -U pip
  python -m pip install \
    "torchcodec==0.2.0" \
    "lerobot==0.4.0" \
    "wandb" "datasets" "pandas" "pyarrow" "huggingface_hub"

  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  cat > "$CONDA_PREFIX/etc/conda/activate.d/hw3_paths.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
} 2>&1 | tee "$LOG"
```

If conda cannot solve the PyTorch stack, keep the environment active and install
the CUDA 12.4 wheels with pip:

```bash
python -m pip install --force-reinstall \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 \
  2>&1 | tee logs/day1_pip_torch_cu124_$(date +%Y%m%d_%H%M%S).log
```

## Verification

```bash
conda activate env_hw3_robot
cd /root/Test/Zhr/DL/HW3

python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/day1_verify_act_import_$(date +%Y%m%d_%H%M%S).log

python topic2_act/scripts/probe_calvin_dataset.py \
  --repo-id huiwon/calvin_task_ABC_D \
  --local-dir /root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_probe \
  --max-meta-files 50 \
  2>&1 | tee logs/day1_probe_calvin_dataset_$(date +%Y%m%d_%H%M%S).log
```

Pass criteria:

- `torch.cuda.is_available: True`
- `ACTPolicy imported`
- `torchcodec` imports
- `ffmpeg -version` succeeds
- Dataset probe downloads metadata and one parquet sample
