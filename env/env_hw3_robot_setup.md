# env_hw3_robot Setup Notes

This file records the GPU-ready Day 1 server setup for member B. Run these
commands on the Linux GPU server, not on Windows.

## Assumptions

- Server project root: `/root/Test/Zhr/DL/HW3`
- Server driver is `535.129.03`; use PyTorch CUDA 11.8 wheels instead of
  CUDA 12.4 wheels to avoid CUDA error 804 on this driver.
- The current broken `env_hw3_robot` can be deleted and rebuilt.
- Course dataset link from the assignment PDF: `huiwon/calvin_task_ABC_D`
- Important foreground commands should save logs with `2>&1 | tee logs/name.log`.
- Default logs use stable filenames and are overwritten on each rerun. Copy a
  log into `logs/archive/` with a version suffix only when a historical backup
  is needed.
- Source `scripts/activate_cuda_driver_shim.sh` before CUDA checks. The server
  container can expose CUDA compat/stubs paths that trigger CUDA error 804 even
  when the PyTorch wheel version is compatible.

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

## CUDA Driver Shim

Run this after activating `env_hw3_robot` and before any PyTorch CUDA command.
The optional argument selects the physical GPU exposed as logical device 0.
Member B should use GPU 6 by default unless it is occupied.

```bash
cd /root/Test/Zhr/DL/HW3
conda activate env_hw3_robot
source scripts/activate_cuda_driver_shim.sh 6 \
  > >(tee logs/day1_cuda_driver_shim.log) 2>&1
```

If the script cannot locate the real driver, find it manually and rerun:

```bash
find /usr /lib /run/nvidia/driver \
  -path '*/compat/*' -prune -o \
  -path '*/stubs/*' -prune -o \
  -type f -name 'libcuda.so.*' -print 2>/dev/null

export REAL_CUDA_DRIVER=/path/to/libcuda.so.535.129.03
source scripts/activate_cuda_driver_shim.sh 6 \
  > >(tee logs/day1_cuda_driver_shim.log) 2>&1
```

## Clean Rebuild env_hw3_robot

Use a clean rebuild to remove the mixed conda CPU PyTorch and pip CUDA 12.4
state from earlier attempts. Keep LeRobot at `0.4.0` and TorchCodec at `0.2.1`;
install PyTorch with CUDA 11.8 wheels so the stack can run on the current 535
driver.

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs

LOG=logs/day1_rebuild_env_hw3_robot.log
{
  set -eo pipefail

  source /opt/conda/etc/profile.d/conda.sh
  source ~/.bashrc || true

  conda activate base
  conda env remove -n env_hw3_robot -y || true
  conda create -y -n env_hw3_robot python=3.12
  conda activate env_hw3_robot

  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  cat > "$CONDA_PREFIX/etc/conda/activate.d/hw3_paths.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

  conda install -y -c conda-forge ffmpeg=6 libstdcxx-ng libgcc-ng

  python -m pip install -U pip
  python -m pip install --force-reinstall \
    torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

  python -m pip install \
    "torchcodec==0.2.1" \
    "lerobot==0.4.0" \
    "wandb" "datasets" "pandas" "pyarrow" "huggingface_hub"

  python -m pip check
} 2>&1 | tee "$LOG"
```

## Verification

If `logs/day1_rebuild_env_hw3_robot.log` already ends with
`No broken requirements found.`, do not rebuild again. Source the CUDA driver
shim and rerun only the verification commands below.

```bash
conda activate env_hw3_robot
cd /root/Test/Zhr/DL/HW3
source scripts/activate_cuda_driver_shim.sh 6 \
  > >(tee logs/day1_cuda_driver_shim.log) 2>&1

python - <<'PY' 2>&1 | tee logs/day1_torch_build_check.log
import os
import torch
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device0", torch.cuda.get_device_name(0))
PY

python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/day1_verify_act_import.log

python topic2_act/scripts/probe_calvin_dataset.py \
  --repo-id huiwon/calvin_task_ABC_D \
  --endpoint https://hf-mirror.com \
  --local-dir /root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_probe \
  --max-meta-files 50 \
  2>&1 | tee logs/day1_probe_calvin_dataset.log
```

Pass criteria:

- `pip check` reports no broken requirements
- `torch` is `2.6.0+cu118`
- `torch.version.cuda` is not `None`
- `torch.cuda.is_available: True`
- `day1_cuda_driver_shim.log` shows a non-compat `libcuda.so.1` path
- `ACTPolicy imported`
- `torchcodec` imports
- `ffmpeg -version` succeeds
- Dataset probe downloads metadata and one parquet sample
