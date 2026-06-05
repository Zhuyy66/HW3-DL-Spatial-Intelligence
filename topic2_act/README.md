# Topic 2: LeRobot ACT on CALVIN

Member B owns this track.

## Day 1 Checklist

- Environment: `env_hw3_robot`, Python 3.12.
- GPU-ready stack: `torch==2.6.0+cu118`, `torchvision==0.21.0+cu118`,
  `torchaudio==2.6.0+cu118`, `torchcodec==0.2.1`, `lerobot==0.4.0`.
- Current known failure mode: CUDA 12.4 PyTorch wheels trigger CUDA error 804
  on the server's 535 driver. Use the CUDA 11.8 wheel rebuild from
  `env/env_hw3_robot_setup.md`.
- If CUDA error 804 persists with `cu118`, source the project driver shim before
  verification so PyTorch loads the real host `libcuda.so.1` instead of CUDA
  compat/stubs libraries:

```bash
source scripts/activate_cuda_driver_shim.sh 6 \
  > >(tee logs/day1_cuda_driver_shim.log) 2>&1
```

- Torch build check:

```bash
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
```

- Verification command:

```bash
python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/day1_verify_act_import.log
```

## Dataset Plan

The assignment PDF points to:

```text
https://huggingface.co/datasets/huiwon/calvin_task_ABC_D/tree/main
```

Do not start by downloading the full dataset. First run the probe script to
download metadata and one parquet episode, then inspect:

- environment label source, if present;
- `observation.state` shape and meaning;
- `action` shape and whether it matches CALVIN evaluation action space;
- camera keys and image shapes;
- episode/task index mapping.

Probe command:

```bash
python topic2_act/scripts/probe_calvin_dataset.py \
  --repo-id huiwon/calvin_task_ABC_D \
  --endpoint https://hf-mirror.com \
  --revision main \
  --local-dir /root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_probe \
  --max-meta-files 50 \
  2>&1 | tee logs/day1_probe_calvin_dataset.log
```

## Logging Rule

Use `2>&1 | tee logs/name.log` for setup, verification, and dataset probes,
because these commands should remain visible in the terminal while preserving
logs for later review. Use stable log filenames and let reruns overwrite the
previous output. If a previous log must be retained, copy it before rerunning:

```bash
mkdir -p logs/archive
cp logs/day1_verify_act_import.log logs/archive/day1_verify_act_import_v1.log
```

Use `> logs/name.log 2>&1 &` only for long background jobs such as training.

## Day 2 Output Target

After the probe passes on the server, write a cleaned data audit summary that
records the schema, episode split strategy, and any blockers before starting
full dataset downloads.
