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

## Day 3 Dataset Plan

The production data path is now the official course split dataset:

```text
https://huggingface.co/datasets/xiaoma26/calvin-lerobot
```

The previous `scene_info.npy` reverse-splitting path has been moved to
`topic2_act/legacy/scene_info_split/` and is retained only as cross-validation
evidence. Use official `splitA` for A-only training.

Official split counts:

| split | scene | episodes | frames | fps |
| --- | --- | ---: | ---: | ---: |
| splitA | A | 6089 | 366693 | 10 |
| splitB | B | 6115 | 367096 | 10 |
| splitC | C | 5666 | 337954 | 10 |
| splitD | D | 5124 | 308918 | 10 |

The official split schema uses `state`, `actions`, `image`, `wrist_image`, and
`task_index`. Do not reuse old `observation.state`, `action`, or
`observation.images.*` keys from the retired Day 2 shard.

The shareable split-definition files are tracked in:

```text
topic2_act/dataset_split/xiaoma26_calvin_lerobot/
```

The server-generated copy under `topic2_act/data/splits/` is kept as a local
run artifact and is ignored by git.

## Day 3 Server Commands

Run the setup checks first:

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot
source scripts/activate_cuda_driver_shim.sh 0 \
  > >(tee logs/day3_cuda_driver_shim.log) 2>&1

python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/day3_verify_act_import.log
```

Prepare full `splitA` plus deterministic A-smoke500 episode view:

```bash
cd /root/Test/Zhr/DL/HW3
nohup bash -lc '
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot
export HF_ENDPOINT=https://hf-mirror.com
python topic2_act/scripts/prepare_xiaoma_calvin_split.py \
  --repo-id xiaoma26/calvin-lerobot \
  --endpoint https://hf-mirror.com \
  --revision main \
  --local-dir /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot \
  --output-dir /root/Test/Zhr/DL/HW3/topic2_act/data/splits/xiaoma26_calvin_lerobot \
  --download-split splitA \
  --metadata-splits splitA splitB splitC splitD \
  --smoke-count 500 \
  --seed 20260606 \
  --manifest /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot_manifest.json
' > logs/day3_prepare_xiaoma_splitA.log 2>&1 &
echo $! > logs/day3_prepare_xiaoma_splitA.pid
tail -f logs/day3_prepare_xiaoma_splitA.log
```

After completion:

```bash
python - <<'PY' 2>&1 | tee logs/day3_check_split_outputs.log
import json
from pathlib import Path

root = Path("/root/Test/Zhr/DL/HW3/topic2_act/data/splits/xiaoma26_calvin_lerobot")
for name in ["episodes_A_full.json", "episodes_A_smoke500.json", "official_split_summary.json"]:
    path = root / name
    print(name, "exists=", path.exists(), "bytes=", path.stat().st_size if path.exists() else 0)
print("A_full", len(json.loads((root / "episodes_A_full.json").read_text())))
print("A_smoke500", len(json.loads((root / "episodes_A_smoke500.json").read_text())))
print((root / "official_split_summary.json").read_text()[:2000])
PY

du -sh /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot \
  2>&1 | tee logs/day3_data_size_check.log
```

Acceptance criteria:

- `day3_prepare_xiaoma_splitA.log` ends with `ok: official CALVIN split preparation completed`;
- `episodes_A_full.json` contains 6089 episode indices;
- `episodes_A_smoke500.json` contains 500 deterministic episode indices;
- `official_split_summary.json` confirms `A=6089`, `B=6115`, `C=5666`, `D=5124`;
- `topic2_act/docs/day3_data_audit.md` documents that the old reverse split is retired.
- tracked split definitions are available under `topic2_act/dataset_split/xiaoma26_calvin_lerobot/`.

## Retired Day 1/2 Dataset Probe

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
