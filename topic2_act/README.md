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

## Day 4 ACT Training Runbook

Day 4 uses a two-stage A-only schedule:

- Stage 1 is an engineering baseline: `episodes_A_smoke500.json`, 50 epochs,
  batch size 8. It verifies the full training chain, loss behavior,
  checkpointing, WandB logging, and real wall-clock cost. Do not use this model
  as the final A-only vs ABC comparison checkpoint.
- Stage 2 is the final full-data A-only run: `episodes_A_full.json`, with the
  same epoch budget that will later be used for full ABC training. Choose that
  shared epoch budget only after Stage 1 gives a measured seconds-per-epoch
  estimate.

The official xiaoma split is LeRobot v2.1 and uses `state`, `actions`, `image`,
and `wrist_image`. `topic2_act/scripts/run_act_train.py` materializes a
canonical selected subset before training:

```text
state       -> observation.state
actions     -> action
image       -> observation.images.image
wrist_image -> observation.images.wrist_image
```

The canonical subset is cached beside the source split and converted to
LeRobot v3.0 with the installed LeRobot converter. Reuse the cached subset for
the 5-epoch smoke and the 50-epoch baseline unless the source split or episode
list changes.

### GPU Selection

Training startup must select the GPU from current `nvidia-smi`, not from an old
snapshot. Keep batch size 8 if any 24GB candidate card has enough free memory;
only use GPU 3/5 if the 24GB cards cannot preserve batch 8; only downgrade
batch size if no allowed card can hold batch 8. If batch is downgraded, record it
as a global A-only/ABC comparison constraint.

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs/Day4 topic2_act/outputs/act_calvin
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot

python topic2_act/scripts/select_training_gpu.py \
  --candidate-gpus 0,1,2,4,6,7 \
  --large-gpus 3,5 \
  --batch-size 8 \
  --min-free-mib 9000 \
  --allow-large-gpus \
  --downgrade-batches 4,2 \
  --write-env logs/Day4/day4_gpu.env \
  2>&1 | tee logs/Day4/day4_select_gpu.log

source logs/Day4/day4_gpu.env
source scripts/activate_cuda_driver_shim.sh "$HW3_GPU_ID" \
  > >(tee logs/Day4/day4_cuda_driver_shim.log) 2>&1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/Day4/day4_verify_act_import.log

wandb status 2>&1 | tee logs/Day4/day4_wandb_status.log
```

Recent WandB API keys may use the longer `wandb_v1_...` format. If the current
SDK rejects the key with `API key must be 40 characters long`, upgrade only the
WandB package first, then revalidate LeRobot:

```bash
python - <<'PY' 2>&1 | tee logs/Day4/day4_wandb_env_before_upgrade.log
import importlib.metadata as md
print("wandb_version=", md.version("wandb"))
print("lerobot_version=", md.version("lerobot"))
for req in md.requires("lerobot") or []:
    if "wandb" in req.lower():
        print("lerobot_wandb_requirement=", req)
PY

python -m pip freeze \
  > logs/Day4/day4_pip_freeze_before_wandb_upgrade.txt

python -m pip install -U "wandb>=0.27.0" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  2>&1 | tee logs/Day4/day4_wandb_upgrade_retry1.log

python - <<'PY' 2>&1 | tee logs/Day4/day4_wandb_env_after_upgrade_retry2.log
import importlib.metadata as md
print("wandb_version=", md.version("wandb"))
print("lerobot_version=", md.version("lerobot"))
PY

python -m pip check \
  2>&1 | tee logs/Day4/day4_pip_check_after_wandb_upgrade_retry2.log

source logs/Day4/day4_gpu.env
source scripts/activate_cuda_driver_shim.sh "$HW3_GPU_ID" \
  > >(tee logs/Day4/day4_cuda_driver_shim_after_wandb_upgrade.log) 2>&1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/Day4/day4_verify_act_import_after_wandb_upgrade_retry2.log
```

`pip check` may report `lerobot 0.4.0 has requirement wandb<0.22.0,>=0.20.0`.
Treat that as an acceptable warning only if it is the sole conflict and the
LeRobot/CUDA import smoke below passes. Stop if `pip check` reports conflicts
for `torch`, `torchvision`, `torchcodec`, or `accelerate`.

If the mirror upgrade fails because the package is unavailable, retry once with
the default pip index:

```bash
python -m pip install -U "wandb>=0.27.0" \
  2>&1 | tee logs/Day4/day4_wandb_upgrade_retry1_default_pypi.log
```

After the upgrade, enter the key without writing it to a log. The shape check
prints length and prefix only:

```bash
read -rsp "WandB API key: " WANDB_API_KEY
echo
export WANDB_API_KEY

python - <<'PY' 2>&1 | tee logs/Day4/day4_wandb_key_shape_retry3.log
import os
key = os.environ.get("WANDB_API_KEY", "")
print("WANDB_API_KEY_present=", bool(key))
print("WANDB_API_KEY_length=", len(key))
print("WANDB_API_KEY_has_wandb_v1_prefix=", key.startswith("wandb_v1_"))
if len(key) < 40:
    raise SystemExit("bad_key_length: too short")
PY
```

Prefer CLI login with verification, then prove the Python SDK can create a run
in the same no-tty server context:

```bash
wandb login --relogin --verify "$WANDB_API_KEY" \
  2>&1 | tee logs/Day4/day4_wandb_login_retry3.log

python - <<'PY' 2>&1 | tee logs/Day4/day4_wandb_probe_retry3.log
import os
from pathlib import Path
import wandb

print("wandb_version=", wandb.__version__)
print("HOME=", os.environ.get("HOME"))
print("WANDB_API_KEY_present=", bool(os.environ.get("WANDB_API_KEY")))
print("netrc_exists=", (Path.home() / ".netrc").exists())

run = wandb.init(
    project="hw3-topic2",
    name="day4_wandb_probe_retry3",
    job_type="preflight",
)
print("wandb_url=", run.url)
run.finish()
PY
```

If upgrade or CLI login fails but `WANDB_API_KEY` is set, try the SDK-only probe
without calling `wandb login`:

```bash
python - <<'PY' 2>&1 | tee logs/Day4/day4_wandb_env_only_probe_retry3.log
import os
import wandb

print("wandb_version=", wandb.__version__)
print("WANDB_API_KEY_present=", bool(os.environ.get("WANDB_API_KEY")))

run = wandb.init(
    project="hw3-topic2",
    name="day4_wandb_env_only_probe_retry3",
    job_type="preflight",
)
print("wandb_url=", run.url)
run.finish()
PY
```

If the SDK-only path still hits local key-length validation, use a private
temporary NETRC file. Do not print the file contents:

```bash
mkdir -p .secrets
chmod 700 .secrets
export NETRC=/root/Test/Zhr/DL/HW3/.secrets/wandb_netrc
umask 077

python - <<'PY'
import os
from pathlib import Path
key = os.environ["WANDB_API_KEY"]
netrc = Path(os.environ["NETRC"])
netrc.write_text(f"machine api.wandb.ai\n  login user\n  password {key}\n", encoding="utf-8")
netrc.chmod(0o600)
PY

python - <<'PY' 2>&1 | tee logs/Day4/day4_wandb_netrc_probe_retry2.log
import os
from pathlib import Path
import wandb

print("wandb_version=", wandb.__version__)
print("NETRC=", os.environ.get("NETRC"))
print("netrc_exists=", Path(os.environ["NETRC"]).exists())

run = wandb.init(
    project="hw3-topic2",
    name="day4_wandb_netrc_probe_retry2",
    job_type="preflight",
)
print("wandb_url=", run.url)
run.finish()
PY
```

Continue to ACT training only after one WandB probe log contains `wandb_url=`.

If the import smoke passes but dry run fails because `wandb 0.27.x` changed an
API used by LeRobot, downgrade to a lower long-key-compatible version and repeat
the import smoke plus WandB probe:

```bash
python -m pip install --force-reinstall "wandb==0.22.3" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  2>&1 | tee logs/Day4/day4_wandb_downgrade_to_0223.log
```

### Dry Run

Run two optimization batches before any long job. This catches dataset
conversion, feature-name, image decoding, CUDA, and backward-pass failures.

```bash
python topic2_act/scripts/run_act_train.py \
  --dataset-root /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot/splitA \
  --episodes-file /root/Test/Zhr/DL/HW3/topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_A_smoke500.json \
  --output-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_dryrun \
  --run-name act_calvin_a_only_smoke500_dryrun \
  --epochs 1 \
  --batch-size "$HW3_BATCH_SIZE" \
  --dry-run-batches 2 \
  --wandb-project hw3-topic2 \
  --rebuild-prepared-dataset \
  --overwrite \
  2>&1 | tee logs/Day4/day4_act_dryrun.log
```

If a previous conversion failed, remove the stale canonical cache and run
outputs before retrying:

```bash
rm -rf topic2_act/data/xiaoma26_calvin_lerobot/splitA_episodes_A_smoke500_canonical_v3*
rm -rf topic2_act/outputs/act_calvin/a_only_smoke500_dryrun
rm -rf topic2_act/outputs/act_calvin/a_only_smoke500_5ep
```

### Five-Epoch Smoke Test

```bash
nohup bash -lc '
set -eo pipefail
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot
source logs/Day4/day4_gpu.env
source scripts/activate_cuda_driver_shim.sh "$HW3_GPU_ID"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=hw3-topic2
python topic2_act/scripts/run_act_train.py \
  --dataset-root /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot/splitA \
  --episodes-file /root/Test/Zhr/DL/HW3/topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_A_smoke500.json \
  --output-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_5ep \
  --run-name act_calvin_a_only_smoke500_5ep \
  --epochs 5 \
  --batch-size "$HW3_BATCH_SIZE" \
  --wandb-project hw3-topic2
' > logs/Day4/day4_act_smoke5.log 2>&1 &
echo $! > logs/Day4/day4_act_smoke5.pid
tail -f logs/Day4/day4_act_smoke5.log
```

Smoke-test acceptance:

```bash
python topic2_act/scripts/summarize_act_run.py \
  --run-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_5ep \
  --log-file logs/Day4/day4_act_smoke5.log \
  --pid-file logs/Day4/day4_act_smoke5.pid \
  --require-checkpoint \
  --require-wandb \
  2>&1 | tee logs/Day4/day4_act_smoke5_summary.log
```

### Overnight 500-Episode Baseline

Start this only after the 5-epoch smoke summary is healthy.

```bash
nohup bash -lc '
set -eo pipefail
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot
source logs/Day4/day4_gpu.env
source scripts/activate_cuda_driver_shim.sh "$HW3_GPU_ID"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=hw3-topic2
python topic2_act/scripts/run_act_train.py \
  --dataset-root /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot/splitA \
  --episodes-file /root/Test/Zhr/DL/HW3/topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_A_smoke500.json \
  --output-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_50ep \
  --run-name act_calvin_a_only_smoke500_50ep \
  --epochs 50 \
  --batch-size "$HW3_BATCH_SIZE" \
  --wandb-project hw3-topic2
' > logs/Day4/day4_act_smoke500_50ep.log 2>&1 &
echo $! > logs/Day4/day4_act_smoke500_50ep.pid
tail -f logs/Day4/day4_act_smoke500_50ep.log
```

Check the first 10 epochs:

```bash
python topic2_act/scripts/summarize_act_run.py \
  --run-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_50ep \
  --log-file logs/Day4/day4_act_smoke500_50ep.log \
  --pid-file logs/Day4/day4_act_smoke500_50ep.pid \
  --min-epochs 10 \
  --require-healthy-loss \
  --require-checkpoint \
  --require-wandb \
  2>&1 | tee logs/Day4/day4_act_10epoch_health_check.log
```

If the 10-epoch verdict is `unhealthy`, stop the run and debug from logs rather
than wasting GPU time:

```bash
kill "$(cat logs/Day4/day4_act_smoke500_50ep.pid)"
```

Day 4 is complete when `a_only_smoke500_50ep` is running or completed and
`day4_act_10epoch_health_check.log` gives a healthy verdict. Commit and push
only after Day 4 completion is confirmed; stage narrowly around Topic2 scripts,
README, and any intentional small config/docs changes.

### Day 4 Troubleshooting

If a dry run log contains `run_act_train.py: error: the following arguments are
required`, the hidden LeRobot worker mode is being blocked by main-entry CLI
validation. Sync the latest `run_act_train.py`, remove the failed dry-run
output, and rerun with `--overwrite`; if the prepared dataset was created by an
older script version, also remove
`topic2_act/data/xiaoma26_calvin_lerobot/splitA_episodes_A_smoke500_canonical_v3*`
or pass `--rebuild-prepared-dataset`.

If LeRobot fails during `Accelerator(...)` with the RTX 4000-series P2P/IB
message, set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` before rerunning.
`run_act_train.py` also enforces these variables for its worker process and
records them in the run manifest.

If WandB fails with `API key must be 40 characters long`, the installed SDK is
too old for the longer `wandb_v1_...` key format. Upgrade `wandb>=0.27.0` and
rerun the LeRobot import smoke with the CUDA shim loaded before training. The
expected `lerobot 0.4.0 requires wandb<0.22.0` `pip check` conflict can be
accepted only if it is the sole conflict and runtime smoke tests pass. If WandB
fails with `api_key not configured (no-tty)`, do not rerun training until one of
`day4_wandb_probe_retry3.log`, `day4_wandb_env_only_probe_retry3.log`, or
`day4_wandb_netrc_probe_retry2.log` contains a `wandb_url=` line. The ordinary
`wandb login` message can be misleading if the worker process cannot see a
usable API key or `NETRC`.

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
