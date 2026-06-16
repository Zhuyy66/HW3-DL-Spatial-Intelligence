# Topic 2: LeRobot ACT on CALVIN

Member B owns this track.

## Week 1 Artifact Index

This section is the Day 7 handoff index for the Topic 2 Week 1 artifacts. It
records the actual production path after the course staff published the official
`xiaoma26/calvin-lerobot` split dataset.

| Artifact | Actual path | Status | Validation evidence |
| --- | --- | --- | --- |
| Official A split metadata | `topic2_act/dataset_split/xiaoma26_calvin_lerobot/` | Tracked small split definitions. Server-side data lives under ignored `topic2_act/data/`. | `logs/Day 3/day3_check_split_outputs.log` confirms `A_full=6089` and `A_smoke500=500`. |
| Production data audit | `topic2_act/docs/day3_data_audit.md` | Current production audit. | Confirms official split schema and counts for A/B/C/D. |
| Legacy reverse split audit | `topic2_act/data/data_audit.md` and `topic2_act/legacy/scene_info_split/scripts/split_env_a.py` | Retired training route, retained as cross-validation evidence. | Independent reverse audit matched the official split counts for A/B/C. |
| A-only ACT engineering baseline | `topic2_act/outputs/act_calvin/a_only_smoke500_50ep/lerobot_train/checkpoints/188300/pretrained_model/` | Ignored heavy checkpoint on the server/local artifact copy. This model validates the chain only. | `logs/Day4/day4_act_10epoch_health_check.log` and `logs/Day5/day5_a_only_smoke500_final_check.log` report `verdict: healthy`, `observed_epoch=50`, 250 checkpoints, and WandB run `wru9vt3x`. |
| ACT training wrapper | `topic2_act/scripts/run_act_train.py` | Tracked. Converts official split fields to canonical LeRobot v3.0 features, then launches LeRobot training. | Dry run, 5-epoch smoke, and 50-epoch baseline completed. |
| CALVIN ACT wrapper | `topic2_act/eval/lerobot_act_wrapper.py` | Tracked. Keeps CALVIN `reset()` / `step(obs, goal)` interface and delegates real ACT inference to a persistent Python 3.12 worker. | `logs/Day6/day6_bridge_fidelity_after_config_fix.log` has exact bridge fidelity. |
| Worker bridge | `topic2_act/eval/lerobot_act_worker.py` and `topic2_act/eval/bridge_protocol.py` | Tracked. Uses length-prefixed pickle protocol over stdio. | Day 6 bridge fidelity exact/allclose check passed. |
| CALVIN rollout launcher | `topic2_act/eval/run_calvin_eval.py` | Tracked. Runs the official CALVIN-side rollout path with the local wrapper. | `logs/Day6/day6_real_act_direct_camera_rollout_after_pickle_fix.log` reaches `single_rollout_smoke_result`, reports action shape `[7]`, and shows robot movement. |

Important interpretation notes:

- `a_only_smoke500_50ep` is a Week 1 engineering baseline for environment,
  data, training, checkpoint, WandB, and wrapper connectivity validation. The
  final A-only vs ABC comparison should use a planned comparable-budget run.
- The old `split_env_a.py` name refers to the retired `scene_info.npy` reverse
  split path. Keep it as evidence that independently derived A/B/C counts agree
  with the official split; do not use it as the production training route.
- Do not add compatibility wrappers for retired paths. For reproducible server
  commands, create a stable `last` symlink to the latest checkpoint directory.

### Day 7 Stable Checkpoint Pointer

Run this on the Linux GPU server after syncing the repository and model outputs:

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs/Day7

CKPT_REAL=$(find topic2_act/outputs/act_calvin/a_only_smoke500_50ep/lerobot_train/checkpoints \
  -mindepth 2 -maxdepth 2 -path '*/pretrained_model' -type d | sort | tail -n 1)
ln -sfn "$CKPT_REAL" topic2_act/outputs/act_calvin/a_only_smoke500_50ep/lerobot_train/checkpoints/last
{
  echo "CKPT_REAL=$CKPT_REAL"
  ls -lah topic2_act/outputs/act_calvin/a_only_smoke500_50ep/lerobot_train/checkpoints/last
} 2>&1 | tee logs/Day7/day7_ckpt_last_link.log
```

### Day 7 Revalidation Commands

Use `2>&1 | tee ...` for these short verification commands so the terminal
remains readable and logs are saved for local inspection.

Confirm the A-only smoke500 baseline:

```bash
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot

python topic2_act/scripts/summarize_act_run.py \
  --run-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_50ep \
  --log-file logs/Day4/day4_act_smoke500_50ep.log \
  --pid-file logs/Day4/day4_act_smoke500_50ep.pid \
  --min-epochs 50 \
  --require-healthy-loss \
  --require-checkpoint \
  --require-wandb \
  2>&1 | tee logs/Day7/day7_a_only_smoke500_final_check.log
```

Recheck ACT worker bridge fidelity:

```bash
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot
source scripts/activate_cuda_driver_shim.sh 4 \
  > >(tee logs/Day7/day7_robot_shim.log) 2>&1

python topic2_act/eval/run_bridge_fidelity.py \
  --checkpoint /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_50ep/lerobot_train/checkpoints/last \
  --dataset-root /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot/splitA_episodes_A_smoke500_canonical_v3 \
  --worker-python /opt/conda/envs/env_hw3_robot/bin/python \
  --device cuda:0 \
  2>&1 | tee logs/Day7/day7_bridge_fidelity.log
```

Run one CALVIN direct-cameras smoke with the real ACT worker:

```bash
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_calvin_eval
source scripts/activate_cuda_driver_shim.sh 4 \
  > >(tee logs/Day7/day7_calvin_shim.log) 2>&1

python topic2_act/eval/run_calvin_eval.py \
  --calvin-root /root/Test/Zhr/DL/HW3/topic2_act/calvin_official \
  --dataset-path /root/Test/Zhr/DL/HW3/topic2_act/calvin_official/dataset/calvin_debug_dataset \
  --checkpoint /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_50ep/lerobot_train/checkpoints/last \
  --worker-python /opt/conda/envs/env_hw3_robot/bin/python \
  --worker-device cuda:0 \
  --worker-log logs/Day7/day7_worker_real_act.log \
  --eval-log-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/calvin_eval/day7_real_act_direct_cameras \
  --cuda-device 0 \
  --egl-policy direct-cameras \
  --single-rollout-smoke \
  --rollout-steps 60 \
  2>&1 | tee logs/Day7/day7_real_act_direct_camera_rollout.log
```

Acceptance:

- `day7_a_only_smoke500_final_check.log` reports `verdict: healthy`,
  `observed_epoch=50`, checkpoint count greater than zero, and a WandB URL.
- `day7_bridge_fidelity.log` reports exact/allclose bridge equivalence.
- `day7_real_act_direct_camera_rollout.log` reaches
  `single_rollout_smoke_result`, returns action shape `[7]`, and shows movement
  evidence.

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

## Week 3 Day 1 DataLoader Benchmark Runbook

Goal: reduce the ACT training input bottleneck before full A-only and ABC
training. The current reference run has `data_s` around `0.31`, `updt_s`
around `0.075`, and manifest wall-clock `seconds_per_step_observed=0.397`.

Source check for LeRobot `0.4.0`:

- `TrainPipelineConfig.num_workers` is an exposed training config field, so
  `run_act_train.py --num-workers ...` reaches LeRobot directly.
- `prefetch_factor` is hard-coded in `lerobot_train.py` as
  `prefetch_factor=2 if cfg.num_workers > 0 else None`.
- `persistent_workers` is not exposed and is not passed into DataLoader.
- `run_act_train.py` applies a process-local DataLoader audit patch inside the
  worker process. The patch logs the final DataLoader parameters and stores
  them in `run_manifest.json` plus `dataloader_audit.jsonl`.

Acceptance:

- Each benchmark log must contain `dataloader_monkeypatch_audit` with the
  expected `num_workers`, `prefetch_factor`, and `persistent_workers` values.
- The worker-process check should show DataLoader subprocesses during a running
  benchmark.
- A config is accepted only if measured `data_s` drops from the previous
  `~0.31` reference and total step time is below `0.397 s/step`.
- If the best config remains above `0.3 s/step`, stop tuning and downshift the
  final fixed step budget according to the Week 3 plan.

### Setup and CUDA Check

Use `tee` for short setup commands so the terminal remains readable.

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs/Week3_Day1 topic2_act/outputs/act_calvin_dataloader
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot

python topic2_act/scripts/select_training_gpu.py \
  --candidate-gpus 0,1,2,4,6,7 \
  --large-gpus 3,5 \
  --batch-size 8 \
  --min-free-mib 9000 \
  --allow-large-gpus \
  --downgrade-batches 4,2 \
  --write-env logs/Week3_Day1/day1_gpu.env \
  2>&1 | tee logs/Week3_Day1/day1_select_gpu.log

source logs/Week3_Day1/day1_gpu.env
source scripts/activate_cuda_driver_shim.sh "$HW3_GPU_ID" \
  > >(tee logs/Week3_Day1/day1_cuda_driver_shim.log) 2>&1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/Week3_Day1/day1_verify_act_import.log

python - <<'PY' 2>&1 | tee logs/Week3_Day1/day1_lerobot_train_defaults.log
from lerobot.configs.train import TrainPipelineConfig
cfg = TrainPipelineConfig
print("TrainPipelineConfig.num_workers_default=", cfg.num_workers)
print("TrainPipelineConfig.batch_size_default=", cfg.batch_size)
print("TrainPipelineConfig.steps_default=", cfg.steps)
print("TrainPipelineConfig.log_freq_default=", cfg.log_freq)
print("TrainPipelineConfig.save_freq_default=", cfg.save_freq)
PY
```

Expected default anchor: LeRobot `0.4.0` uses `steps=100000` in
`TrainPipelineConfig`.

### Full SplitA Cache Probe

Run once if the full splitA canonical v3 cache is absent or stale. This command
is backgrounded because conversion can take a while.

```bash
nohup bash -lc '
set -eo pipefail
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_robot
source logs/Week3_Day1/day1_gpu.env
source scripts/activate_cuda_driver_shim.sh "$HW3_GPU_ID"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
python topic2_act/scripts/run_act_train.py \
  --dataset-root /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot/splitA \
  --episodes-file /root/Test/Zhr/DL/HW3/topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_A_full.json \
  --output-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin_dataloader/splitA_prep_probe \
  --run-name act_calvin_splitA_prep_probe \
  --epochs 1 \
  --dry-run-batches 2 \
  --batch-size "$HW3_BATCH_SIZE" \
  --num-workers 4 \
  --prefetch-factor 2 \
  --no-persistent-workers \
  --disable-wandb \
  --rebuild-prepared-dataset \
  --overwrite
' > logs/Week3_Day1/day1_splitA_prep_probe.log 2>&1 &
echo $! > logs/Week3_Day1/day1_splitA_prep_probe.pid
tail -f logs/Week3_Day1/day1_splitA_prep_probe.log
```

After it finishes:

```bash
python topic2_act/scripts/summarize_act_run.py \
  --run-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin_dataloader/splitA_prep_probe \
  --log-file logs/Week3_Day1/day1_splitA_prep_probe.log \
  --baseline-seconds-per-step 0.397 \
  2>&1 | tee logs/Week3_Day1/day1_splitA_prep_probe_summary.log
```

### Steps-Only Argument Preflight

Run this before the 2000-step sweep to confirm the wrapper accepts fixed-step
budgets without `--epochs`.

```bash
python - <<'PY' 2>&1 | tee logs/Week3_Day1/day1_argparse_steps_only_check.log
import sys
from pathlib import Path

sys.path.insert(0, str(Path("topic2_act/scripts").resolve()))
import run_act_train

sys.argv = [
    "run_act_train.py",
    "--dataset-root", "/tmp/dummy_dataset",
    "--episodes-file", "/tmp/dummy_episodes.json",
    "--output-dir", "/tmp/dummy_output",
    "--run-name", "argparse_steps_only",
    "--steps", "2000",
]
args = run_act_train.parse_args()
print("steps=", args.steps)
print("epochs=", args.epochs)
assert args.steps == 2000 and args.epochs is None
print("OK: steps-only parse accepted")
PY
```

### 2000-Step Benchmark Sweep

Start with a baseline equivalent to the previous settings, then compare worker
and prefetch candidates. Keep WandB disabled for these timing runs.

```bash
for spec in nw4_pf2_np nw8_pf4_p nw12_pf4_p nw16_pf4_p; do
  case "$spec" in
    nw4_pf2_np)  NW=4;  PF=2; PERSIST="--no-persistent-workers" ;;
    nw8_pf4_p)   NW=8;  PF=4; PERSIST="--persistent-workers" ;;
    nw12_pf4_p)  NW=12; PF=4; PERSIST="--persistent-workers" ;;
    nw16_pf4_p)  NW=16; PF=4; PERSIST="--persistent-workers" ;;
  esac

  nohup bash -lc "
  set -eo pipefail
  cd /root/Test/Zhr/DL/HW3
  source /opt/conda/etc/profile.d/conda.sh
  conda activate env_hw3_robot
  source logs/Week3_Day1/day1_gpu.env
  source scripts/activate_cuda_driver_shim.sh \"\$HW3_GPU_ID\"
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  python topic2_act/scripts/run_act_train.py \
    --dataset-root /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot/splitA \
    --episodes-file /root/Test/Zhr/DL/HW3/topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_A_full.json \
    --output-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin_dataloader/splitA_${spec}_2k_fix1 \
    --run-name act_calvin_splitA_${spec}_2k_fix1 \
    --steps 2000 \
    --batch-size \"\$HW3_BATCH_SIZE\" \
    --num-workers $NW \
    --prefetch-factor $PF \
    $PERSIST \
    --disable-wandb \
    --overwrite
  " > logs/Week3_Day1/day1_splitA_${spec}_2k_fix1.log 2>&1 &
  echo $! > logs/Week3_Day1/day1_splitA_${spec}_2k_fix1.pid
  echo "started $spec pid=$(cat logs/Week3_Day1/day1_splitA_${spec}_2k_fix1.pid)"
  sleep 20
  ps -eo pid,ppid,stat,cmd \
    | grep -E "run_act_train.py|DataLoader|pt_data_worker|python" \
    | grep -v grep \
    2>&1 | tee logs/Week3_Day1/day1_splitA_${spec}_2k_fix1_worker_ps.log
  wait "$(cat logs/Week3_Day1/day1_splitA_${spec}_2k_fix1.pid)"
done
```

Summarize every candidate:

```bash
for spec in nw4_pf2_np nw8_pf4_p nw12_pf4_p nw16_pf4_p; do
  python topic2_act/scripts/summarize_act_run.py \
    --run-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin_dataloader/splitA_${spec}_2k_fix1 \
    --log-file logs/Week3_Day1/day1_splitA_${spec}_2k_fix1.log \
    --baseline-seconds-per-step 0.397 \
    --io-window 20 \
    2>&1 | tee logs/Week3_Day1/day1_splitA_${spec}_2k_fix1_summary.log
done
```

Check the audit lines explicitly:

```bash
grep -H "dataloader_monkeypatch_audit" logs/Week3_Day1/day1_splitA_*_2k_fix1.log \
  2>&1 | tee logs/Week3_Day1/day1_dataloader_audit_lines_fix1.log
```

Completed benchmark results:

| Config | DataLoader settings | wall s/step | data_s | updt_s | wall speedup vs 0.397 |
|---|---|---:|---:|---:|---:|
| `nw4_pf2_np` | `num_workers=4`, `prefetch_factor=2`, `persistent_workers=False` | 0.4030 | 0.32345 | 0.07340 | 0.99x |
| `nw8_pf4_p` | `num_workers=8`, `prefetch_factor=4`, `persistent_workers=True` | 0.2110 | 0.13300 | 0.07095 | 1.88x |
| `nw12_pf4_p` | `num_workers=12`, `prefetch_factor=4`, `persistent_workers=True` | 0.1460 | 0.06930 | 0.06950 | 2.72x |
| `nw16_pf4_p` | `num_workers=16`, `prefetch_factor=4`, `persistent_workers=True` | 0.1135 | 0.04045 | 0.06580 | 3.50x |

Selected config for subsequent full-budget runs:

```bash
--num-workers 16 --prefetch-factor 4 --persistent-workers
```

The selected config reduced wall-clock time from the `0.397 s/step` reference
to `0.1135 s/step`, and reduced `data_s` from about `0.31` to `0.04045`.
All four fixed-step runs completed with `planned_steps=2000`,
`step_budget_source=steps`, and return code `0`. The audit log confirmed the
requested DataLoader parameters, and the worker `ps` logs showed 4 / 8 / 12 /
16 child worker processes respectively.

LeRobot `0.4.0` default anchor from the server environment:

- `TrainPipelineConfig.steps_default=100000`
- `TrainPipelineConfig.log_freq_default=200`
- `TrainPipelineConfig.save_freq_default=20000`

Decision: the best measured step time is below `0.15 s/step`, so the Day 1
`>0.3 s/step` risk fallback is not triggered. The Week 3 fixed-step budget can
be set as high as `150K` if the downstream schedule allows it, with A-only and
ABC kept at the exact same total number of gradient steps.

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

## Day 5 CALVIN Eval Scaffold

Day 5 uses the completed `a_only_smoke500_50ep` run as the A-only training
evidence for interface work. This is an engineering baseline for the evaluation
chain, not the final A-only model for the A-only vs ABC comparison. Keep the
full-data A-only and ABC runs in the Week 2 plan after the dataloader bottleneck,
fixed-step budget, and ABC split loading decisions are resolved.

The CALVIN evaluation path is intentionally a thin wrapper around the official
CALVIN implementation:

- official CALVIN clone: `topic2_act/calvin_official/`, ignored by git;
- pinned official commit: `fa03f01f19c65920e18cf37398a9ce859274af76`;
- local wrapper: `topic2_act/eval/lerobot_act_wrapper.py`;
- local launcher: `topic2_act/eval/run_calvin_eval.py`;
- local env verifier: `topic2_act/eval/verify_calvin_eval_env.py`.

References:

- CALVIN README: <https://github.com/mees/calvin>
- CALVIN EGL/CUDA mapping wrapper:
  <https://github.com/mees/calvin_env/blob/main/calvin_env/envs/play_lmp_wrapper.py>
- EGL device probe: <https://github.com/StanfordVL/egl_probe>

### Confirm A-only Baseline Evidence

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs/Day5 topic2_act/outputs/calvin_eval

python topic2_act/scripts/summarize_act_run.py \
  --run-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_50ep \
  --log-file logs/Day4/day4_act_smoke500_50ep.log \
  --pid-file logs/Day4/day4_act_smoke500_50ep.pid \
  --min-epochs 50 \
  --require-healthy-loss \
  --require-checkpoint \
  --require-wandb \
  2>&1 | tee logs/Day5/day5_a_only_smoke500_final_check.log
```

Acceptance:

- command exits 0;
- `verdict` is `healthy`;
- `observed_epoch` is `50`;
- `checkpoint_count` is non-zero and the WandB URL is present.

### Create or Reuse `env_hw3_calvin_eval`

Run these commands on the Linux GPU server. If `apt-get` is unavailable in the
server container, stop and record that blocker instead of silently skipping EGL
system dependencies.

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs/Day5

apt-get update 2>&1 | tee logs/Day5/day5_apt_update.log
apt-get install -y libegl1 libgl1 libosmesa6-dev libglfw3 libgles2-mesa-dev patchelf mesa-utils \
  2>&1 | tee logs/Day5/day5_apt_egl_deps.log

source /opt/conda/etc/profile.d/conda.sh
conda env list | tee logs/Day5/day5_conda_env_list_before.log
conda create -y -n env_hw3_calvin_eval python=3.8 \
  2>&1 | tee logs/Day5/day5_create_env_hw3_calvin_eval.log
conda activate env_hw3_calvin_eval

python -m pip install -U pip wheel "setuptools==57.5.0" "cmake==3.18.4.post1" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  2>&1 | tee logs/Day5/day5_calvin_eval_pip_bootstrap.log
```

If this environment already exists from a failed attempt, reuse it and install
the fixed dependency set below instead of deleting the env.

### Repair and Install Official CALVIN

The official CALVIN tree may be copied from Windows when the server cannot
clone from GitHub. Before running any official shell script on Linux, remove
CRLF line endings and confirm that recursive submodules, especially
`calvin_env/tacto`, are present.

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs/Day5

if [ ! -d topic2_act/calvin_official/.git ]; then
  git clone --recurse-submodules https://github.com/mees/calvin.git topic2_act/calvin_official
fi

git -C topic2_act/calvin_official checkout fa03f01f19c65920e18cf37398a9ce859274af76
git -C topic2_act/calvin_official rev-parse HEAD \
  2>&1 | tee logs/Day5/day5_fix_calvin_head.log

find topic2_act/calvin_official -type f \( -name '*.sh' -o -name '*.py' -o -name '*.yaml' -o -name '*.yml' \) -print0 \
  | xargs -0 sed -i 's/\r$//'

test -d topic2_act/calvin_official/calvin_env/tacto \
  && echo "[OK] tacto submodule directory exists" \
  || echo "[ERROR] tacto submodule directory missing; recopy full local topic2_act/calvin_official including submodules"

git -C topic2_act/calvin_official submodule status --recursive \
  2>&1 | tee logs/Day5/day5_fix_calvin_submodule_status.log
```

If `tacto submodule directory missing` appears, recopy the full local
`topic2_act/calvin_official/` directory, including submodules, to the same
server path and rerun the repair block. Do not continue to install until
`calvin_env/tacto` exists.

Do not run the upstream `sh install.sh` after a failed attempt. It is fragile in
this server context because it can inherit CRLF line endings, pins
`cmake==3.18.4` while the mirror serves `3.18.4.post1`, and continues after
directory failures. Install the Day 5 eval smoke dependencies explicitly:

```bash
cd /root/Test/Zhr/DL/HW3

source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_calvin_eval

python -m pip install -U pip wheel "setuptools==57.5.0" "cmake==3.18.4.post1" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  2>&1 | tee logs/Day5/day5_fix_pip_bootstrap.log

set -o pipefail
python -m pip install "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  2>&1 | tee logs/Day5/day5_fix_torch113_cu117.log \
|| python -m pip install "torch==1.13.1" "torchvision==0.14.1" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  2>&1 | tee logs/Day5/day5_fix_torch113_tuna_fallback.log

python -m pip install \
  "numpy==1.23.5" "gitpython" "pyhash" \
  "hydra-core==1.1.1" "hydra-colorlog" "omegaconf==2.1.2" \
  "pytorch-lightning==1.8.6" "lightning-lite" \
  "termcolor" "tqdm" "gym" "numpy-quaternion" \
  "opencv-python-headless" "scipy" "matplotlib" "pandas" "rich" \
  "pybullet" "egl_probe" "safetensors" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  2>&1 | tee logs/Day5/day5_fix_eval_min_deps.log

python -m pip install -e topic2_act/calvin_official/calvin_env/tacto --no-deps \
  2>&1 | tee logs/Day5/day5_fix_install_tacto.log
python -m pip install -e topic2_act/calvin_official/calvin_env --no-deps \
  2>&1 | tee logs/Day5/day5_fix_install_calvin_env.log
python -m pip install -e topic2_act/calvin_official/calvin_models --no-deps \
  2>&1 | tee logs/Day5/day5_fix_install_calvin_models.log

python -m pip check \
  2>&1 | tee logs/Day5/day5_fix_pip_check.log
```

`pip check` is diagnostic here. Missing optional official CALVIN packages that
are not used by the Day 5 placeholder smoke, such as `sentence-transformers`,
`MulticoreTSNE`, `moviepy`, or `wandb`, are not blockers if the verifier and
wrapper smoke below pass.

Download only the official debug dataset for Day 5:

```bash
cd /root/Test/Zhr/DL/HW3/topic2_act/calvin_official/dataset
sed -i 's/\r$//' download_data.sh

bash download_data.sh debug \
  2>&1 | tee /root/Test/Zhr/DL/HW3/logs/Day5/day5_fix_calvin_download_debug.log

ls -lah /root/Test/Zhr/DL/HW3/topic2_act/calvin_official/dataset \
  2>&1 | tee /root/Test/Zhr/DL/HW3/logs/Day5/day5_fix_calvin_dataset_listing.log
```

Acceptance: the dataset root is
`/root/Test/Zhr/DL/HW3/topic2_act/calvin_official/dataset/calvin_debug_dataset/`.

### Verify PyBullet and EGL

Pick a currently safe physical GPU from `nvidia-smi`; GPU `4` is only an
example. The shim exposes that physical GPU as logical CUDA device `0`, which is
why the verifier uses `--cuda-device 0`. If `egl_probe` still reports
`Graphics Devices: []` after imports pass, do not keep reinstalling Python
packages. At that point the likely blocker is system NVIDIA EGL/GLVND visibility.

```bash
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_calvin_eval
source scripts/activate_cuda_driver_shim.sh 4 \
  > >(tee logs/Day5/day5_fix_verify_shim.log) 2>&1

python -m egl_probe.get_available_devices \
  2>&1 | tee logs/Day5/day5_fix_egl_probe_devices.log

python topic2_act/eval/verify_calvin_eval_env.py \
  --calvin-root /root/Test/Zhr/DL/HW3/topic2_act/calvin_official \
  --cuda-device 0 \
  --egl-diagnostics-dir /root/Test/Zhr/DL/HW3/logs/Day5/egl_diag_verify \
  2>&1 | tee logs/Day5/day5_fix_verify_calvin_eval_env.log
```

Acceptance:

- `calvin_agent`, `calvin_env`, `pybullet`, `egl_probe`, `safetensors`, and
  `torch` imports are marked `ok`;
- if `egl.ok` is false but imports are ok, continue to deep EGL diagnostics
  below before changing dependencies.

If `day5_fix_egl_probe_devices.log` reports `Graphics Devices: []`, collect
GLVND, NVIDIA EGL library, and CALVIN raw checker evidence:

```bash
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_calvin_eval
source scripts/activate_cuda_driver_shim.sh 4 \
  > >(tee logs/Day5/day5_egl_deep_shim.log) 2>&1

echo "== GLVND vendor dirs ==" \
  2>&1 | tee logs/Day5/day5_egl_deep_glvnd.log
ls -lah /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d \
  2>&1 | tee -a logs/Day5/day5_egl_deep_glvnd.log
cat /usr/share/glvnd/egl_vendor.d/*.json /etc/glvnd/egl_vendor.d/*.json \
  2>&1 | tee -a logs/Day5/day5_egl_deep_glvnd.log

ldconfig -p | grep -E 'libEGL|libGLX|libOpenGL|nvidia' \
  2>&1 | tee logs/Day5/day5_egl_deep_ldconfig.log

nvidia-smi \
  2>&1 | tee logs/Day5/day5_egl_deep_nvidia_smi.log

env | sort | grep -E 'CUDA|EGL|LD_LIBRARY|DISPLAY|NVIDIA' \
  2>&1 | tee logs/Day5/day5_egl_deep_gpu_env.log

find /usr /lib /opt /run \
  \( -type f -o -type l \) \
  \( -name 'libEGL_nvidia.so*' -o -name 'libGLX_nvidia.so*' -o -name 'libnvidia-egl*.so*' -o -name 'libcuda.so*' \) -print \
  2>&1 | tee logs/Day5/day5_egl_deep_driver_libs.log

dpkg -l | grep -E 'nvidia|libegl|mesa|glvnd' \
  2>&1 | tee logs/Day5/day5_egl_deep_dpkg.log

apt-cache policy libnvidia-gl-535 nvidia-driver-535 nvidia-utils-535 \
  2>&1 | tee logs/Day5/day5_egl_deep_apt_policy.log

eglinfo -B \
  2>&1 | tee logs/Day5/day5_egl_deep_eglinfo.log
```

Run CALVIN's raw EGL checker separately so stdout and stderr are easy to inspect:

```bash
cd /root/Test/Zhr/DL/HW3/topic2_act/calvin_official/calvin_env/egl_check
rm -f EGL_options.o
bash build.sh \
  2>&1 | tee /root/Test/Zhr/DL/HW3/logs/Day5/day5_egl_deep_calvin_build.log

./EGL_options.o \
  > /root/Test/Zhr/DL/HW3/logs/Day5/day5_egl_deep_calvin_egl_stdout.log \
  2> /root/Test/Zhr/DL/HW3/logs/Day5/day5_egl_deep_calvin_egl_stderr.log || true

cat /root/Test/Zhr/DL/HW3/logs/Day5/day5_egl_deep_calvin_egl_stderr.log
cat /root/Test/Zhr/DL/HW3/logs/Day5/day5_egl_deep_calvin_egl_stdout.log
```

Interpretation:

- no `libEGL_nvidia.so*` or no NVIDIA GLVND JSON means system NVIDIA EGL
  userspace is missing or hidden;
- `libEGL_nvidia.so*` exists but GLVND JSON is missing means the ICD needs to be
  repaired;
- both exist but `eglQueryDevicesEXT` still finds zero devices means check driver
  library version mixing and `LD_LIBRARY_PATH` ordering.

Do not install `libnvidia-gl-535` merely because the candidate is in the 535
series. The server currently reports kernel driver `535.129.03`; an apt
candidate such as `535.309.01-0ubuntu0.22.04.1` is the same major branch but not
an exact userspace/kernel point-version match. Before installing, check whether
the exact `535.129.03` userspace package is available or ask the server
administrator whether the newer 535 userspace is acceptable:

```bash
cd /root/Test/Zhr/DL/HW3
mkdir -p logs/Day5

apt-cache policy libnvidia-gl-535 nvidia-driver-535 nvidia-utils-535 \
  2>&1 | tee logs/Day5/day5_nvidia_gl_policy_before_install.log

apt-cache madison libnvidia-gl-535 nvidia-driver-535 nvidia-utils-535 \
  2>&1 | tee logs/Day5/day5_nvidia_gl_madison.log

apt-cache policy 'libnvidia-gl-535=535.129.03*' \
  2>&1 | tee logs/Day5/day5_nvidia_gl_535129_policy.log
```

Install system NVIDIA EGL userspace only if an exact `535.129.03` package is
available, an administrator confirms `535.309.01` userspace is safe with the
current driver, or the host libraries can be mounted/exposed without changing
system packages.

### Run Placeholder Wrapper Smoke

The wrapper smoke loads the A-only smoke500 checkpoint and runs a deliberately
short CALVIN rollout. The placeholder action is zero arm motion plus an open
gripper (`action[-1] = 1.0`), because CALVIN requires the gripper action to be
either `-1` or `1`. Success rate is not a Day 5 metric; the target is that the
official env calls our `reset()` and `step()` without interface errors.

Use `--egl-policy direct` as the Day 5 unblock path when NVIDIA EGL device
enumeration is still broken. This temporarily disables the CALVIN EGL plugin and
clears all camera configs, so the env becomes state-only PyBullet DIRECT. This
avoids the tactile `tacto -> pyrender` rendering dependency and is not the final
Week 2 evaluation mode.

```bash
cd /root/Test/Zhr/DL/HW3
source /opt/conda/etc/profile.d/conda.sh
conda activate env_hw3_calvin_eval
source scripts/activate_cuda_driver_shim.sh 4 \
  > >(tee logs/Day5/day5_direct_nocamera_gripperfix_wrapper_shim.log) 2>&1

DATASET=/root/Test/Zhr/DL/HW3/topic2_act/calvin_official/dataset/calvin_debug_dataset
CKPT=$(find /root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_smoke500_50ep/lerobot_train/checkpoints \
  -path '*/pretrained_model/model.safetensors' -printf '%h\n' | sort | tail -n 1)

echo "DATASET=$DATASET"
echo "CKPT=$CKPT"

python topic2_act/eval/run_calvin_eval.py \
  --calvin-root /root/Test/Zhr/DL/HW3/topic2_act/calvin_official \
  --dataset-path "$DATASET" \
  --checkpoint "$CKPT" \
  --eval-log-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/calvin_eval/day5_zero_action_debug_direct_nocamera_gripperfix \
  --cuda-device 0 \
  --num-sequences 1 \
  --ep-len 2 \
  --egl-policy direct \
  2>&1 | tee logs/Day5/day5_direct_nocamera_gripperfix_lerobot_act_wrapper_smoke.log
```

Acceptance:

- `wrapper_checkpoint_summary.loaded` is true;
- `tensor_count` is non-zero;
- `egl_audit.egl_policy` is `direct` and notes that CUDA/EGL mapping was
  skipped for the temporary Day 5 smoke;
- `calvin_eval_start.direct_camera_mode` is `none`;
- logs show `LeRobotACTWrapper.reset called` and at least one
  `LeRobotACTWrapper.step called`;
- logs no longer show `assert self.gripper_action in (-1, 1)`;
- the command reaches `calvin_eval_result` or a clear CALVIN environment error
  that can be debugged from the saved log.

After NVIDIA EGL is repaired, rerun the same command with `--egl-policy strict`
and `eval-log-dir` set to a separate output directory.

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
