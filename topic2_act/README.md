# Topic 2: LeRobot ACT on CALVIN

Member B owns this track.

## Day 1 Checklist

- Environment: `env_hw3_robot`, Python 3.12.
- GPU-ready stack: `torch==2.6.0`, CUDA 12.4, `torchcodec==0.2.0`,
  `lerobot==0.4.0`.
- Verification command:

```bash
python topic2_act/scripts/verify_act_import.py --require-cuda \
  2>&1 | tee logs/day1_verify_act_import_$(date +%Y%m%d_%H%M%S).log
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
  --local-dir /root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_probe \
  --max-meta-files 50 \
  2>&1 | tee logs/day1_probe_calvin_dataset_$(date +%Y%m%d_%H%M%S).log
```

## Logging Rule

Use `2>&1 | tee logs/name.log` for setup, verification, and dataset probes,
because these commands should remain visible in the terminal while preserving
logs for later review. Use `> logs/name.log 2>&1 &` only for long background
jobs such as training.

## Day 2 Output Target

After the probe passes on the server, write a cleaned data audit summary that
records the schema, episode split strategy, and any blockers before starting
full dataset downloads.
