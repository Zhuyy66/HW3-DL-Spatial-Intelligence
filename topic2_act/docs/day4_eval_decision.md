# Week3 Day4 Topic2 Evaluation Decision

## Goal

Day 4 focuses on A-only evaluation on environment D. The required metric is
open-loop Action L1 on the official splitD offline data. Closed-loop CALVIN
success rate is auxiliary and is recorded only when the EGL or fallback rollout
route can run.

## Starting State

- Remote root: `/root/Test/Zhr/DL/HW3`
- A-only checkpoint:
  `/root/Test/Zhr/DL/HW3/topic2_act/outputs/act_calvin/a_only_full_150k/lerobot_train/checkpoints/last/pretrained_model`
- Official splitD metadata: 5124 episodes, 308918 frames, scene D, 10 fps.
- splitD raw parquet data must be downloaded before the open-loop metric can run.
- Prior Day 6 bridge route is reusable: CALVIN client stays thin, LeRobot ACT worker owns preprocessing and inference, and worker actions are serialized as plain `list[float]`.

## EGL Decision Protocol

The strict EGL route is checked first. System-package repair is allowed for this
task, so the runbook includes `libnvidia-gl-535` and `libnvidia-egl-wayland1`
dry-run and install logs. If strict EGL still cannot map CUDA to an NVIDIA EGL
device after repair, Day 4 uses the already validated `direct-cameras` route for
optional rollout evidence.

Report wording should state that strict NVIDIA EGL remained a system-level
graphics dependency when the after-repair check fails, and that the primary
metric is the offline splitD Action L1. The optional closed-loop smoke uses the
same checkpoint and bridge path as Day 6.

## Evidence Paths

Remote logs:

- `logs/Week3_Day4/day4_gpu_snapshot.log`
- `logs/Week3_Day4/day4_prepare_splitD.log`
- `logs/Week3_Day4/day4_a_only_splitD_action_l1_smoke.log`
- `logs/Week3_Day4/day4_a_only_splitD_action_l1_full.log`
- `logs/Week3_Day4/day4_egl_strict_before.log`
- `logs/Week3_Day4/day4_egl_apt_dry_run.log`
- `logs/Week3_Day4/day4_egl_apt_install.log`
- `logs/Week3_Day4/day4_egl_strict_after.log`
- `logs/Week3_Day4/day4_egl_strict_after_shim.log`
- `logs/Week3_Day4/day4_a_only_closed_loop_direct_cameras.log`
- `logs/Week3_Day4/day4_worker_a_only_closed_loop_direct_cameras.log`
- `logs/Week3_Day4/day4_remote_open_loop_unittest.log`

Expected result files:

- `topic2_act/eval/results/a_only_splitD_action_l1_smoke.json`
- `topic2_act/eval/results/a_only_splitD_action_l1.json`

## Current Result

Completed on 2026-06-22.

### EGL route decision

Decision: use `direct-cameras` for Day 4 optional closed-loop evidence.

Strict EGL was tested before and after a system-package repair attempt:

- Before repair, `day4_egl_strict_before.log` reported
  `EglDeviceNotFoundError()` and the CALVIN EGL probe rendered with
  `GL_RENDERER=llvmpipe`.
- The dry run in `day4_egl_apt_dry_run.log` selected
  `libnvidia-common-535`, `libnvidia-egl-wayland1`, and `libnvidia-gl-535`.
- The installation in `day4_egl_apt_install.log` completed, and
  `day4_egl_strict_after.log` confirmed the NVIDIA EGL vendor file
  `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` exists.
- A final strict check with the project CUDA shim loaded is recorded in
  `day4_egl_strict_after_shim.log`. Torch CUDA was available on the visible
  RTX 4090, but EGL mapping still failed with `EglDeviceNotFoundError()` and
  the CALVIN probe still reported Mesa llvmpipe.

Operational note: remote GPU commands must prepend the relevant conda env
`bin` directory to `PATH` before sourcing `scripts/activate_cuda_driver_shim.sh`.
The shim uses `python` while filtering `LD_LIBRARY_PATH`; without this prefix,
`LD_LIBRARY_PATH` can miss the project `.cuda_driver_shim` entry and PyTorch can
raise CUDA error 804.

### Open-loop A-only splitD result

Primary metric result file:
`topic2_act/eval/results/a_only_splitD_action_l1.json`.

- `status`: `completed`
- `action_l1_valid_mean`: `0.658629341784413`
- `forward_l1_loss_equivalent`: `0.2068762783325708`
- `raw_action_l1_valid_mean`: `0.19335991432529512`
- `episode_count`: `5124`
- `frame_count`: `308918`
- `missing_parquet_count`: `0`
- `valid_action_element_count`: `67922064`
- `batch_size`: `16`
- `elapsed_seconds`: `989.1339224614203`
- Dataset metadata: official v2.1 splitD, scene D, 10 fps, `task_D_D`.

Smoke result file:
`topic2_act/eval/results/a_only_splitD_action_l1_smoke.json`.

- `episode_count`: `4`
- `frame_count`: `260`
- `action_l1_valid_mean`: `0.6965860536140261`
- `missing_parquet_count`: `0`

### Closed-loop auxiliary smoke

Auxiliary rollout used `--egl-policy direct-cameras` with the A-only checkpoint,
the py3.8 CALVIN client, and the py3.12 LeRobot worker. The run completed 60
steps and is recorded in
`logs/Week3_Day4/day4_a_only_closed_loop_direct_cameras.log`.

Observed smoke facts:

- Worker loaded the checkpoint on `cuda:0` and reported `chunk_size=100`,
  `n_action_steps=100`, and `action_dim=7`.
- `single_rollout_smoke_result.steps`: `60`
- `wrapper_step_count`: `60`
- `last_action_shape`: `[7]`
- `moved`: `true`
- `max_tcp_delta`: `0.26423480109713104`
- `gripper_values`: `[-1.0, 1.0]`

This smoke is recorded as bridge and runtime evidence. It is auxiliary to the
offline Action L1 metric.
