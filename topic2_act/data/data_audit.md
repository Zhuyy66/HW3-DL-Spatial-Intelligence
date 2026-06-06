# Day 2 CALVIN Data Audit

- Generated at: `2026-06-06T05:58:46.446715+00:00`
- Dataset root: `/root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_first_shard/calvin_task_ABC_D_lerobot_0_4`
- Dataset name: `calvin_task_ABC_D_lerobot_0_4`
- LeRobot codebase version: `v2.1`
- Robot type: `franka`
- Total episodes: `4468`
- Total frames: `267682`
- FPS: `30`

## Format Compatibility

- Warning: Dataset metadata is LeRobot codebase v2.1. Day 2 metadata audit and episode-list splitting are valid, but direct training with lerobot==0.4.0 may require confirming the v2.1 loader path or converting to v3.0.

## Schema

- `observation.state`: `{'dtype': 'float32', 'shape': [15]}`
- `action`: `{'dtype': 'float32', 'shape': [7]}`
- Action modality: `{"eef_pos_delta": {"start": 0, "end": 3, "absolute": false}, "eef_rot_delta": {"start": 3, "end": 6, "absolute": false, "rotation_type": "euler_angles_rpy"}, "gripper_close": {"start": 6, "end": 7}}`
- Video modality: `{"image": {"original_key": "observation.images.image"}, "wrist_image": {"original_key": "observation.images.wrist_image"}}`

## Environment Label Source

- No explicit `env`, `scene_id`, or `environment` metadata field was found.
- Environment labels are derived from `episodes_stats.original_frame_idx` intersected with the fixed `scene_info.npy` ranges.
- Scene info: `/root/Test/Zhr/DL/HW3/topic2_act/data/scene_info_fix/training/scene_info.npy`
- Scene ranges: `[{"name": "calvin_scene_B", "label": "B", "start": 0, "end": 598909}, {"name": "calvin_scene_C", "label": "C", "start": 598910, "end": 1191338}, {"name": "calvin_scene_A", "label": "A", "start": 1191339, "end": 1795044}]`
- Label counts: `{"A": 1551, "B": 1482, "C": 1435}`
- Dataset frame range: `805..1794868`
- Scene frame range: `0..1795044`
- Validation ok: `True`
- Validation failures: `[]`
- Validation warnings: `['episode_index order is interleaved across environments; this is expected for this converted shard, and split logic must not depend on episode order']`
- Episode-order transitions: `2928`
- Frame-sorted transitions: `2`

## Parquet Probe

- Sample parquet: `/root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_first_shard/calvin_task_ABC_D_lerobot_0_4/data/chunk-000/episode_000000.parquet`
- Rows: `43`
- `observation.state`: dtype=`object`, type=`ndarray`, shape=`[15]`
- `action`: dtype=`object`, type=`ndarray`, shape=`[7]`
- `annotation.human.action.task_description`: dtype=`int64`, type=`int64`, shape=`[]`
- `original_frame_idx`: dtype=`int64`, type=`int64`, shape=`[]`
- `timestamp`: dtype=`float32`, type=`float32`, shape=`[]`
- `frame_index`: dtype=`int64`, type=`int64`, shape=`[]`
- `episode_index`: dtype=`int64`, type=`int64`, shape=`[]`
- `index`: dtype=`int64`, type=`int64`, shape=`[]`
- `task_index`: dtype=`int64`, type=`int64`, shape=`[]`

## Decision Notes

- Do not infer A/B/C/D from `episode_index` order or shard order.
- Use the generated episode index lists for training filters; do not copy large video/parquet data.
- D offline episodes are not required for the core zero-shot rollout path; CALVIN simulator evaluation will handle environment D later.
