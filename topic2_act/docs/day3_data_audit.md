# Day 3 Topic 2 Data Audit

## Production Dataset Decision

The production data path is the official course split dataset:

```text
https://huggingface.co/datasets/xiaoma26/calvin-lerobot
```

Use `splitA` for A-only ACT training. The prior `scene_info.npy` reverse split
is retired and preserved only under `topic2_act/legacy/scene_info_split/`.

## Official Split Metadata

Read from `xiaoma26/calvin-lerobot` metadata on 2026-06-06:

| split | scene | episodes | frames | tasks | fps |
| --- | --- | ---: | ---: | ---: | ---: |
| splitA | A | 6089 | 366693 | 389 | 10 |
| splitB | B | 6115 | 367096 | 389 | 10 |
| splitC | C | 5666 | 337954 | 389 | 10 |
| splitD | D | 5124 | 308918 | 389 | 10 |

Official schema:

| key | shape | note |
| --- | --- | --- |
| `state` | `[15]` | proprioceptive state |
| `actions` | `[7]` | end-effector delta pose plus gripper |
| `image` | `[200, 200, 3]` | static RGB camera |
| `wrist_image` | `[84, 84, 3]` | wrist RGB camera |
| `task_index` | `[1]` | task annotation index |
| `scene` | episode metadata | official environment label |

The official split uses `state` and `actions`; the retired Day 2 converted
shard used `observation.state` and `action`. Training code/config must use the
official keys for the new production path.

## Cross-validation Against Official Split

The Day 2 server audit independently reverse-labelled the original data by
intersecting `episodes_stats.original_frame_idx` with fixed CALVIN scene ranges.
That path is no longer the production splitter, but it is important evidence:

- prior full reverse audit: `A=6089`, `B=6115`, `C=5666`;
- prior quality checks: `cross=0`, `orphan=0`;
- official split metadata: `splitA=6089`, `splitB=6115`, `splitC=5666`.

The exact agreement between the independent reverse audit and the official
course split supports both the official dataset choice and the correctness of
the Day 2 data-auditing method. The old generated episode lists are retired
because the official split is simpler and less error-prone.

## Day 3 Prepared Outputs

`prepare_xiaoma_calvin_split.py` writes these server-side outputs:

```text
topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_A_full.json
topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_A_smoke500.json
topic2_act/data/splits/xiaoma26_calvin_lerobot/official_split_summary.json
topic2_act/data/splits/xiaoma26_calvin_lerobot/download_manifest.json
```

The small shareable split-definition files are tracked separately under:

```text
topic2_act/dataset_split/xiaoma26_calvin_lerobot/
```

`episodes_A_smoke500.json` is the default Day 3 ACT training view. It is a
deterministic seeded subset of full `splitA`; `episodes_A_full.json` remains
available for the full A-only run.

## CALVIN CustomModel Interface Notes

From CALVIN `evaluate_policy.py` and `CalvinBaseModel`:

- `reset()` is called before each subtask rollout.
- `step(obs, goal)` is called every environment step.
- `obs` is the CALVIN environment observation.
- `goal` is the language annotation selected for the current subtask.
- The return value of `step()` is passed directly into `env.step(action)`.

For ACT evaluation, the wrapper must cache or refresh an action chunk after
`reset()`, map CALVIN observations into the LeRobot ACT input keys, and return a
single 7D action at each `step()` call.

## ABC Joint Training TODO

Do not create an ABC episode list by concatenating raw episode indices from the
official splits. `splitA`, `splitB`, and `splitC` each start `episode_index` at
0. ABC joint training must either use LeRobot multi-dataset loading or remap
episode indices before writing a merged dataset.
