# xiaoma26 CALVIN LeRobot Split Package

This directory contains the small, shareable split-definition artifacts for
Topic 2 Day 3. It intentionally does not contain parquet data, videos,
checkpoints, logs, or training outputs.

## Source

- Dataset: `xiaoma26/calvin-lerobot`
- Endpoint used on server: `https://hf-mirror.com`
- Revision: `main`
- Prepared split: `splitA`
- Server data root: `/root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot`
- Server generated split root: `/root/Test/Zhr/DL/HW3/topic2_act/data/splits/xiaoma26_calvin_lerobot`

## Verified Metadata

| split | scene | episodes | frames | tasks | fps |
| --- | --- | ---: | ---: | ---: | ---: |
| splitA | A | 6089 | 366693 | 389 | 10 |
| splitB | B | 6115 | 367096 | 389 | 10 |
| splitC | C | 5666 | 337954 | 389 | 10 |
| splitD | D | 5124 | 308918 | 389 | 10 |

## Files

| file | purpose | count / note | sha256 |
| --- | --- | --- | --- |
| `episodes_A_full.json` | Full A-only episode index list | 6089 episodes | `E196073328970CBD375885A16271C3E06CE559426ABAF6533C9F8A911F7526A6` |
| `episodes_A_smoke500.json` | Default Day 3 ACT smoke-training view | 500 episodes, seed `20260606` | `6E25F23C8C13EA034ABDD13D6D1B5197FF32C6E33ECF37343911F7109DDD48FE` |
| `official_split_summary.json` | Validation summary copied from the server run | `metadata_only=false`; all split checks passed | `B4A260289EB802C5BD2FB0414970283E3EC9E967D20297B284DA25AEF936D464` |

## Notes

- `splitA`, `splitB`, and `splitC` each start `episode_index` at 0. Do not
  concatenate raw episode indices for ABC joint training.
- Use LeRobot multi-dataset loading or remap episode indices before creating an
  ABC joint dataset.
- The retired `scene_info.npy` reverse split is archived under
  `topic2_act/legacy/scene_info_split/`.
