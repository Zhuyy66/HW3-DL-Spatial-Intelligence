# HW3-DL-Spatial-Intelligence

Final homework project for the graduate course Deep Learning and Spatial
Intelligence.

Remote repository:
<https://github.com/Zhuyy66/HW3-DL-Spatial-Intelligence>

## Tasks

This repository tracks two project lines:

- Topic 1: multi-source 3D asset generation and real-scene fusion with COLMAP,
  3D Gaussian Splatting, threestudio, Stable Zero123, and Blender.
- Topic 2: cross-environment generalization of ACT policies on CALVIN with
  LeRobot.

## Team Division

- Member A owns Topic 1, including COLMAP, 3DGS, threestudio, Zero123, and
  Blender fusion rendering.
- Member B owns Topic 2, including LeRobot, CALVIN dataset processing, ACT
  training, and zero-shot evaluation.

## Repository Structure

```text
topic1_fusion/     # Topic 1: 3DGS + AIGC + scene fusion
topic2_act/        # Topic 2: LeRobot + ACT + CALVIN
env/               # Environment setup notes
reports/           # LaTeX report source and figures
assets/            # Small public assets
md/                # Local planning exports, ignored by git
logs/              # Local/server logs, ignored by git
```

Large data, model outputs, checkpoints, logs, PDF course files, and planning
exports are intentionally ignored by git. Reproducible commands and small
metadata files should be tracked; heavy artifacts should be published through
external storage or regenerated on the GPU server.

## Week 1 Status Snapshot

This snapshot reflects the Day 7 handoff state after the Week 1 engineering
smoke tests. Topic 1 evidence comes from the downloaded Feishu project document
`md/0611_深度学习与空间智能期末PJ.md`. Topic 2 evidence comes from tracked code,
local logs, and run manifests.

| Line | Week 1 status | Evidence | Repo state |
| --- | --- | --- | --- |
| Topic 1: 3DGS + AIGC fusion | Environment setup, counter 7k/30k 3DGS runs, object_A 7k/30k 3DGS runs, hamburger threestudio smoke, and Blender trajectory checks are recorded in the shared project document. | Feishu export records counter 30k test PSNR 29.2630, object_A 30k test PSNR 31.3322, DreamFusion hamburger smoke, and Blender checks. | Current tracked `topic1_fusion/` still contains only the initial directory skeleton. Member A's small configs, scripts, screenshots, and README updates still need to be merged. Heavy `.ply`, videos, and generated meshes should stay outside git with documented download paths. |
| Topic 2: ACT + CALVIN | Environment, official splitA data preparation, A-only smoke500 ACT baseline, CALVIN eval scaffold, real ACT worker bridge, and direct-cameras rollout smoke are complete for Week 1. | `topic2_act/docs/day3_data_audit.md`, `logs/Day4/day4_act_10epoch_health_check.log`, `logs/Day5/day5_a_only_smoke500_final_check.log`, `logs/Day6/day6_bridge_fidelity_after_config_fix.log`, and `logs/Day6/day6_real_act_direct_camera_rollout_after_pickle_fix.log`. | Code, configs, split metadata, wrappers, tests, and runbooks are tracked on `topic2-act`. The `a_only_smoke500_50ep` model is an engineering baseline for chain validation, not the final A-only vs ABC comparison checkpoint. |

Repository fact checked before this Day 7 documentation commit:

- `topic2-act` was synchronized with `origin/topic2-act` at commit `cee2a3a`.
- `origin/main` and `origin/topic1-fusion` were still at the initial project
  structure commit `69eb6eb`.
- The Week 1 tag should therefore be read as a documented handoff point for the
  current branch, with Topic 1 generated artifacts still pending Git-side
  integration.

## Topic 2 Current Entrypoints

Member B uses `env_hw3_robot` for LeRobot ACT training and
`env_hw3_calvin_eval` for the official CALVIN-side rollout wrapper. The server
setup and verification commands are documented in:

```text
env/env_hw3_robot_setup.md
topic2_act/README.md
topic2_act/eval/README.md
```

The production CALVIN data path is the official course split dataset:

```text
https://huggingface.co/datasets/xiaoma26/calvin-lerobot
```

Topic 2 now uses official `splitA` for A-only preparation. The old
`huiwon/calvin_task_ABC_D` plus `scene_info.npy` reverse-splitting path is kept
as cross-validation evidence under `topic2_act/legacy/scene_info_split/`.

## Report

The report skeleton is in:

```text
reports/main.tex
```

It follows a NeurIPS-style structure with sections for Abstract, Introduction,
Related Work, Methods, Experiments, Discussion, and Conclusion. The first
writing pass should fill background, related work, datasets, and methods while
the remaining Week 3-4 experiments run.

## Compressed Week 3-4 Risk List

The original Week 2-4 plan has been compressed into two remaining weeks. The
next plan should prioritize:

1. Topic 1 Git integration: add small scripts, configs, READMEs, screenshots,
   and artifact manifests for the Feishu-recorded 3DGS, threestudio, and
   Blender outputs.
2. Topic 2 formal experiments: train or prepare the final A-only and ABC
   checkpoints under a comparable budget, separate from the smoke500 baseline.
3. CALVIN evaluation: repair strict NVIDIA EGL if possible; otherwise define a
   transparent fallback evaluation route and document its limitations.
4. Results packaging: export loss curves, rollout metrics, failure cases,
   Topic 1 visual figures, and report tables into `reports/figures/`.
5. Final reproducibility: keep README commands aligned with tracked code and
   external artifact paths.
