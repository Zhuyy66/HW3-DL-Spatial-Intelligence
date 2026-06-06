# HW3-DL-Spatial-Intelligence

Final homework project for Deep Learning and Spatial Intelligence.

## Tasks

This repository contains two tasks:

- Topic 1: Multi-source 3D asset generation and real-scene fusion based on 3D Gaussian Splatting and AIGC.
- Topic 2: Cross-environment generalization of ACT policy based on LeRobot and CALVIN.

## Team Division

- Member A: Topic 1, including COLMAP, 3DGS, threestudio, Zero123, and Blender fusion rendering.
- Member B: Topic 2, including LeRobot, CALVIN dataset processing, ACT training, and zero-shot evaluation.

## Repository Structure

```text
topic1_fusion/     # Topic 1: 3DGS + AIGC + scene fusion
topic2_act/        # Topic 2: LeRobot + ACT + CALVIN
env/               # Environment configuration files
reports/           # Report source files and figures
assets/            # Small public assets
```

## Day 1 Topic 2 Notes

Member B uses `env_hw3_robot` for LeRobot ACT training. The server setup and
verification commands are documented in `env/env_hw3_robot_setup.md`.

The assignment PDF points to the course dataset
`huiwon/calvin_task_ABC_D`. Probe metadata first before downloading the full
dataset.

Day 3 update: the course staff later published official environment splits at
`xiaoma26/calvin-lerobot`. Topic 2 now uses that dataset as the production data
path for A-only ACT preparation.
