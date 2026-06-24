# HW3-DL-Spatial-Intelligence

Final project for the Deep Learning and Spatial Intelligence course.

The repository contains two project tracks:

- `topic1_fusion/`: multi-source 3D asset generation and real-scene fusion.
- `topic2_act/`: ACT policy training and cross-environment evaluation on CALVIN.

## Repository Layout

```text
topic1_fusion/     Topic 1 source scripts, documentation, and lightweight configs
topic2_act/        Topic 2 project skeleton and evaluation/training entry points
env/               Environment notes and dependency constraints
reports/           LaTeX report source and report-ready figures
assets/            Small public assets
```

Large datasets, model checkpoints, generated meshes, videos, pretrained weights, and third-party source trees are intentionally kept out of Git. Their server paths and cloud-share targets are documented in the topic-specific manifests.

## Topic 1 Summary

Topic 1 builds a fused 3D scene from four assets:

- Background: Mip-NeRF 360 `counter`, reconstructed with 3DGS and converted to a textured mesh for Blender placement.
- Object A: a real phone-captured object reconstructed from multi-view video with COLMAP + 3DGS, then cleaned and textured for Blender.
- Object B: a DreamFusion text-to-3D hamburger generated from the prompt `a hamburger`.
- Object C: a Stable Zero123 single-image-to-3D toy generated from a cropped RGBA input image.

The final scene is rendered in Blender along a 300-frame orbit trajectory. Report figures are stored under `reports/figures/final_report/`.

## Topic 2 Summary

Topic 2 compares ACT policies trained on CALVIN-LeRobot splits:

- A-only: trained on splitA.
- ABC: trained on splitA, splitB, and splitC with the same network and training budget.
- Evaluation: open-loop Action L1 on the unseen splitD environment.

Topic 2 figures are included in the report draft folder and final report assets.

## Report

The main LaTeX report entry is:

```text
reports/main.tex
```

The current report figures are in:

```text
reports/figures/final_report/
```

