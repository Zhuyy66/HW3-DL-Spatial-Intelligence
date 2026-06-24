# Topic 1: Multi-Source 3D Asset Fusion

Topic 1 constructs a unified 3D scene from a real reconstructed background, a real captured object, a text-generated object, and a single-image-generated object. The final result is rendered as a 300-frame Blender orbit video in the Mip-NeRF 360 `counter` scene.

## Pipeline Overview

The project follows four stages:

1. Reconstruct the `counter` background with 3D Gaussian Splatting.
2. Reconstruct `object_A` from phone-captured multi-view video with COLMAP and 3DGS.
3. Generate object B and object C with threestudio:
   - object B: DreamFusion, prompt `a hamburger`
   - object C: Stable Zero123 from one RGBA input image
4. Convert usable assets to textured meshes and align them in Blender under a shared 300-frame orbit camera trajectory.

The final rendering path uses textured meshes in Blender. The Gaussian assets are kept as reconstruction evidence and as a fallback representation for same-trajectory rendering.

## Main Results

### 3DGS Reconstruction Metrics

| Asset | Iteration | Test PSNR | Train PSNR | W&B run id |
|---|---:|---:|---:|---|
| `counter` | 7000 | 27.3455 | 28.9319 | `zo8odz6d` |
| `counter` | 30000 | 29.2630 | 31.5692 | `q8rwxr2b` |
| `object_A` | 7000 | 31.1154 | 34.7399 | `2ix2e3sx` |
| `object_A` | 30000 | 31.3322 | 37.7307 | `stb4ukx9` |

### Generated Assets

| Asset | Method | Selected version | Output form |
|---|---|---|---|
| object B hamburger | DreamFusion + SD 1.5 | 10000 iterations | textured OBJ/MTL |
| object C toy | Stable Zero123 | 3000 iterations | textured OBJ/MTL |

The 3000-iteration object C run is used in the final scene because it has a complete mesh/material/texture export and was visually more stable for Blender placement. The 10000-iteration run is retained as a longer-training comparison video.

### Final Fusion Video

The final fusion render is a 300-frame, 30 fps, 10-second Blender video at `3114 x 2076` resolution. The mp4 and keyframes are kept outside Git and documented in the artifact manifest.

Report-ready keyframes are tracked in:

```text
reports/figures/final_report/
```

## Important Local Artifacts

Large artifacts are intentionally not tracked by Git.

```text
topic1_fusion/outputs/counter/gs_30k_gpu5/
topic1_fusion/outputs/counter/texrecon/
topic1_fusion/outputs/object_A/gs_30k/
topic1_fusion/outputs/object_A/texrecon/
topic1_fusion/outputs/object_B/threestudio_runs/dreamfusion-sd15-full/
topic1_fusion/outputs/object_C/threestudio_runs/stable-zero123-objectC/
reports/topic1_delivery_2026-06-22.zip
```

The detailed local artifact index is:

```text
topic1_fusion/docs/final_submission_index_2026-06-20.md
```

## Scripts

Key scripts kept in Git:

```text
topic1_fusion/scripts/filter_blurry.py
topic1_fusion/scripts/run_counter_7k.sh
topic1_fusion/scripts/sample_camera_trajectory.py
topic1_fusion/scripts/blender_import_cameras.py
topic1_fusion/scripts/export_blender_transforms.py
topic1_fusion/scripts/run_dreamfusion_sd15_smoke.sh
topic1_fusion/scripts/run_dreamfusion_sd15_full.sh
topic1_fusion/scripts/run_object_c_stable_zero123.sh
topic1_fusion/scripts/run_colmap_dense_mesh.sh
topic1_fusion/scripts/prepare_texrecon_scene.py
```

## Report Figures

The final report uses:

```text
reports/figures/final_report/
```

This folder contains the 3DGS render-vs-GT contact sheets, DreamFusion/Zero123 asset checks, counter orbit frames, and final fusion keyframes.

