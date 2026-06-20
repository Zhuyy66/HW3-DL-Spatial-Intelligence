# Topic 1 Final Submission Index

## Git-tracked items

- `topic1_fusion/README.md`
- `topic1_fusion/docs/`
- `topic1_fusion/scripts/`
- `reports/figures/topic1_week1_urgent/MANIFEST.md`
- Small illustrative figures under `reports/figures/topic1_week1_urgent/`

## Local large artifacts kept outside Git

### Background `counter`

- 3DGS training outputs: `/root/HW3/topic1_fusion/outputs/counter_7k/`, `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/`
- Counter orbit render: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300_render/orbit.mp4`
- Dense mesh workspace: `/root/HW3/topic1_fusion/outputs/counter_colmap_dense/`
- Textured mesh: `/root/HW3/topic1_fusion/outputs/counter_texrecon/`

### Object A

- 3DGS training outputs: `/root/HW3/topic1_fusion/outputs/object_A_7k/`, `/root/HW3/topic1_fusion/outputs/object_A_30k/`
- Cleaned Blender asset: `/root/HW3/topic1_fusion/outputs/object_A_assets/object_A_clean.blend`
- Dense mesh workspace: `/root/HW3/topic1_fusion/outputs/object_A_colmap_dense/`
- Textured mesh: `/root/HW3/topic1_fusion/outputs/object_A_texrecon/`

### Object B

- DreamFusion outputs: `/root/HW3/topic1_fusion/outputs/threestudio/runs/dreamfusion-sd15-full/`
- Smoke outputs: `/root/HW3/topic1_fusion/outputs/threestudio/runs/dreamfusion-sd15-smoke/`

### Object C

- Stable Zero123 outputs: `/root/HW3/topic1_fusion/outputs/threestudio/runs/stable-zero123-objectC/`

## Key scripts

- Counter 7k training: `topic1_fusion/scripts/run_counter_7k.sh`
- Garden smoke render: `topic1_fusion/scripts/run_garden_smoke_render.sh`
- Camera trajectory sampling: `topic1_fusion/scripts/sample_camera_trajectory.py`
- Blender camera export/import helpers:
  - `topic1_fusion/scripts/export_blender_transforms.py`
  - `topic1_fusion/scripts/blender_import_cameras.py`
- COLMAP dense mesh helper: `topic1_fusion/scripts/run_colmap_dense_mesh.sh`
- Texrecon scene prep: `topic1_fusion/scripts/prepare_texrecon_scene.py`
- Object C Zero123 run: `topic1_fusion/scripts/run_object_c_stable_zero123.sh`
- DreamFusion runs:
  - `topic1_fusion/scripts/run_dreamfusion_sd15_smoke.sh`
  - `topic1_fusion/scripts/run_dreamfusion_sd15_full.sh`

## Notes

- `topic1_fusion/code/` contains only lightweight compatibility placeholders in Git; large external codebases are intentionally ignored.
- `topic1_fusion/data/`, `topic1_fusion/outputs/`, `topic1_fusion/pretrained/`, `hf_home/`, and `third_party/` remain local-only to keep the repository lightweight.
