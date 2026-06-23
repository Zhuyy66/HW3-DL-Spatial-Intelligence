# Topic 1 Final Submission Index

Latest delivery folder:

- `/root/HW3/reports/topic1_delivery_2026-06-22/`
- Object C mesh zip: `/root/HW3/reports/topic1_delivery_2026-06-22/object_C_mesh/objectC_stable_zero123_it3000_mesh.zip`
- Final fusion video: `/root/HW3/reports/topic1_delivery_2026-06-22/place_final_fusion_video_here/topic1_final_fusion_counter_ABC.mp4`
- Final fusion keyframes: `/root/HW3/reports/topic1_delivery_2026-06-22/place_final_fusion_video_here/keyframes/`

## Git-tracked items

- `topic1_fusion/README.md`
- `topic1_fusion/docs/`
- `topic1_fusion/scripts/`
- `reports/figures/topic1_week1_urgent/MANIFEST.md`
- Small illustrative figures under `reports/figures/topic1_week1_urgent/`

## Local large artifacts kept outside Git

### Background `counter`

- 3DGS training outputs: `/root/HW3/topic1_fusion/outputs/counter/gs_7k/`, `/root/HW3/topic1_fusion/outputs/counter/gs_30k_gpu5/`
- Counter orbit render: `/root/HW3/topic1_fusion/outputs/counter/gs_30k_gpu5/counter_orbit_300_render/orbit.mp4`
- Dense mesh workspace: `/root/HW3/topic1_fusion/outputs/counter/colmap_dense/`
- Textured mesh: `/root/HW3/topic1_fusion/outputs/counter/texrecon/`

### Object A

- 3DGS training outputs: `/root/HW3/topic1_fusion/outputs/object_A/gs_7k/`, `/root/HW3/topic1_fusion/outputs/object_A/gs_30k/`
- Cleaned Blender asset: `/root/HW3/topic1_fusion/outputs/object_A/assets/object_A_clean.blend`
- Dense mesh workspace: `/root/HW3/topic1_fusion/outputs/object_A/colmap_dense/`
- Textured mesh: `/root/HW3/topic1_fusion/outputs/object_A/texrecon/`

### Object B

- DreamFusion outputs: `/root/HW3/topic1_fusion/outputs/object_B/threestudio_runs/dreamfusion-sd15-full/`
- Smoke outputs: `/root/HW3/topic1_fusion/outputs/object_B/threestudio_runs/dreamfusion-sd15-smoke/`

### Object C

- Stable Zero123 outputs: `/root/HW3/topic1_fusion/outputs/object_C/threestudio_runs/stable-zero123-objectC/`
- Selected 3000-iteration mesh export: `/root/HW3/topic1_fusion/outputs/object_C/threestudio_runs/stable-zero123-objectC/objectC_202new_zero123_3000_20260616/save/it3000-export/`
- 10000-iteration comparison videos: `/root/HW3/topic1_fusion/outputs/object_C/threestudio_runs/stable-zero123-objectC/objectC_202new_zero123_10000_20260616/save/`

### Final report share package

- Delivery folder: `/root/HW3/reports/topic1_delivery_2026-06-22/`
- Figures: `/root/HW3/reports/topic1_delivery_2026-06-22/figures/`
- Existing validation videos: `/root/HW3/reports/topic1_delivery_2026-06-22/videos/`
- Final fusion video: `/root/HW3/reports/topic1_delivery_2026-06-22/place_final_fusion_video_here/topic1_final_fusion_counter_ABC.mp4`

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
