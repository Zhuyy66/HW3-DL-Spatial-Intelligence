# Topic1 Experiments Numbers

## 3DGS Metrics

| Asset | Iteration | Test PSNR | Train PSNR | W&B run id | Main output |
|---|---:|---:|---:|---|---|
| counter | 7000 | 27.3455 | 28.9319 | `zo8odz6d` | `/root/HW3/topic1_fusion/outputs/counter/gs_7k/` |
| counter | 30000 | 29.2630 | 31.5692 | `q8rwxr2b` | `/root/HW3/topic1_fusion/outputs/counter/gs_30k_gpu5/` |
| object_A | 7000 | 31.1154 | 34.7399 | `2ix2e3sx` | `/root/HW3/topic1_fusion/outputs/object_A/gs_7k/` |
| object_A | 30000 | 31.3322 | 37.7307 | `stb4ukx9` | `/root/HW3/topic1_fusion/outputs/object_A/gs_30k/` |

## Generated Assets

| Asset | Method | Iterations | Selected version | Export files |
|---|---|---:|---|---|
| object B hamburger | DreamFusion + SD 1.5 | 100 smoke / 10000 full | 10000 full | `/root/HW3/topic1_fusion/outputs/object_B/threestudio_runs/dreamfusion-sd15-full/hamburger_full_20260603/save/it10000-export/` |
| object C toy | Stable Zero123 | 600 / 3000 / 10000 | 3000 | `/root/HW3/topic1_fusion/outputs/object_C/threestudio_runs/stable-zero123-objectC/objectC_202new_zero123_3000_20260616/save/it3000-export/` |

## Report Figures

| Figure | File |
|---|---|
| Garden pretrained render/GT check | `figures/garden_pretrained_render_gt_contact_sheet.png` |
| Counter 30k render/GT | `figures/counter_30k_render_gt_contact_sheet.png` |
| Object A 30k render/GT | `figures/object_A_30k_render_gt_contact_sheet.png` |
| Hamburger smoke | `figures/hamburger_smoke_it100.png` |
| Hamburger 10000 validation | `figures/hamburger_it10000_validation.png` |
| Hamburger texture | `figures/hamburger_texture_kd.jpg` |
| Object C training strip | `figures/objectC_training_strip.png` |
| Object C texture | `figures/objectC_texture_kd.jpg` |
| Counter orbit frames | `figures/counter_orbit_frame_0000.png`, `figures/counter_orbit_frame_0075.png` |

## Fusion Video

Status: complete.

Final path:

`/root/HW3/reports/topic1_delivery_2026-06-22/place_final_fusion_video_here/topic1_final_fusion_counter_ABC.mp4`

Video metadata:

- 300 frames
- 30 fps
- 10 seconds
- resolution: 3114 x 2076

Keyframes:

- `place_final_fusion_video_here/keyframes/fusion_frame_0000.png`
- `place_final_fusion_video_here/keyframes/fusion_frame_0075.png`
- `place_final_fusion_video_here/keyframes/fusion_frame_0150.png`
- `place_final_fusion_video_here/keyframes/fusion_frame_0225.png`
- `place_final_fusion_video_here/keyframes/fusion_frame_0299.png`
