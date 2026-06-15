# Topic 1 Week 1 Urgent Artifact Manifest

This folder collects the files that can be sent to teammates first. Large raw outputs remain outside Git and are referenced below.

## Ready To Send

- `topic1_week1_metrics_snapshot.png`: compact metric card for counter/object_A 7k and 30k runs.
- `garden_pretrained/garden_pretrained_render_gt_contact_sheet.png`: 3DGS pretrained garden render vs GT check.
- `counter/counter_30k_render_gt_contact_sheet.png`: counter 30k render vs GT contact sheet.
- `object_A/object_A_30k_render_gt_contact_sheet.png`: object_A 30k render vs GT contact sheet.
- `threestudio/hamburger_it10000_validation.png`: hamburger DreamFusion 10k validation image.
- `threestudio/hamburger_it10000-test.mp4`: hamburger DreamFusion 10k test orbit video.
- `threestudio/hamburger_smoke_it100.png`: hamburger 100-step smoke validation image.
- `threestudio/hamburger_smoke_it100-test.mp4`: hamburger 100-step smoke test video.
- `blender/counter_orbit_frame_0000.png` and `blender/counter_orbit_frame_0075.png`: sampled counter orbit frames.
- `blender/counter_orbit_300.mp4`: 300-frame/10-second counter orbit video. This file is about 48 MB and should be sent directly or put in external storage, not committed.

## Metrics

- counter 7k: test PSNR 27.3455, train PSNR 28.9319, W&B run id `zo8odz6d`.
- counter 30k: test PSNR 29.2630, train PSNR 31.5692, W&B run id `q8rwxr2b`.
- object_A 7k: test PSNR 31.1154, train PSNR 34.7399, W&B run id `2ix2e3sx`.
- object_A 30k: test PSNR 31.3322, train PSNR 37.7307, W&B run id `stb4ukx9`.
- hamburger DreamFusion full: 10000 steps completed, OBJ/MTL/texture exported, W&B run id `0w5yldmf`.

## Important Local Source Paths

- Garden pretrained renders: `/root/HW3/topic1_fusion/pretrained/garden/test/ours_30000/`
- Counter 7k output: `/root/HW3/topic1_fusion/outputs/counter_7k/`
- Counter 30k output: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/`
- object_A 7k output: `/root/HW3/topic1_fusion/outputs/object_A_7k/`
- object_A 30k output: `/root/HW3/topic1_fusion/outputs/object_A_30k/`
- Hamburger full output: `/root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-full/hamburger_full_20260603/`

## Gap To Fill

- A true Blender validation screenshot with `counter` camera trajectory viewing placed `object_A` was not found on disk. Current folder includes the counter orbit frames and object_A render evidence, but the combined Blender screenshot should be captured manually from Blender after loading the placed object.
- object_A is still visually point-cloud/Gaussian-like rather than a clean mesh. The report should either show point-cloud cleanup as the shortest repair path or clearly state this limitation.
