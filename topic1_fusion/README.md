# Topic 1 Fusion

Week 1 work focuses on validating the end-to-end assets for the final 3DGS + AIGC fusion pipeline:

- Background scene: Mip-NeRF 360 `counter`, trained with 3DGS.
- Object A: real phone-captured object reconstructed with COLMAP + 3DGS.
- Object B smoke/full chain: text-to-3D with threestudio DreamFusion, prompt `a hamburger`.
- Camera bridge: sampled `counter` orbit trajectory and Blender import validation.

## Week 1 Artifact Index

Urgent share package:

- `/root/HW3/reports/figures/topic1_week1_urgent/`
- Manifest: `/root/HW3/reports/figures/topic1_week1_urgent/MANIFEST.md`

Recommended Git-tracked files:

- `topic1_fusion/README.md`
- `topic1_fusion/docs/*.md`
- `topic1_fusion/scripts/*.py`
- `topic1_fusion/scripts/*.sh`
- Small report images in `reports/figures/`

Recommended external-storage files:

- Raw datasets under `topic1_fusion/data/`
- 3DGS outputs under `topic1_fusion/outputs/`
- Pretrained model zips and checkpoints under `topic1_fusion/pretrained/`
- threestudio outputs, checkpoints, meshes, videos, and W&B run folders
- `.ply`, `.ckpt`, `.mp4`, `.zip`, `.bin`, and other large binary artifacts

## Current Outputs

### 3DGS Pretrained Garden Render Check

- Source model: `/root/HW3/topic1_fusion/pretrained/garden/`
- Render output: `/root/HW3/topic1_fusion/pretrained/garden/test/ours_30000/`
- Share figure: `/root/HW3/reports/figures/topic1_week1_urgent/garden_pretrained/garden_pretrained_render_gt_contact_sheet.png`

Command:

```bash
bash /root/HW3/topic1_fusion/scripts/run_garden_smoke_render.sh \
  /root/HW3/topic1_fusion/pretrained/garden \
  /root/HW3/topic1_fusion/data/mipnerf360/garden \
  3
```

### Counter Background 3DGS

- 7k output: `/root/HW3/topic1_fusion/outputs/counter_7k/`
- 30k output: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/`
- 30k render output: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/test/ours_30000/`
- Share figure: `/root/HW3/reports/figures/topic1_week1_urgent/counter/counter_30k_render_gt_contact_sheet.png`
- Orbit video: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300_render/orbit.mp4`

Metrics:

- 7k: test PSNR 27.3455, train PSNR 28.9319, W&B run id `zo8odz6d`.
- 30k: test PSNR 29.2630, train PSNR 31.5692, W&B run id `q8rwxr2b`.

Commands:

```bash
bash /root/HW3/topic1_fusion/scripts/run_counter_7k.sh \
  /root/HW3/topic1_fusion/data/mipnerf360/counter \
  /root/HW3/topic1_fusion/outputs/counter_7k \
  5
```

```bash
# Full 30k run used GPU 5 and wrote to counter_30k_gpu5.
# See local W&B summary and output.log for exact metrics.
```

### object_A Real Capture 3DGS

- 7k output: `/root/HW3/topic1_fusion/outputs/object_A_7k/`
- 30k output: `/root/HW3/topic1_fusion/outputs/object_A_30k/`
- 30k render output: `/root/HW3/topic1_fusion/outputs/object_A_30k/test/ours_30000/`
- Final Gaussian file: `/root/HW3/topic1_fusion/outputs/object_A_30k/point_cloud/iteration_30000/point_cloud.ply`
- Share figure: `/root/HW3/reports/figures/topic1_week1_urgent/object_A/object_A_30k_render_gt_contact_sheet.png`

Metrics:

- 7k: test PSNR 31.1154, train PSNR 34.7399, W&B run id `2ix2e3sx`.
- 30k: test PSNR 31.3322, train PSNR 37.7307, W&B run id `stb4ukx9`.

Current limitation:

- object_A is still best treated as a Gaussian/point-cloud-like asset for placement validation.
- A direct point-cloud-to-mesh attempt produced an unusable coarse box-like OBJ, so the short repair path is point cleanup, better foreground masking/frame selection, or transparent reporting of the limitation.

### Blender Camera Bridge

- Counter original camera import: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_camera_import.blend`
- Sampled orbit JSON: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300.json`
- Blender-compatible orbit JSON: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300_blender.json`
- Blender orbit import: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300_import.blend`
- Rendered orbit: `/root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300_render/orbit.mp4`

Commands:

```bash
python /root/HW3/topic1_fusion/scripts/sample_camera_trajectory.py \
  --input_cameras /root/HW3/topic1_fusion/outputs/counter_30k_gpu5/cameras.json \
  --output_json /root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300.json \
  --num_frames 300
```

```bash
blender -b --python /root/HW3/topic1_fusion/scripts/blender_import_cameras.py -- \
  --transforms_json /root/HW3/topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300_blender.json \
  --camera_name CounterOrbitCamera \
  --collection_name CounterOrbit300 \
  --create_markers
```

Open issue:

- A combined Blender screenshot with `object_A` placed and viewed by the counter orbit camera still needs to be captured and added to `reports/figures/`.

### Threestudio DreamFusion Hamburger

- Smoke output: `/root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-smoke/hamburger_100step_20260602_v2/`
- Full output: `/root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-full/hamburger_full_20260603/`
- Full validation image: `/root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-full/hamburger_full_20260603/save/it10000-0.png`
- Full test video: `/root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-full/hamburger_full_20260603/save/it10000-test.mp4`
- Exported mesh: `/root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-full/hamburger_full_20260603/save/it10000-export/model.obj`
- Exported material: `/root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-full/hamburger_full_20260603/save/it10000-export/model.mtl`
- Exported texture: `/root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-full/hamburger_full_20260603/save/it10000-export/texture_kd.jpg`
- W&B run id: `0w5yldmf`

Commands:

```bash
bash /root/HW3/topic1_fusion/scripts/run_dreamfusion_sd15_smoke.sh
```

```bash
nohup bash /root/HW3/topic1_fusion/scripts/run_dreamfusion_sd15_full.sh \
  > /root/HW3/topic1_fusion/code/threestudio/outputs/dreamfusion-sd15-full_hamburger_20260603.nohup.log 2>&1 &
```

## Next Fixes

- Capture one Blender screenshot/video frame where `object_A` is placed in the `counter` camera/orbit scene.
- Decide object_A repair path: foreground-only point cleanup, stronger frame filtering, or explicit limitation in the final report.
- Keep large binary outputs outside Git and record external download paths when sharing with teammates.
