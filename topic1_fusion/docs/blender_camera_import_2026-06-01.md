# Blender Camera Import for Topic 1

## Goal

Import the background-scene camera trajectory from 3DGS into Blender so we can later align and composite object A with the `counter` scene.

## Files

- Export helper: `/root/HW3/topic1_fusion/scripts/export_blender_transforms.py`
- Blender importer: `/root/HW3/topic1_fusion/scripts/blender_import_cameras.py`

## Step 1: Convert 3DGS `cameras.json`

### Counter

```bash
python /root/HW3/topic1_fusion/scripts/export_blender_transforms.py \
  --cameras_json /root/HW3/topic1_fusion/outputs/counter_30k_gpu5/cameras.json \
  --images_dir /root/HW3/topic1_fusion/data/mipnerf360/counter/images \
  --output_json /root/HW3/topic1_fusion/outputs/counter_30k_gpu5/transforms_blender.json \
  --scene_name counter_30k_gpu5
```

### Object A

```bash
python /root/HW3/topic1_fusion/scripts/export_blender_transforms.py \
  --cameras_json /root/HW3/topic1_fusion/outputs/object_A_30k/cameras.json \
  --images_dir /root/HW3/topic1_fusion/data/object_A/gs_dataset/images \
  --output_json /root/HW3/topic1_fusion/outputs/object_A_30k/transforms_blender.json \
  --scene_name object_A_30k
```

## Step 2: Import into Blender

Inside Blender:

```bash
blender --python /root/HW3/topic1_fusion/scripts/blender_import_cameras.py -- \
  --transforms_json /root/HW3/topic1_fusion/outputs/counter_30k_gpu5/transforms_blender.json \
  --camera_name CounterCamera \
  --collection_name CounterTrajectory
```

Optional:

- Add `--create_markers` to place empties at every imported camera center.

## What the importer creates

- One animated camera object
- One 3D polyline trajectory curve
- Render resolution copied from the source frames
- Frame range matched to the number of imported cameras

## Important convention note

3DGS `cameras.json` stores camera poses in a camera-to-world form, but with an OpenCV-style camera axis convention. The exporter converts that to Blender camera convention by applying `diag(1, -1, -1)` on the camera basis.

If the camera appears to face backward in Blender, the first thing to check is whether a scene-level axis conversion is also being applied elsewhere in your Blender file.
