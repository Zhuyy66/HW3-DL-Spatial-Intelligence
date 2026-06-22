# Topic1 Delivery Package

Date: 2026-06-22

This folder collects the current Topic1 deliverables that can be shared before the final fusion video is added.

## Folder Layout

- `object_C_mesh/`: Stable Zero123 final object C mesh package at 3000 iterations.
- `figures/`: Report-ready figures for counter, object A, hamburger, object C, and camera trajectory checks.
- `videos/`: Existing validation videos for counter orbit, hamburger, and object C.
- `docs/`: Report draft and submission index snapshots.
- `scripts/`: Small reproducibility scripts and run helpers.
- `place_final_fusion_video_here/`: Placeholder for the final rendered fusion video and extracted keyframes.

## Urgent Files To Share

- Object C mesh zip:
  `object_C_mesh/objectC_stable_zero123_it3000_mesh.zip`
- Object C raw mesh files:
  `object_C_mesh/model.obj`
  `object_C_mesh/model.mtl`
  `object_C_mesh/texture_kd.jpg`
- Object C validation videos:
  `videos/objectC_it3000-val.mp4`
  `videos/objectC_it3000-test.mp4`
  `videos/objectC_it10000-val.mp4`
  `videos/objectC_it10000-test.mp4`
- Existing report figures:
  `figures/`

## Object C 3000 vs 10000 Iteration Choice

The 3000-iteration Stable Zero123 run is selected as the final object C mesh version because it has a complete export package (`obj`, `mtl`, `texture_kd.jpg`) and was visually judged to be cleaner and more stable for Blender placement.

The 10000-iteration run is retained as a longer-training comparison. It has validation/test videos and checkpoint logs, but no exported mesh package was found in the current output directory. In the report, it is best described as evidence that longer SDS optimization does not necessarily improve visible quality for this input.

## Final Fusion Video

The final fused counter-scene render has been added:

- `place_final_fusion_video_here/topic1_final_fusion_counter_ABC.mp4`
- `place_final_fusion_video_here/keyframes/fusion_frame_0000.png`
- `place_final_fusion_video_here/keyframes/fusion_frame_0075.png`
- `place_final_fusion_video_here/keyframes/fusion_frame_0150.png`
- `place_final_fusion_video_here/keyframes/fusion_frame_0225.png`
- `place_final_fusion_video_here/keyframes/fusion_frame_0299.png`

The video has 300 frames at 30 fps and lasts 10 seconds.
