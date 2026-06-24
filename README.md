# HW3-DL-Spatial-Intelligence

Final project for the graduate course **Deep Learning and Spatial Intelligence**.

This repository contains two tracks:

- `topic1_fusion/`: multi-source 3D asset generation and real-scene fusion.
- `topic2_act/`: ACT policy training and cross-environment evaluation on CALVIN.

Remote repository: <https://github.com/Zhuyy66/HW3-DL-Spatial-Intelligence>

## Repository Layout

```text
topic1_fusion/     Topic 1 scripts, notes, and lightweight configs
topic2_act/        Topic 2 LeRobot/CALVIN training and evaluation code
env/               Conda environment files and setup notes
reports/           LaTeX report source and report-ready figures
assets/            Small public assets
```

Large datasets, checkpoints, meshes, videos, Blender files, generated outputs,
third-party source trees, and local logs are kept outside Git. They should be
restored from the course mirrors or the cloud package listed below.

## Cloud Artifacts

The final cloud package contains:

- Topic 2 final policies:
  - `Topic2/policies/a_only/last/`
  - `Topic2/policies/abc/last/`
- Topic 2 run metadata and splitD evaluation summaries.
- Topic 1 counter/object_A `.ply` files.
- Topic 1 counter/object_A/object_B/object_C exported meshes and textures.
- Topic 1 final Blender scene and final fusion mp4.

Cloud link and extraction code:

```text
Baidu Netdisk: https://pan.baidu.com/s/1WNUytEJB6iWO8gU2to5lOA?pwd=quw1
Extraction code: quw1
No-login download verification: verified by project member on 2026-06-24.
```

After downloading, extract the package into any local artifact directory, for
example:

```bash
mkdir -p external_artifacts
unzip HW3_week4_day4_cloud_upload_bundle.zip -d external_artifacts
```

The examples below use:

```bash
export HW3_ARTIFACT_ROOT=$PWD/external_artifacts/HW3_week4_day4_cloud_upload_bundle
```

## Environments

Two Python environments are used because LeRobot training targets modern Python,
while the official CALVIN rollout stack is Python 3.8-oriented.

Create the LeRobot/ACT environment:

```bash
conda env create -f env/environment_hw3_robot_py312.yml
conda activate env_hw3_robot
python -m pip check
```

Create the CALVIN rollout environment:

```bash
conda env create -f env/environment_hw3_calvin_py38.yml
conda activate env_hw3_calvin_eval
python -m pip check
```

GPU/CUDA compatibility depends on the machine used for reproduction. The project
was validated with CUDA-enabled PyTorch in the ACT environment. If your server
uses a different NVIDIA driver or CUDA runtime, install the matching PyTorch
wheel while keeping the same Python, LeRobot, and CALVIN-side package roles.

Topic 1 additionally needs COLMAP, Blender, 3D Gaussian Splatting, and
threestudio/Stable Zero123. The setup notes are:

```text
env/env_hw3_recon_setup.md
env/env_hw3_gen3d_setup.md
```

## Data Preparation

Topic 2 uses the official course mirror:

```text
https://huggingface.co/datasets/xiaoma26/calvin-lerobot
```

The original local raw `splitA`, `splitB`, `splitC`, and `splitD` directories
were removed during disk cleanup. Reproducers should redownload from the mirror
instead of relying on local paths from the original run.

Download or prepare the raw split data:

```bash
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p topic2_act/data/xiaoma26_calvin_lerobot

for split in splitA splitB splitC splitD; do
  python topic2_act/scripts/prepare_xiaoma_calvin_split.py \
    --repo-id xiaoma26/calvin-lerobot \
    --endpoint "$HF_ENDPOINT" \
    --local-dir topic2_act/data/xiaoma26_calvin_lerobot \
    --download-split "$split"
done
```

For the final ABC training run, prepare splitA/splitB/splitC and build the
canonical merged dataset using:

```bash
python topic2_act/scripts/build_abc_canonical_dataset.py \
  --source-root topic2_act/data/xiaoma26_calvin_lerobot \
  --output-root topic2_act/data/xiaoma26_calvin_lerobot/abc_joint_canonical_v3 \
  --episode-output-dir topic2_act/data/splits/xiaoma26_calvin_lerobot \
  --rebuild
```

## Topic 2 Train Commands

A-only training on splitA:

```bash
conda activate env_hw3_robot
mkdir -p logs/topic2_train topic2_act/outputs/act_calvin

python topic2_act/scripts/run_act_train.py \
  --dataset-root topic2_act/data/xiaoma26_calvin_lerobot/splitA_episodes_A_full_canonical_v3 \
  --prepared-dataset-root topic2_act/data/xiaoma26_calvin_lerobot/splitA_episodes_A_full_canonical_v3 \
  --episodes-file topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_A_full.json \
  --output-dir topic2_act/outputs/act_calvin/a_only_full_150k \
  --run-name act_calvin_a_only_full_150k \
  --repo-id hw3/calvin-splitA-canonical \
  --steps 150000 \
  --batch-size 8 \
  --learning-rate 1.0e-5 \
  --weight-decay 1.0e-4 \
  --chunk-size 100 \
  --n-action-steps 100 \
  --num-workers 16 \
  --prefetch-factor 4 \
  --persistent-workers \
  --save-freq 50000 \
  --overwrite \
  2>&1 | tee logs/topic2_train/a_only_full_150k.log
```

ABC training on splitA/splitB/splitC:

```bash
conda activate env_hw3_robot
mkdir -p logs/topic2_train topic2_act/outputs/act_calvin

python topic2_act/scripts/run_act_train.py \
  --dataset-root topic2_act/data/xiaoma26_calvin_lerobot/abc_joint_canonical_v3 \
  --prepared-dataset-root topic2_act/data/xiaoma26_calvin_lerobot/abc_joint_canonical_v3 \
  --episodes-file topic2_act/data/splits/xiaoma26_calvin_lerobot/episodes_ABC_full.json \
  --output-dir topic2_act/outputs/act_calvin/abc_full_150k \
  --run-name act_calvin_abc_full_150k \
  --repo-id hw3/calvin-abc-canonical \
  --steps 150000 \
  --batch-size 8 \
  --learning-rate 1.0e-5 \
  --weight-decay 1.0e-4 \
  --chunk-size 100 \
  --n-action-steps 100 \
  --num-workers 16 \
  --prefetch-factor 4 \
  --persistent-workers \
  --save-freq 50000 \
  --overwrite \
  2>&1 | tee logs/topic2_train/abc_full_150k.log
```

For grading reproduction from the cloud package, restore checkpoint aliases:

```bash
mkdir -p topic2_act/outputs/act_calvin/a_only_full_150k/lerobot_train/checkpoints
mkdir -p topic2_act/outputs/act_calvin/abc_full_150k/lerobot_train/checkpoints
mkdir -p topic2_act/outputs/act_calvin/a_only_full_150k/lerobot_train/checkpoints/last
mkdir -p topic2_act/outputs/act_calvin/abc_full_150k/lerobot_train/checkpoints/last

cp -R "$HW3_ARTIFACT_ROOT/Topic2/policies/a_only/last/." \
  topic2_act/outputs/act_calvin/a_only_full_150k/lerobot_train/checkpoints/last/
cp -R "$HW3_ARTIFACT_ROOT/Topic2/policies/abc/last/." \
  topic2_act/outputs/act_calvin/abc_full_150k/lerobot_train/checkpoints/last/
```

## Topic 2 Test Commands

Open-loop Action L1 on unseen splitD:

```bash
conda activate env_hw3_robot
mkdir -p topic2_act/eval/results logs/topic2_eval

python topic2_act/eval/run_open_loop_action_l1.py \
  --dataset-root topic2_act/data/xiaoma26_calvin_lerobot/splitD \
  --checkpoint topic2_act/outputs/act_calvin/a_only_full_150k/lerobot_train/checkpoints/last \
  --output topic2_act/eval/results/a_only_splitD_action_l1.json \
  --device cuda:0 \
  --batch-size 16 \
  2>&1 | tee logs/topic2_eval/a_only_splitD_action_l1.log

python topic2_act/eval/run_open_loop_action_l1.py \
  --dataset-root topic2_act/data/xiaoma26_calvin_lerobot/splitD \
  --checkpoint topic2_act/outputs/act_calvin/abc_full_150k/lerobot_train/checkpoints/last \
  --output topic2_act/eval/results/abc_splitD_action_l1.json \
  --device cuda:0 \
  --batch-size 16 \
  2>&1 | tee logs/topic2_eval/abc_splitD_action_l1.log
```

CALVIN closed-loop smoke, if the official CALVIN environment is available:

```bash
conda activate env_hw3_calvin_eval
mkdir -p logs/topic2_eval topic2_act/outputs/calvin_eval

python topic2_act/eval/run_calvin_eval.py \
  --calvin-root topic2_act/calvin_official \
  --dataset-path topic2_act/calvin_official/dataset/calvin_debug_dataset \
  --checkpoint topic2_act/outputs/act_calvin/abc_full_150k/lerobot_train/checkpoints/last \
  --worker-python "$(conda run -n env_hw3_robot python -c 'import sys; print(sys.executable)')" \
  --worker-device cuda:0 \
  --worker-log logs/topic2_eval/abc_worker.log \
  --eval-log-dir topic2_act/outputs/calvin_eval/abc_direct_cameras_smoke \
  --cuda-device 0 \
  --egl-policy direct-cameras \
  --single-rollout-smoke \
  --rollout-steps 60 \
  2>&1 | tee logs/topic2_eval/abc_direct_cameras_smoke.log
```

The report's main quantitative claim uses the full splitD open-loop Action L1
metric because it is deterministic over the offline dataset and independent of
machine-specific rendering availability.

## Topic 1 Reproduction Commands

Restore cloud assets:

```bash
mkdir -p topic1_fusion/external_artifacts
cp -R "$HW3_ARTIFACT_ROOT/Topic1" topic1_fusion/external_artifacts/
```

Counter orbit trajectory sampling:

```bash
python topic1_fusion/scripts/sample_camera_trajectory.py \
  --cameras_json topic1_fusion/outputs/counter_30k_gpu5/cameras.json \
  --output_json topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300.json \
  --output_blender_json topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300_blender.json \
  --num_frames 300 \
  --images_dir topic1_fusion/data/mipnerf360/counter/images \
  --scene_name counter_orbit
```

Import the sampled trajectory into Blender:

```bash
blender -b --python topic1_fusion/scripts/blender_import_cameras.py -- \
  --transforms_json topic1_fusion/outputs/counter_30k_gpu5/counter_orbit_300_blender.json \
  --camera_name CounterOrbitCamera \
  --collection_name CounterOrbit300 \
  --create_markers
```

Render the final assembled scene:

```bash
blender -b "$HW3_ARTIFACT_ROOT/Topic1/Final_fusion/Final.blend" \
  -o topic1_fusion/outputs/final_fusion/frame_#### \
  -a

ffmpeg -framerate 30 \
  -i topic1_fusion/outputs/final_fusion/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p \
  topic1_fusion/outputs/final_fusion/topic1_final_fusion_counter_ABC.mp4
```

The submitted cloud package already includes the final mp4:

```text
Topic1/Final_fusion/topic1_final_fusion_counter_ABC.mp4
```

## Report

The main LaTeX report entry is:

```text
reports/main.tex
```

Report-ready figures are stored in:

```text
reports/figures/final_report/
```
