# env_hw3_gen3d Setup

Environment purpose: threestudio + Stable Zero123 for text-to-3D and single-image-to-3D generation.

## Base Environment

- Python: 3.10
- CUDA: 12.1
- PyTorch: 2.4.0
- torchvision: 0.19.0
- torchaudio: 2.4.0

## Core Commands

```bash
conda create -n env_hw3_gen3d python=3.10 -y
conda activate env_hw3_gen3d

python -m pip install --upgrade pip setuptools wheel
pip install ninja

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

cd /root/HW3/topic1_fusion/code/threestudio

pip install "setuptools<70" wheel packaging
pip install pybind11
export TCNN_CUDA_ARCHITECTURES=89
pip install --no-build-isolation -r requirements.txt -c constraints_hw3.txt
pip install -e .
```

## Smoke-Test Checks

```bash
conda activate env_hw3_gen3d
cd /root/HW3/topic1_fusion/code/threestudio

python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import threestudio; print('threestudio import ok')"
python launch.py --help
```

## Day 2 Target

On 2026-05-29, the goal is only to get the installation onto a stable path:

1. PyTorch matches CUDA 12.1.
2. `requirements.txt` installs with `constraints_hw3.txt`.
3. `python launch.py --help` works, or the remaining blocker is clearly identified.


## 2026-06-01 Debug Notes

Current validated fixes for `env_hw3_gen3d`:

1. `pip install -e /root/HW3/topic1_fusion/code/threestudio`
2. `huggingface_hub` must stay below `0.26` for this `diffusers` stack.
3. `threestudio/threestudio/utils/ops.py` needed a small `libigl` compatibility fix because `libigl 2.6.2` exposes `fast_winding_number` and `readOBJ` instead of `fast_winding_number_for_meshes` and `read_obj`.

After these fixes, `conda run -n env_hw3_gen3d python -c "import threestudio"` succeeds.

## Offline Model Paths

Recommended cache root:

```bash
export HF_HOME=/root/HW3/hf_home
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Helper scripts added:

- `/root/HW3/topic1_fusion/scripts/setup_threestudio_offline_models.sh`
- `/root/HW3/topic1_fusion/scripts/verify_threestudio_offline_models.py`
- `/root/HW3/topic1_fusion/scripts/run_dreamfusion_sd15_smoke.sh`

## 100-Step DreamFusion Smoke Test

```bash
bash /root/HW3/topic1_fusion/scripts/run_dreamfusion_sd15_smoke.sh 0
```

This runs:

- config: `configs/dreamfusion-sd.yaml`
- prompt: `a hamburger`
- steps: `100`
- offline Hugging Face mode enabled
- Stable Diffusion model override: `runwayml/stable-diffusion-v1-5`
