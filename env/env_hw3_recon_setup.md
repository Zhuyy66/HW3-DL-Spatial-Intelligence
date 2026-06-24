# env_hw3_recon Setup

Environment purpose: COLMAP + 3D Gaussian Splatting for real-scene reconstruction and Mip-NeRF 360 background training.

## Base Environment

- Python: 3.10
- CUDA: 12.1
- PyTorch: 2.4.0
- torchvision: 0.19.0
- torchaudio: 2.4.0

## Create Environment

```bash
conda create -n env_hw3_recon python=3.10 -y
conda activate env_hw3_recon

python -m pip install --upgrade pip setuptools wheel
pip install ninja
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c conda-forge colmap -y
```

## Install 3DGS

```bash
cd topic1_fusion/code/gaussian-splatting

pip install -e submodules/simple-knn
pip install -e submodules/diff-gaussian-rasterization

# Optional acceleration only. Training can fall back to the Python SSIM implementation.
pip install -e submodules/fused-ssim
```

## Smoke-Test Checks

Core checks:

```bash
conda activate env_hw3_recon
cd topic1_fusion/code/gaussian-splatting

python train.py --help
python render.py --help
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import diff_gaussian_rasterization, simple_knn; print('core 3dgs extensions ok')"
colmap -h | head
```

Optional check:

```bash
python -c "import fused_ssim; print('fused_ssim ok')"
```

If the optional check fails, `train.py` will still run and automatically fall back to the non-fused SSIM path.

## Day 2 Targets

1. `python train.py --help` prints normally.
2. `diff_gaussian_rasterization` and `simple_knn` import without errors.
3. `colmap -h` works.
4. `fused_ssim` is optional on Day 2.
5. A garden or counter render job can start successfully.
