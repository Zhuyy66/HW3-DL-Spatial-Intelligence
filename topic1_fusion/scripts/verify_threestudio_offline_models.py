from pathlib import Path
import os
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_HOME = os.environ.get("HF_HOME", str(REPO_ROOT / "hf_home"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

print(f"HF_HOME={HF_HOME}")

sd15_snapshot = str(Path(HF_HOME) / "models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14")
pipe = StableDiffusionPipeline.from_pretrained(
    sd15_snapshot,
    local_files_only=True,
    safety_checker=None,
)
print("Stable Diffusion v1.5 offline load ok")
del pipe

ckpt_path = hf_hub_download(
    repo_id="stabilityai/stable-zero123",
    filename="stable_zero123.ckpt",
    cache_dir=HF_HOME,
    local_files_only=True,
)
state = torch.load(ckpt_path, map_location="cpu")
print("Stable Zero123 offline checkpoint load ok", ckpt_path)
print("Stable Zero123 top-level type:", type(state).__name__)
