#!/usr/bin/env python3

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1] / "code" / "gaussian-splatting"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov, getProjectionMatrix


def load_trajectory_frames(trajectory_json):
    with open(trajectory_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("frames", payload)


def camera_entry_to_minicam(entry, znear=0.01, zfar=100.0, device="cuda"):
    width = entry["width"]
    height = entry["height"]
    fx = entry["fx"]
    fy = entry["fy"]
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(entry["rotation"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(entry["position"], dtype=np.float32)
    w2c = np.linalg.inv(c2w)

    world_view = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)


def render_trajectory(dataset, iteration, pipeline, trajectory_json, output_dir, fps):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frames = load_trajectory_frames(trajectory_json)
        if not frames:
            raise ValueError("No frames found in trajectory json.")

        output_dir = Path(output_dir)
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        video_path = output_dir / "orbit.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frames[0]["width"], frames[0]["height"]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

        try:
            for idx, entry in enumerate(tqdm(frames, desc="Rendering sampled trajectory")):
                view = camera_entry_to_minicam(entry)
                rendering = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=False,
                )["render"]
                frame_path = frame_dir / f"{idx:04d}.png"
                torchvision.utils.save_image(rendering, frame_path)

                frame_np = (
                    rendering.detach()
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

        print(f"Rendered {len(frames)} frames to {frame_dir}")
        print(f"Wrote orbit video to {video_path}")
        print(f"Loaded model iteration: {scene.loaded_iter}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a sampled camera trajectory with 3DGS.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--trajectory_json", required=True, help="Sampled trajectory json path")
    parser.add_argument("--output_dir", required=True, help="Directory for frames and orbit.mp4")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    render_trajectory(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.trajectory_json,
        args.output_dir,
        args.fps,
    )
