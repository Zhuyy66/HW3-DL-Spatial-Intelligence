#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np

from export_blender_transforms import camera_entry_to_frame


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Encountered near-zero vector during trajectory construction.")
    return vec / norm


def estimate_focus_point(centers, forwards):
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for center, forward in zip(centers, forwards):
        direction = normalize(forward)
        projector = eye - np.outer(direction, direction)
        lhs += projector
        rhs += projector @ center
    return np.linalg.solve(lhs, rhs)


def fit_plane_basis(points, focus_point):
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])

    first = normalize(points[0] - focus_point)
    tangent_u = first - np.dot(first, normal) * normal
    if np.linalg.norm(tangent_u) < 1e-6:
        tangent_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_u = tangent_u - np.dot(tangent_u, normal) * normal
    tangent_u = normalize(tangent_u)
    tangent_v = normalize(np.cross(normal, tangent_u))
    return normal, tangent_u, tangent_v


def compute_angles(points, center, basis_u, basis_v):
    vectors = points - center
    x = vectors @ basis_u
    y = vectors @ basis_v
    return np.unwrap(np.arctan2(y, x))


def make_rotation_opencv(position, target, orbit_normal):
    forward = normalize(target - position)
    up_hint = orbit_normal.copy()
    if abs(np.dot(forward, up_hint)) > 0.98:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(forward, up_hint)) > 0.98:
            up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    right = normalize(np.cross(up_hint, forward))
    cam_up = normalize(np.cross(forward, right))

    rotation = np.stack([right, -cam_up, forward], axis=1)
    return rotation


def build_orbit(entries, num_frames, vertical_amplitude):
    centers = np.asarray([entry["position"] for entry in entries], dtype=np.float64)
    rotations = np.asarray([entry["rotation"] for entry in entries], dtype=np.float64)
    forwards = rotations[:, :, 2]

    focus_point = estimate_focus_point(centers, forwards)
    plane_normal, basis_u, basis_v = fit_plane_basis(centers, focus_point)

    projected_focus = focus_point
    radii = np.linalg.norm(
        np.stack(
            [
                (centers - projected_focus) @ basis_u,
                (centers - projected_focus) @ basis_v,
            ],
            axis=1,
        ),
        axis=1,
    )
    radius = float(np.median(radii))

    source_angles = compute_angles(centers, projected_focus, basis_u, basis_v)
    theta0 = float(source_angles[0])
    theta_delta = np.diff(source_angles)
    direction = 1.0 if np.median(theta_delta) >= 0.0 else -1.0

    heights = (centers - projected_focus) @ plane_normal
    height_bias = float(np.median(heights))

    width = int(np.median([entry["width"] for entry in entries]))
    height = int(np.median([entry["height"] for entry in entries]))
    fx = float(np.median([entry["fx"] for entry in entries]))
    fy = float(np.median([entry["fy"] for entry in entries]))

    orbit_entries = []
    for idx in range(num_frames):
        phase = idx / num_frames
        theta = theta0 + direction * 2.0 * math.pi * phase
        position = (
            projected_focus
            + radius * math.cos(theta) * basis_u
            + radius * math.sin(theta) * basis_v
            + (height_bias + vertical_amplitude * math.sin(theta)) * plane_normal
        )
        rotation = make_rotation_opencv(position, focus_point, plane_normal)
        orbit_entries.append(
            {
                "id": idx,
                "img_name": f"orbit_{idx:04d}.png",
                "width": width,
                "height": height,
                "position": position.tolist(),
                "rotation": rotation.tolist(),
                "fy": fy,
                "fx": fx,
            }
        )

    meta = {
        "focus_point": focus_point.tolist(),
        "plane_normal": plane_normal.tolist(),
        "basis_u": basis_u.tolist(),
        "basis_v": basis_v.tolist(),
        "radius": radius,
        "height_bias": height_bias,
        "vertical_amplitude": vertical_amplitude,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
    }
    return orbit_entries, meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a new orbit camera trajectory from counter scene cameras."
    )
    parser.add_argument("--cameras_json", required=True, help="Input 3DGS cameras.json")
    parser.add_argument("--output_json", required=True, help="Output sampled trajectory json")
    parser.add_argument(
        "--output_blender_json",
        required=True,
        help="Output Blender-friendly transforms json for the sampled trajectory",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=300,
        help="Number of output frames, e.g. 300 for 10 seconds at 30 fps",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second metadata for downstream rendering",
    )
    parser.add_argument(
        "--vertical_amplitude",
        type=float,
        default=0.0,
        help="Optional sinusoidal motion along the fitted orbit normal",
    )
    parser.add_argument(
        "--images_dir",
        default="",
        help="Optional image directory used when exporting Blender file_path fields",
    )
    parser.add_argument(
        "--scene_name",
        default="counter_orbit",
        help="Scene label written into the Blender transforms json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cameras_path = Path(args.cameras_json)
    output_json = Path(args.output_json)
    output_blender_json = Path(args.output_blender_json)
    images_dir = Path(args.images_dir) if args.images_dir else None

    with cameras_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    orbit_entries, meta = build_orbit(entries, args.num_frames, args.vertical_amplitude)
    payload = {
        "scene_name": args.scene_name,
        "source_cameras_json": str(cameras_path.resolve()),
        "fps": args.fps,
        "num_frames": args.num_frames,
        "trajectory_type": "fitted_orbit",
        "meta": meta,
        "frames": orbit_entries,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    blender_payload = {
        "scene_name": args.scene_name,
        "source_cameras_json": str(cameras_path.resolve()),
        "coordinate_note": (
            "transform_matrix is camera-to-world in Blender camera convention "
            "(OpenCV camera axes converted with diag(1,-1,-1))."
        ),
        "fps": args.fps,
        "trajectory_type": "fitted_orbit",
        "meta": meta,
        "frames": [
            camera_entry_to_frame(entry, images_dir)
            for entry in orbit_entries
        ],
    }
    output_blender_json.parent.mkdir(parents=True, exist_ok=True)
    with output_blender_json.open("w", encoding="utf-8") as f:
        json.dump(blender_payload, f, indent=2)

    print(f"Wrote sampled trajectory with {len(orbit_entries)} frames to {output_json}")
    print(f"Wrote Blender trajectory to {output_blender_json}")


if __name__ == "__main__":
    main()
