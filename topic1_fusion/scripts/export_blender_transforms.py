#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def matmul3(a, b):
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(sum(a[i][k] * b[k][j] for k in range(3)))
        out.append(row)
    return out


def build_transform_matrix(rotation, position):
    # 3DGS cameras.json stores camera-to-world rotation in an OpenCV-like
    # camera convention: +X right, +Y down, +Z forward.
    # Blender camera local axes are +X right, +Y up, and looks along -Z.
    cv_to_blender = [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    blender_rot = matmul3(rotation, cv_to_blender)
    return [
        [blender_rot[0][0], blender_rot[0][1], blender_rot[0][2], position[0]],
        [blender_rot[1][0], blender_rot[1][1], blender_rot[1][2], position[1]],
        [blender_rot[2][0], blender_rot[2][1], blender_rot[2][2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def camera_entry_to_frame(entry, images_dir):
    width = entry["width"]
    height = entry["height"]
    fx = entry["fx"]
    fy = entry["fy"]
    transform_matrix = build_transform_matrix(entry["rotation"], entry["position"])
    return {
        "id": entry["id"],
        "img_name": entry["img_name"],
        "file_path": str((images_dir / entry["img_name"]).resolve()) if images_dir else entry["img_name"],
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "camera_angle_x": 2.0 * math.atan(width / (2.0 * fx)),
        "camera_angle_y": 2.0 * math.atan(height / (2.0 * fy)),
        "position": entry["position"],
        "rotation_3dgs": entry["rotation"],
        "transform_matrix": transform_matrix,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 3DGS cameras.json into a Blender-friendly transforms JSON."
    )
    parser.add_argument("--cameras_json", required=True, help="Path to 3DGS cameras.json")
    parser.add_argument("--output_json", required=True, help="Output transforms_blender.json path")
    parser.add_argument(
        "--images_dir",
        default="",
        help="Optional directory containing the corresponding input images",
    )
    parser.add_argument(
        "--scene_name",
        default="scene",
        help="Scene label written into the output JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cameras_path = Path(args.cameras_json)
    output_path = Path(args.output_json)
    images_dir = Path(args.images_dir) if args.images_dir else None

    with cameras_path.open("r", encoding="utf-8") as f:
        cameras = json.load(f)

    frames = [camera_entry_to_frame(entry, images_dir) for entry in cameras]
    payload = {
        "scene_name": args.scene_name,
        "source_cameras_json": str(cameras_path.resolve()),
        "coordinate_note": (
            "transform_matrix is camera-to-world in Blender camera convention "
            "(OpenCV camera axes converted with diag(1,-1,-1))."
        ),
        "frames": frames,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(frames)} frames to {output_path}")


if __name__ == "__main__":
    main()
