#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def mat_transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def mat_vec_mul(m, v):
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def negate(v):
    return [-x for x in v]


def infer_principal_point(width, height):
    # Most COLMAP/3DGS pipelines end up very close to the image center.
    return 0.5, 0.5


def camera_entry_to_cam_lines(entry):
    width = float(entry["width"])
    height = float(entry["height"])
    fx = float(entry["fx"])
    fy = float(entry["fy"])
    c2w_r = [[float(x) for x in row] for row in entry["rotation"]]
    center = [float(x) for x in entry["position"]]

    # texrecon expects world-to-camera extrinsics.
    w2c_r = mat_transpose(c2w_r)
    w2c_t = negate(mat_vec_mul(w2c_r, center))

    max_dim = max(width, height)
    flen = fx / max_dim
    paspect = fy / fx if fx != 0.0 else 1.0
    ppx, ppy = infer_principal_point(width, height)

    ext = w2c_t + [w2c_r[i][j] for i in range(3) for j in range(3)]
    intr = [flen, 0.0, 0.0, paspect, ppx, ppy]
    ext_line = " ".join(f"{x:.17g}" for x in ext)
    intr_line = " ".join(f"{x:.17g}" for x in intr)
    return ext_line, intr_line


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a texrecon scene directory from 3DGS cameras.json."
    )
    parser.add_argument("--cameras_json", required=True, help="Path to 3DGS cameras.json")
    parser.add_argument("--images_dir", required=True, help="Directory containing source images")
    parser.add_argument("--output_scene_dir", required=True, help="Output texrecon scene directory")
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images instead of symlinking them into the scene directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cameras_path = Path(args.cameras_json)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_scene_dir)

    with cameras_path.open("r", encoding="utf-8") as f:
        cameras = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_images:
        import shutil

    count = 0
    for entry in cameras:
        img_name = entry["img_name"]
        src_img = images_dir / img_name
        if not src_img.exists():
            raise FileNotFoundError(f"Missing source image: {src_img}")

        dst_img = out_dir / img_name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()

        if args.copy_images:
            shutil.copy2(src_img, dst_img)
        else:
            dst_img.symlink_to(src_img.resolve())

        ext_line, intr_line = camera_entry_to_cam_lines(entry)
        cam_path = out_dir / f"{Path(img_name).stem}.cam"
        with cam_path.open("w", encoding="utf-8") as f:
            f.write(ext_line + "\n")
            f.write(intr_line + "\n")
        count += 1

    print(f"Prepared texrecon scene with {count} views at {out_dir}")


if __name__ == "__main__":
    main()
