#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path

import cv2


def natural_key(path: Path):
    parts = []
    token = ""
    is_digit = None
    for ch in path.stem:
        ch_is_digit = ch.isdigit()
        if is_digit is None or ch_is_digit == is_digit:
            token += ch
        else:
            parts.append(int(token) if is_digit else token)
            token = ch
        is_digit = ch_is_digit
    if token:
        parts.append(int(token) if is_digit else token)
    parts.append(path.suffix)
    return parts


def laplacian_variance(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def write_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "source_name", "score", "selected"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Filter blurry frames using Laplacian variance.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--score_csv", type=Path, default=None)
    parser.add_argument("--target_count", type=int, default=220)
    parser.add_argument("--min_keep", type=int, default=150)
    parser.add_argument("--max_keep", type=int, default=250)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--suffix", type=str, default=".jpg")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    score_csv = args.score_csv or output_dir.parent / "logs" / "filter_blurry_scores.csv"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=natural_key,
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    scored = []
    for path in image_paths:
        score = laplacian_variance(path)
        scored.append({"path": path, "score": score})

    scored_by_sharpness = sorted(scored, key=lambda item: item["score"], reverse=True)
    eligible = [item for item in scored_by_sharpness if item["score"] >= args.min_score]
    keep_count = min(args.target_count, args.max_keep, len(eligible))
    if keep_count < args.min_keep:
        keep_count = min(len(eligible), args.min_keep)
    if keep_count == 0:
        raise RuntimeError("No frames passed the blur filter. Try lowering --min_score.")

    selected_paths = {item["path"] for item in eligible[:keep_count]}
    selected_in_time = [item for item in scored if item["path"] in selected_paths]

    if output_dir.exists():
        for old_file in output_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(selected_in_time, start=1):
        dst = output_dir / f"{idx:06d}{args.suffix}"
        if args.copy:
            shutil.copy2(item["path"], dst)
        else:
            try:
                dst.hardlink_to(item["path"])
            except OSError:
                shutil.copy2(item["path"], dst)

    csv_rows = []
    for rank, item in enumerate(scored_by_sharpness, start=1):
        csv_rows.append(
            {
                "rank": rank,
                "source_name": item["path"].name,
                "score": f"{item['score']:.6f}",
                "selected": "yes" if item["path"] in selected_paths else "no",
            }
        )
    write_csv(csv_rows, score_csv)

    selected_scores = [item["score"] for item in selected_in_time]
    print(f"Input frames: {len(scored)}")
    print(f"Eligible frames (score >= {args.min_score}): {len(eligible)}")
    print(f"Selected frames: {len(selected_in_time)}")
    print(f"Output directory: {output_dir}")
    print(f"Score CSV: {score_csv}")
    print(
        "Selected score range: "
        f"min={min(selected_scores):.3f}, median={sorted(selected_scores)[len(selected_scores)//2]:.3f}, max={max(selected_scores):.3f}"
    )


if __name__ == "__main__":
    main()
