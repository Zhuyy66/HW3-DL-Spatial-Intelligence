"""Probe the course CALVIN LeRobot dataset without downloading everything.

The Day 1 goal is to confirm the repository structure and collect enough schema
evidence for the Day 2 data audit. By default this script downloads metadata
files and one parquet episode only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="huiwon/calvin_task_ABC_D")
    parser.add_argument(
        "--local-dir",
        default="/root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_probe",
    )
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--max-meta-files", type=int, default=50)
    return parser.parse_args()


def first_or_none(items: Iterable[str]) -> str | None:
    for item in items:
        return item
    return None


def maybe_print_json(path: Path) -> None:
    if path.suffix not in {".json", ".jsonl"}:
        return

    print(f"\n--- preview: {path.name} ---")
    with path.open("r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if idx >= 5:
                break
            line = line.strip()
            if not line:
                continue
            try:
                print(json.dumps(json.loads(line), ensure_ascii=False)[:1000])
            except json.JSONDecodeError:
                print(line[:1000])


def print_parquet_schema(path: Path) -> None:
    print(f"\n--- parquet sample: {path} ---")
    frame = pd.read_parquet(path)
    print(f"rows: {len(frame)}")
    print("columns:")
    for column in frame.columns:
        value = frame[column].iloc[0] if len(frame) else None
        shape = getattr(value, "shape", None)
        print(
            f"- {column}: dtype={frame[column].dtype}, "
            f"sample_type={type(value).__name__}, shape={shape}"
        )


def main() -> int:
    args = parse_args()
    endpoint = os.environ.get("HF_ENDPOINT") or None
    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    files = api.list_repo_files(args.repo_id, repo_type=args.repo_type)
    print(f"repo: {args.repo_id}")
    print(f"file_count: {len(files)}")

    top_level = sorted({name.split("/")[0] for name in files if "/" in name})
    print("top_level_dirs:")
    for dirname in top_level:
        print(f"- {dirname}")

    meta_files = [name for name in files if "/meta/" in name]
    parquet_files = [name for name in files if name.endswith(".parquet")]
    print(f"meta_file_count: {len(meta_files)}")
    print(f"parquet_file_count: {len(parquet_files)}")

    selected_meta = meta_files[: args.max_meta_files]
    selected_parquet = first_or_none(parquet_files)
    selected = selected_meta + ([selected_parquet] if selected_parquet else [])

    downloaded: list[Path] = []
    for filename in selected:
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type=args.repo_type,
            local_dir=str(local_dir),
        )
        downloaded.append(Path(path))
        print(f"downloaded: {filename}")

    for path in downloaded:
        maybe_print_json(path)

    if selected_parquet is not None:
        sample_path = local_dir / selected_parquet
        print_parquet_schema(sample_path)
    else:
        print("warning: no parquet files found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
