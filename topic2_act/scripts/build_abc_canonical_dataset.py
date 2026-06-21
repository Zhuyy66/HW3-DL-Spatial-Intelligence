"""Build a remapped single-dataset ABC training view for HW3 Topic2.

The official splitA/splitB/splitC datasets each start episode_index at 0.
This script rewrites them into one continuous v2.1 canonical dataset, converts
that dataset to LeRobot v3.0, and writes a small remap audit manifest.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_act_train as train_utils


EXPECTED = {
    "splitA": {"scene": "A", "episodes": 6089, "frames": 366693},
    "splitB": {"scene": "B", "episodes": 6115, "frames": 367096},
    "splitC": {"scene": "C", "episodes": 5666, "frames": 337954},
}
EXPECTED_ABC_EPISODES = 17870
EXPECTED_ABC_FRAMES = 1071743
ABC_TRAIN_SCHEMA_DROP_COLUMNS = (
    "source_task_index",
    "source_split",
    "source_scene",
    "source_split_episode_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["splitA", "splitB", "splitC"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episode-output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--resume-v30",
        action="store_true",
        help="Reuse an existing <output-root>_v30 data conversion and finish official v3 metadata conversion.",
    )
    parser.add_argument(
        "--allow-datasets-disk-check-bypass",
        action="store_true",
        help="Temporarily bypass HuggingFace Datasets local disk-space precheck during converter metadata generation.",
    )
    parser.add_argument(
        "--repair-v3-train-schema",
        action="store_true",
        help="Drop ABC audit-only columns from v3 data parquet files so they match meta/info.json features.",
    )
    return parser.parse_args()


def split_root(source_root: Path, split: str) -> Path:
    return source_root / split


def load_split_context(root: Path, split: str) -> dict[str, Any]:
    dataset_root = split_root(root, split)
    info = train_utils.read_json(dataset_root / "meta" / "info.json")
    modality = train_utils.read_json(dataset_root / "meta" / "modality.json")
    episodes = train_utils.read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    stats = train_utils.read_jsonl(dataset_root / "meta" / "episodes_stats.jsonl")
    tasks = train_utils.read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    return {
        "dataset_root": dataset_root,
        "info": info,
        "modality": modality,
        "episodes": episodes,
        "stats": stats,
        "tasks": tasks,
    }


def validate_source_split(split: str, context: dict[str, Any]) -> None:
    expected = EXPECTED.get(split)
    if expected is None:
        raise ValueError(f"unsupported split {split!r}; expected one of {sorted(EXPECTED)}")
    info = context["info"]
    episodes = context["episodes"]
    stats = context["stats"]
    if str(info.get("scene")) != expected["scene"]:
        raise ValueError(f"{split}: scene {info.get('scene')!r} != {expected['scene']!r}")
    if int(info.get("total_episodes") or -1) != expected["episodes"]:
        raise ValueError(f"{split}: total_episodes {info.get('total_episodes')} != {expected['episodes']}")
    if int(info.get("total_frames") or -1) != expected["frames"]:
        raise ValueError(f"{split}: total_frames {info.get('total_frames')} != {expected['frames']}")
    if len(episodes) != expected["episodes"]:
        raise ValueError(f"{split}: episodes rows {len(episodes)} != {expected['episodes']}")
    if len(stats) != expected["episodes"]:
        raise ValueError(f"{split}: stats rows {len(stats)} != {expected['episodes']}")
    scenes = Counter(str(row.get("scene")) for row in episodes)
    if dict(scenes) != {expected["scene"]: expected["episodes"]}:
        raise ValueError(f"{split}: episode scenes {dict(scenes)} do not match {expected['scene']}")


def build_task_remap(contexts: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[int, int]]]:
    first_split = "splitA" if "splitA" in contexts else next(iter(contexts))
    reference = sorted(contexts[first_split]["tasks"], key=lambda row: int(row["task_index"]))
    reference_by_task = {str(row["task"]): int(row["task_index"]) for row in reference}
    if len(reference_by_task) != len(reference):
        raise ValueError(f"{first_split}: tasks.jsonl contains duplicate task strings")

    reference_task_set = set(reference_by_task)
    task_index_maps: dict[str, dict[int, int]] = {}
    for split, context in contexts.items():
        tasks = context["tasks"]
        current_by_task = {str(row["task"]): int(row["task_index"]) for row in tasks}
        if len(current_by_task) != len(tasks):
            raise ValueError(f"{split}: tasks.jsonl contains duplicate task strings")

        current_task_set = set(current_by_task)
        if current_task_set != reference_task_set:
            missing = sorted(reference_task_set - current_task_set)
            extra = sorted(current_task_set - reference_task_set)
            raise ValueError(
                f"{split}: tasks.jsonl task set differs from {first_split}; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

        task_index_maps[split] = {
            int(row["task_index"]): reference_by_task[str(row["task"])]
            for row in tasks
        }

    return reference, task_index_maps


def clean_output_roots(output_root: Path, rebuild: bool) -> None:
    if output_root.exists() and not rebuild:
        raise RuntimeError(f"output root already exists; pass --rebuild to replace: {output_root}")
    if not rebuild:
        return
    for suffix in ("", "_old", "_v30"):
        candidate = output_root.with_name(output_root.name + suffix)
        if candidate.exists():
            train_utils.safe_remove_tree(candidate, allowed_parent=output_root.parent)


@contextlib.contextmanager
def datasets_disk_check_bypass(enabled: bool):
    if not enabled:
        yield False
        return

    import datasets.builder as datasets_builder

    original = datasets_builder.has_sufficient_disk_space

    def _always_enough_disk_space(*_args: Any, **_kwargs: Any) -> bool:
        return True

    datasets_builder.has_sufficient_disk_space = _always_enough_disk_space
    try:
        yield True
    finally:
        datasets_builder.has_sufficient_disk_space = original


def convert_dataset_with_optional_disk_bypass(output_root: Path, allow_bypass: bool) -> dict[str, Any]:
    with datasets_disk_check_bypass(allow_bypass) as bypassed:
        train_utils.convert_prepared_subset_to_v30(output_root)
    return {
        "conversion_mode": "full",
        "datasets_disk_check_bypassed": bool(bypassed),
        "v30_partial_reused": False,
        "v30_row_count_audit": None,
    }


def data_path_for_episode(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or train_utils.DEFAULT_CHUNK_SIZE)
    template = str(info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return train_utils.source_episode_path(dataset_root, template, episode_index, chunks_size)


def first_task_index(frame: Any) -> int | None:
    if "task_index" not in frame.columns or len(frame) == 0:
        return None
    values = sorted(int(item) for item in frame["task_index"].dropna().unique().tolist())
    if len(values) != 1:
        raise ValueError(f"expected exactly one task_index per episode sample, got {values[:10]}")
    return values[0]


def remap_frame_task_index(frame: Any, split: str, task_index_map: dict[int, int]) -> tuple[int | None, int | None]:
    source_task_index = first_task_index(frame)
    if source_task_index is None:
        return None, None
    if source_task_index not in task_index_map:
        raise ValueError(f"{split}: source task_index {source_task_index} missing from canonical remap")
    frame["source_task_index"] = source_task_index
    frame["task_index"] = frame["task_index"].map(lambda value: task_index_map[int(value)]).astype("int64")
    return source_task_index, first_task_index(frame)


def materialize_abc_v21(
    *,
    source_root: Path,
    output_root: Path,
    splits: list[str],
    contexts: dict[str, dict[str, Any]],
    tasks: list[dict[str, Any]],
    task_index_maps: dict[str, dict[int, int]],
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow as pa
    from datasets import Dataset, Features, Image

    first_context = contexts[splits[0]]
    first_info = first_context["info"]
    first_modality = first_context["modality"]
    chunks_size = int(first_info.get("chunks_size") or train_utils.DEFAULT_CHUNK_SIZE)

    output_root.mkdir(parents=True, exist_ok=True)
    new_episodes: list[dict[str, Any]] = []
    new_stats: list[dict[str, Any]] = []
    remap_rows: list[dict[str, Any]] = []
    canonical_task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    total_frames = 0
    new_idx = 0

    for split in splits:
        context = contexts[split]
        episodes = sorted(context["episodes"], key=lambda row: int(row["episode_index"]))
        stats_by_idx = {int(row["episode_index"]): row for row in context["stats"]}
        info = context["info"]
        dataset_root = context["dataset_root"]

        for episode in episodes:
            old_idx = int(episode["episode_index"])
            length = int(episode["length"])
            src = data_path_for_episode(dataset_root, info, old_idx)
            if not src.exists():
                raise FileNotFoundError(f"{split}: episode parquet not found: {src}")

            chunk_idx = new_idx // chunks_size
            dst = output_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{new_idx:06d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)

            frame = pd.read_parquet(src).rename(columns=train_utils.FIELD_MAPPING)
            if len(frame) != length:
                raise ValueError(f"{split}:{old_idx}: parquet rows {len(frame)} != metadata length {length}")
            source_task_index, canonical_task_index = remap_frame_task_index(
                frame, split, task_index_maps[split]
            )
            frame["episode_index"] = new_idx
            frame["index"] = range(total_frames, total_frames + length)
            frame["source_split"] = split
            frame["source_scene"] = str(episode.get("scene"))
            frame["source_split_episode_index"] = old_idx
            if "source_episode_index" not in frame.columns:
                frame["source_episode_index"] = old_idx

            table = pa.Table.from_pandas(frame, preserve_index=False)
            features = Features.from_arrow_schema(table.schema)
            features["observation.images.image"] = Image()
            features["observation.images.wrist_image"] = Image()
            Dataset.from_pandas(frame, features=features, preserve_index=False).to_parquet(dst)

            episode_row = dict(episode)
            episode_row.update(
                {
                    "episode_index": new_idx,
                    "source_split": split,
                    "source_scene": episode.get("scene"),
                    "source_split_episode_index": old_idx,
                }
            )
            new_episodes.append(episode_row)

            stat_row = dict(stats_by_idx[old_idx])
            stat_row["episode_index"] = new_idx
            stat_row["stats"] = train_utils.canonicalize_stats(dict(stat_row.get("stats") or {}), length)
            new_stats.append(stat_row)

            remap_rows.append(
                {
                    "new_episode_index": new_idx,
                    "source_split": split,
                    "source_episode_index": old_idx,
                    "source_scene": episode.get("scene"),
                    "task_index": canonical_task_index,
                    "source_task_index": source_task_index,
                    "canonical_task_index": canonical_task_index,
                    "task": canonical_task_by_index.get(canonical_task_index),
                    "length": length,
                    "new_start_frame": total_frames,
                    "new_end_frame": total_frames + length - 1,
                }
            )
            total_frames += length
            new_idx += 1

    new_info = dict(first_info)
    new_info.update(
        {
            "codebase_version": "v2.1",
            "scene": "ABC",
            "total_episodes": len(new_episodes),
            "total_frames": total_frames,
            "total_chunks": math.ceil(len(new_episodes) / chunks_size),
            "splits": {"train": f"0:{len(new_episodes)}"},
            "features": train_utils.canonicalize_features(first_info),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "hw3_source_root": str(source_root),
            "hw3_source_splits": splits,
            "hw3_task_namespace": "splitA task ordering",
            "hw3_remap": "ABC continuous episode_index with canonical task_index",
        }
    )
    new_modality = dict(first_modality)
    new_modality["state"] = {"state": {"start": 0, "end": 15, "original_key": "observation.state"}}
    new_modality["action"] = {"action": {"start": 0, "end": 7, "original_key": "action"}}
    new_modality["video"] = {
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    }

    train_utils.write_json(output_root / "meta" / "info.json", new_info)
    train_utils.write_json(output_root / "meta" / "modality.json", new_modality)
    train_utils.write_jsonl(output_root / "meta" / "episodes.jsonl", new_episodes)
    train_utils.write_jsonl(output_root / "meta" / "episodes_stats.jsonl", new_stats)
    train_utils.write_jsonl(output_root / "meta" / "tasks.jsonl", tasks)
    train_utils.write_jsonl(output_root / "abc_remap_manifest.jsonl", remap_rows)

    return {
        "total_episodes": len(new_episodes),
        "total_frames": total_frames,
        "remap_rows": len(remap_rows),
        "scene_counts": dict(Counter(row["source_scene"] for row in remap_rows)),
        "split_counts": dict(Counter(row["source_split"] for row in remap_rows)),
        "task_count": len(tasks),
        "task_namespace": "splitA task ordering",
    }


def summarize_existing_abc_v21(output_root: Path, remap_rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    info = train_utils.read_json(output_root / "meta" / "info.json")
    return {
        "total_episodes": int(info.get("total_episodes") or len(remap_rows)),
        "total_frames": int(info.get("total_frames") or sum(int(row["length"]) for row in remap_rows)),
        "remap_rows": len(remap_rows),
        "scene_counts": dict(Counter(row["source_scene"] for row in remap_rows)),
        "split_counts": dict(Counter(row["source_split"] for row in remap_rows)),
        "task_count": len(tasks),
        "task_namespace": str(info.get("hw3_task_namespace") or "splitA task ordering"),
    }


def assert_expected_abc_counts(build_summary: dict[str, Any]) -> None:
    if build_summary["total_episodes"] != EXPECTED_ABC_EPISODES:
        raise ValueError(f"materialized episodes {build_summary['total_episodes']} != {EXPECTED_ABC_EPISODES}")
    if build_summary["total_frames"] != EXPECTED_ABC_FRAMES:
        raise ValueError(f"materialized frames {build_summary['total_frames']} != {EXPECTED_ABC_FRAMES}")


def parse_v30_data_indices(path: Path) -> tuple[int, int]:
    chunk_name = path.parent.name
    file_stem = path.stem
    try:
        chunk_index = int(chunk_name.removeprefix("chunk-"))
        if file_stem.startswith("file-"):
            file_index = int(file_stem.removeprefix("file-"))
        elif file_stem.startswith("file_"):
            file_index = int(file_stem.removeprefix("file_"))
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"unexpected v3 data path format: {path}") from exc
    return chunk_index, file_index


def v3_data_file(output_root: Path, chunk_index: int, file_index: int) -> Path:
    info = train_utils.read_json(output_root / "meta" / "info.json")
    data_path = str(info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    rel = data_path.format(chunk_index=chunk_index, file_index=file_index)
    return output_root / rel


def v3_feature_columns(output_root: Path) -> tuple[list[str], dict[str, Any]]:
    info = train_utils.read_json(output_root / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"schema repair requires a v3.0 dataset, got {info.get('codebase_version')!r}")
    features = info.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError(f"{output_root / 'meta' / 'info.json'} is missing a non-empty features object")
    return [str(key) for key in features.keys()], info


def sorted_v3_data_parquets(output_root: Path) -> list[Path]:
    files = sorted(
        (path for path in (output_root / "data").glob("chunk-*/*.parquet")),
        key=lambda path: parse_v30_data_indices(path),
    )
    if not files:
        raise FileNotFoundError(f"no v3 data parquet files found under {output_root / 'data'}")
    return files


def repair_v3_train_schema_file(path: Path, feature_columns: list[str]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    feature_set = set(feature_columns)
    drop_set = set(ABC_TRAIN_SCHEMA_DROP_COLUMNS)
    metadata = pq.read_metadata(path)
    original_columns = [str(name) for name in pq.read_schema(path).names]
    original_set = set(original_columns)
    missing_feature_columns = sorted(feature_set - original_set)
    unexpected_extra_columns = sorted(original_set - feature_set - drop_set)
    if missing_feature_columns:
        raise ValueError(f"{path}: missing feature columns {missing_feature_columns}")
    if unexpected_extra_columns:
        raise ValueError(f"{path}: unexpected extra columns {unexpected_extra_columns}")

    row_count = int(metadata.num_rows)
    dropped_columns = [column for column in ABC_TRAIN_SCHEMA_DROP_COLUMNS if column in original_set]
    if not dropped_columns and original_columns == feature_columns:
        return {
            "path": str(path),
            "rows": row_count,
            "repaired": False,
            "dropped_columns": [],
            "original_columns": original_columns,
            "final_columns": original_columns,
        }

    tmp_path = path.with_suffix(path.suffix + ".hw3_schema_repair_tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        repaired_table = pq.read_table(path, columns=feature_columns)
        if int(repaired_table.num_rows) != row_count:
            raise ValueError(f"{path}: repaired table row count changed from {row_count} to {repaired_table.num_rows}")
        pq.write_table(repaired_table, tmp_path)
        verify_columns = [str(name) for name in pq.read_schema(tmp_path).names]
        verify_rows = int(pq.read_metadata(tmp_path).num_rows)
        if verify_columns != feature_columns:
            raise ValueError(f"{tmp_path}: repaired columns {verify_columns} do not match expected {feature_columns}")
        if verify_rows != row_count:
            raise ValueError(f"{tmp_path}: verified rows {verify_rows} != original rows {row_count}")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "path": str(path),
        "rows": row_count,
        "repaired": True,
        "dropped_columns": dropped_columns,
        "original_columns": original_columns,
        "final_columns": feature_columns,
    }


def repair_v3_train_schema(output_root: Path) -> dict[str, Any]:
    feature_columns, info = v3_feature_columns(output_root)
    files = sorted_v3_data_parquets(output_root)

    scanned_files = 0
    repaired_files = 0
    skipped_files = 0
    total_rows = 0
    repaired_examples: list[dict[str, Any]] = []
    skipped_examples: list[dict[str, Any]] = []
    dropped_columns_seen: set[str] = set()
    for path in files:
        result = repair_v3_train_schema_file(path, feature_columns)
        scanned_files += 1
        total_rows += int(result["rows"])
        dropped_columns_seen.update(str(column) for column in result["dropped_columns"])
        if result["repaired"]:
            repaired_files += 1
            if len(repaired_examples) < 5:
                repaired_examples.append(result)
        else:
            skipped_files += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(result)

    expected_total_frames = int(info.get("total_frames") or -1)
    total_rows_matches_info = total_rows == expected_total_frames
    if not total_rows_matches_info:
        raise ValueError(f"repaired data rows {total_rows} != info.total_frames {expected_total_frames}")

    summary = {
        "generated_at": train_utils.utc_now(),
        "output_root": str(output_root),
        "codebase_version": info.get("codebase_version"),
        "drop_columns": list(ABC_TRAIN_SCHEMA_DROP_COLUMNS),
        "dropped_columns_seen": sorted(dropped_columns_seen),
        "feature_columns": feature_columns,
        "scanned_files": scanned_files,
        "repaired_files": repaired_files,
        "skipped_files": skipped_files,
        "total_rows": total_rows,
        "expected_total_frames": expected_total_frames,
        "total_rows_matches_info": total_rows_matches_info,
        "unexpected_extra_columns": [],
        "missing_feature_columns": [],
        "schema_matches_info_features": True,
        "repaired_examples": repaired_examples,
        "skipped_examples": skipped_examples,
    }
    train_utils.write_json(output_root / "abc_schema_repair_summary.json", summary)
    return summary


def audit_v30_data_files(v30_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    files = sorted(
        (path for path in (v30_root / "data").glob("chunk-*/*.parquet")),
        key=lambda path: parse_v30_data_indices(path),
    )
    if not files:
        raise FileNotFoundError(f"no converted v30 data parquet files found under {v30_root / 'data'}")

    data_files: list[dict[str, Any]] = []
    total_rows = 0
    for path in files:
        chunk_index, file_index = parse_v30_data_indices(path)
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        data_files.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(v30_root)),
                "chunk_index": chunk_index,
                "file_index": file_index,
                "rows": rows,
            }
        )
        total_rows += rows

    if total_rows != EXPECTED_ABC_FRAMES:
        raise ValueError(f"converted v30 data rows {total_rows} != {EXPECTED_ABC_FRAMES}")

    return {
        "file_count": len(data_files),
        "total_rows": total_rows,
        "data_files": data_files,
    }


def reconstruct_episodes_metadata_from_v30_files(v21_root: Path, v30_audit: dict[str, Any]) -> list[dict[str, int]]:
    episodes = sorted(
        train_utils.read_jsonl(v21_root / "meta" / "episodes.jsonl"),
        key=lambda row: int(row["episode_index"]),
    )
    if len(episodes) != EXPECTED_ABC_EPISODES:
        raise ValueError(f"v2.1 episodes rows {len(episodes)} != {EXPECTED_ABC_EPISODES}")

    metadata: list[dict[str, int]] = []
    episode_pos = 0
    dataset_index = 0
    for data_file in v30_audit["data_files"]:
        remaining = int(data_file["rows"])
        if remaining <= 0:
            raise ValueError(f"converted data file has no rows: {data_file['path']}")
        while remaining > 0:
            if episode_pos >= len(episodes):
                raise ValueError(f"converted data file rows exceed v2.1 episode metadata at {data_file['path']}")
            episode = episodes[episode_pos]
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            if length > remaining:
                raise ValueError(
                    f"v30 file boundary splits episode {episode_index}: length={length} remaining_rows={remaining}"
                )
            metadata.append(
                {
                    "episode_index": episode_index,
                    "data/chunk_index": int(data_file["chunk_index"]),
                    "data/file_index": int(data_file["file_index"]),
                    "dataset_from_index": dataset_index,
                    "dataset_to_index": dataset_index + length,
                }
            )
            remaining -= length
            dataset_index += length
            episode_pos += 1

    if episode_pos != len(episodes):
        raise ValueError(f"only mapped {episode_pos} episodes out of {len(episodes)}")
    if dataset_index != EXPECTED_ABC_FRAMES:
        raise ValueError(f"mapped frame count {dataset_index} != {EXPECTED_ABC_FRAMES}")
    return metadata


def finalize_resumed_roots(v21_root: Path, v30_root: Path) -> None:
    old_root = v21_root.with_name(v21_root.name + "_old")
    if old_root.exists():
        raise RuntimeError(f"refusing to overwrite existing intermediate root: {old_root}")
    if not v21_root.exists():
        raise FileNotFoundError(f"missing v2.1 root for resume: {v21_root}")
    if not v30_root.exists():
        raise FileNotFoundError(f"missing converted v30 root for resume: {v30_root}")
    shutil.move(str(v21_root), str(old_root))
    shutil.move(str(v30_root), str(v21_root))


def resume_v30_metadata_conversion(output_root: Path, allow_bypass: bool) -> dict[str, Any]:
    v30_root = output_root.with_name(output_root.name + "_v30")
    for required in (
        output_root / "meta" / "info.json",
        output_root / "meta" / "episodes.jsonl",
        output_root / "meta" / "episodes_stats.jsonl",
        output_root / "meta" / "tasks.jsonl",
        output_root / "abc_remap_manifest.jsonl",
    ):
        if not required.exists():
            raise FileNotFoundError(f"resume requires existing v2.1 artifact: {required}")

    v21_info = train_utils.read_json(output_root / "meta" / "info.json")
    if v21_info.get("codebase_version") != "v2.1":
        raise ValueError(f"resume requires a v2.1 root, got {v21_info.get('codebase_version')!r}")
    if v21_info.get("video_path") is not None:
        raise ValueError("ABC resume path expects image-only v2.1 data with video_path=None")

    v30_audit = audit_v30_data_files(v30_root)
    episodes_metadata = reconstruct_episodes_metadata_from_v30_files(output_root, v30_audit)

    from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as converter

    data_size = converter.DEFAULT_DATA_FILE_SIZE_IN_MB
    video_size = converter.DEFAULT_VIDEO_FILE_SIZE_IN_MB
    with datasets_disk_check_bypass(allow_bypass) as bypassed:
        converter.convert_info(output_root, v30_root, data_size, video_size)
        converter.convert_tasks(output_root, v30_root)
        converter.convert_episodes_metadata(output_root, v30_root, episodes_metadata, None)

    finalize_resumed_roots(output_root, v30_root)
    return {
        "conversion_mode": "resume_v30_metadata",
        "datasets_disk_check_bypassed": bool(bypassed),
        "v30_partial_reused": True,
        "v30_row_count_audit": {
            "file_count": v30_audit["file_count"],
            "total_rows": v30_audit["total_rows"],
            "first_files": v30_audit["data_files"][:5],
            "last_files": v30_audit["data_files"][-5:],
        },
    }


def load_v3_episode_file_map(output_root: Path) -> dict[int, dict[str, int]]:
    import pandas as pd

    episode_files = sorted((output_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"no v3 episode metadata parquet files found under {output_root / 'meta' / 'episodes'}")
    frames = [pd.read_parquet(path) for path in episode_files]
    episodes = pd.concat(frames, ignore_index=True)
    required = {"episode_index", "data/chunk_index", "data/file_index"}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"v3 episode metadata missing columns {missing}")
    return {
        int(row["episode_index"]): {
            "data/chunk_index": int(row["data/chunk_index"]),
            "data/file_index": int(row["data/file_index"]),
        }
        for row in episodes.to_dict("records")
    }


def validate_v3_dataset(
    *,
    output_root: Path,
    remap_rows: list[dict[str, Any]],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    import pandas as pd

    info = train_utils.read_json(output_root / "meta" / "info.json")
    stats = train_utils.read_json(output_root / "meta" / "stats.json")
    total_episodes = int(info.get("total_episodes") or -1)
    total_frames = int(info.get("total_frames") or -1)
    if total_episodes != EXPECTED_ABC_EPISODES:
        raise ValueError(f"ABC v3 total_episodes {total_episodes} != {EXPECTED_ABC_EPISODES}")
    if total_frames != EXPECTED_ABC_FRAMES:
        raise ValueError(f"ABC v3 total_frames {total_frames} != {EXPECTED_ABC_FRAMES}")

    for key in ("action", "observation.state", "observation.images.image", "observation.images.wrist_image"):
        count = stats.get(key, {}).get("count")
        if count != [EXPECTED_ABC_FRAMES]:
            raise ValueError(f"stats[{key!r}].count={count} does not cover merged ABC frames")

    required_columns = {
        "observation.state",
        "action",
        "observation.images.image",
        "observation.images.wrist_image",
        "task_index",
        "episode_index",
    }
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in remap_rows:
        by_split.setdefault(str(row["source_split"]), []).append(row)
    episode_file_map = load_v3_episode_file_map(output_root)

    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    per_split_count = max(1, math.ceil(sample_count / max(1, len(by_split))))
    for split, rows in sorted(by_split.items()):
        chosen = rng.sample(rows, min(per_split_count, len(rows)))
        samples.extend(chosen)
    samples = samples[:sample_count]

    sample_results: list[dict[str, Any]] = []
    for row in samples:
        episode_index = int(row["new_episode_index"])
        episode_file = episode_file_map.get(episode_index)
        if episode_file is None:
            raise ValueError(f"episode {episode_index}: missing v3 episode metadata row")
        parquet = v3_data_file(
            output_root,
            episode_file["data/chunk_index"],
            episode_file["data/file_index"],
        )
        if not parquet.exists():
            raise FileNotFoundError(f"v3 sample parquet not found: {parquet}")
        frame = pd.read_parquet(parquet)
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"{parquet}: missing canonical columns {missing}")
        episode_frame = frame[frame["episode_index"] == episode_index]
        if len(episode_frame) != int(row["length"]):
            raise ValueError(
                f"episode {episode_index}: sampled rows {len(episode_frame)} != length {row['length']}"
            )
        task_values = sorted(int(item) for item in episode_frame["task_index"].dropna().unique().tolist())
        expected_task_index = int(row.get("canonical_task_index", row["task_index"]))
        if expected_task_index not in task_values:
            raise ValueError(f"episode {episode_index}: task_index {expected_task_index} not in {task_values}")
        sample_results.append(
            {
                "episode_index": episode_index,
                "source_split": row["source_split"],
                "source_scene": row["source_scene"],
                "rows": len(episode_frame),
                "source_task_index": row.get("source_task_index"),
                "canonical_task_index": expected_task_index,
                "task": row.get("task"),
                "task_index": task_values,
                "parquet": str(parquet),
            }
        )

    sample_splits = Counter(item["source_split"] for item in sample_results)
    for split in ("splitA", "splitB", "splitC"):
        if sample_splits[split] == 0:
            raise ValueError(f"validation sample did not cover {split}")

    return {
        "codebase_version": info.get("codebase_version"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "normalization_stats_counts": {
            key: stats.get(key, {}).get("count")
            for key in ("action", "observation.state", "observation.images.image", "observation.images.wrist_image")
        },
        "sample_results": sample_results,
        "sample_split_counts": dict(sample_splits),
    }


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    episode_output_dir = Path(args.episode_output_dir)
    splits = list(args.splits)
    started = time.monotonic()

    if args.repair_v3_train_schema:
        if args.rebuild or args.resume_v30:
            raise ValueError("--repair-v3-train-schema cannot be combined with --rebuild or --resume-v30")
        summary = repair_v3_train_schema(output_root)
        print("ok: ABC v3 train schema repaired")
        print(f"files={summary['scanned_files']} repaired={summary['repaired_files']} skipped={summary['skipped_files']}")
        print(f"rows={summary['total_rows']} summary={output_root / 'abc_schema_repair_summary.json'}")
        return 0

    if set(splits) != {"splitA", "splitB", "splitC"}:
        raise ValueError(f"ABC build requires splitA splitB splitC, got {splits}")
    if args.sample_count < 3:
        raise ValueError("--sample-count must be at least 3 to cover A/B/C")
    if args.resume_v30 and args.rebuild:
        raise ValueError("--resume-v30 reuses existing roots; do not combine it with --rebuild")

    contexts = {split: load_split_context(source_root, split) for split in splits}
    for split, context in contexts.items():
        validate_source_split(split, context)
    tasks, task_index_maps = build_task_remap(contexts)

    if args.resume_v30:
        remap_rows = train_utils.read_jsonl(output_root / "abc_remap_manifest.jsonl")
        build_summary = summarize_existing_abc_v21(output_root, remap_rows, tasks)
        assert_expected_abc_counts(build_summary)
        print(f"resuming ABC v3 metadata conversion from {output_root.with_name(output_root.name + '_v30')}", flush=True)
        conversion_audit = resume_v30_metadata_conversion(
            output_root,
            allow_bypass=args.allow_datasets_disk_check_bypass,
        )
    else:
        clean_output_roots(output_root, rebuild=args.rebuild)
        build_summary = materialize_abc_v21(
            source_root=source_root,
            output_root=output_root,
            splits=splits,
            contexts=contexts,
            tasks=tasks,
            task_index_maps=task_index_maps,
        )
        assert_expected_abc_counts(build_summary)
        remap_rows = train_utils.read_jsonl(output_root / "abc_remap_manifest.jsonl")
        print(f"materialized ABC v2.1 canonical dataset at {output_root}", flush=True)
        print(f"episodes={build_summary['total_episodes']} frames={build_summary['total_frames']}", flush=True)
        print("converting ABC dataset to LeRobot v3.0", flush=True)
        conversion_audit = convert_dataset_with_optional_disk_bypass(
            output_root,
            allow_bypass=args.allow_datasets_disk_check_bypass,
        )

    train_utils.write_jsonl(output_root / "abc_remap_manifest.jsonl", remap_rows)
    validation = validate_v3_dataset(
        output_root=output_root,
        remap_rows=remap_rows,
        sample_count=args.sample_count,
        seed=args.seed,
    )

    episode_output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = episode_output_dir / "episodes_ABC_full.json"
    train_utils.write_json(episodes_path, list(range(EXPECTED_ABC_EPISODES)))

    summary = {
        "generated_at": train_utils.utc_now(),
        "source_root": str(source_root),
        "source_splits": splits,
        "output_root": str(output_root),
        "episodes_file": str(episodes_path),
        "expected_episodes": EXPECTED_ABC_EPISODES,
        "expected_frames": EXPECTED_ABC_FRAMES,
        "build_summary": build_summary,
        "validation": validation,
        "conversion_mode": conversion_audit["conversion_mode"],
        "datasets_disk_check_bypassed": conversion_audit["datasets_disk_check_bypassed"],
        "v30_partial_reused": conversion_audit["v30_partial_reused"],
        "v30_row_count_audit": conversion_audit["v30_row_count_audit"],
        "elapsed_seconds": time.monotonic() - started,
    }
    train_utils.write_json(output_root / "abc_canonical_summary.json", summary)

    print("ok: ABC canonical dataset validated")
    print(f"episodes={validation['total_episodes']} frames={validation['total_frames']}")
    print(f"episodes_file={episodes_path}")
    print(f"summary={output_root / 'abc_canonical_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - preserve concise server diagnostics.
        traceback.print_exc(file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
