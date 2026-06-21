"""Run HW3 Topic2 ACT training through the LeRobot 0.4.0 training pipeline.

The official xiaoma26 split uses LeRobot v2.1 metadata and non-canonical
feature keys (`state`, `actions`, `image`, `wrist_image`). LeRobot 0.4.0
discovers policy inputs from canonical keys, so this wrapper first materializes
a selected v2.1 subset with canonical columns, converts that subset to v3.0
using LeRobot's official converter, then calls the official training function.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIELD_MAPPING = {
    "state": "observation.state",
    "actions": "action",
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}
IMAGE_FEATURE_KEYS = ("observation.images.image", "observation.images.wrist_image")
IMAGENET_IMAGE_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}
IMAGE_MIN = [[[0.0]], [[0.0]], [[0.0]]]
IMAGE_MAX = [[[1.0]], [[1.0]], [[1.0]]]
SCRIPT_VERSION = 3
DEFAULT_SEED = 20260529
DEFAULT_CHUNK_SIZE = 1000
REQUIRED_MAIN_ARGS = ("dataset_root", "episodes_file", "output_dir", "run_name")
RUNTIME_ENV_DEFAULTS = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root")
    parser.add_argument("--episodes-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run-batches", type=int, default=0)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "hw3-topic2"))
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--prepared-dataset-root", default=None)
    parser.add_argument("--rebuild-prepared-dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-datasets-disk-check-bypass",
        action="store_true",
        help="Temporarily bypass HuggingFace Datasets local disk-space precheck inside the LeRobot worker.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-id", default="hw3/calvin-splitA-canonical")
    parser.add_argument("--_worker-config", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args._worker_config:
        missing = [f"--{name.replace('_', '-')}" for name in REQUIRED_MAIN_ARGS if getattr(args, name) in (None, "")]
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def merge_manifest_fields(path: Path, updates: dict[str, Any]) -> None:
    if not path.exists():
        return
    try:
        manifest = read_json(path)
    except Exception as exc:  # noqa: BLE001 - audit writing should not mask training.
        print(f"warning: failed to read manifest for audit merge: {path}: {exc}", flush=True)
        return
    manifest.update(updates)
    write_json(path, manifest)


def apply_runtime_env(env: MutableMapping[str, str] | None = None) -> dict[str, str]:
    target = os.environ if env is None else env
    for key, value in RUNTIME_ENV_DEFAULTS.items():
        target[key] = value
    return {key: target[key] for key in RUNTIME_ENV_DEFAULTS}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_seconds(value: float) -> str:
    return f"{value:.3f}"


def safe_remove_tree(path: Path, *, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if parent not in resolved.parents and resolved != parent:
        raise ValueError(f"refusing to remove {resolved}; it is not under {parent}")
    if path.exists():
        shutil.rmtree(path)


def same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def default_prepared_root(dataset_root: Path, episodes_file: Path) -> Path:
    name = f"{dataset_root.name}_{episodes_file.stem}_canonical_v3"
    return dataset_root.parent / name


def source_episode_path(dataset_root: Path, data_path_template: str, episode_index: int, chunks_size: int) -> Path:
    episode_chunk = episode_index // chunks_size
    rel = data_path_template.format(episode_chunk=episode_chunk, episode_index=episode_index)
    return dataset_root / rel


def canonicalize_features(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json is missing a features object")

    renamed: dict[str, Any] = {}
    for key, value in features.items():
        renamed[FIELD_MAPPING.get(key, key)] = value
    return renamed


def ensure_stat_count(feature_stats: dict[str, Any], episode_length: int) -> dict[str, Any]:
    updated = dict(feature_stats)
    updated.setdefault("count", [episode_length])
    return updated


def image_placeholder_stats(episode_length: int) -> dict[str, Any]:
    return {
        "min": IMAGE_MIN,
        "max": IMAGE_MAX,
        "mean": IMAGENET_IMAGE_STATS["mean"],
        "std": IMAGENET_IMAGE_STATS["std"],
        "count": [episode_length],
    }


def canonicalize_stats(stats: dict[str, Any], episode_length: int) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    for key, value in stats.items():
        canonical_key = FIELD_MAPPING.get(key, key)
        if isinstance(value, dict):
            renamed[canonical_key] = ensure_stat_count(value, episode_length)
        else:
            renamed[canonical_key] = value
    for image_key in IMAGE_FEATURE_KEYS:
        renamed.setdefault(image_key, image_placeholder_stats(episode_length))
    return renamed


def selected_episode_metadata(dataset_root: Path, selected: list[int]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    stats = read_jsonl(dataset_root / "meta" / "episodes_stats.jsonl")
    episodes_by_idx = {int(row["episode_index"]): row for row in episodes}
    stats_by_idx = {int(row["episode_index"]): row for row in stats}
    missing = [idx for idx in selected if idx not in episodes_by_idx or idx not in stats_by_idx]
    if missing:
        raise ValueError(f"episodes file references missing episodes; first missing={missing[:10]}")
    return episodes_by_idx, stats_by_idx


def manifest_matches(path: Path, expected_hash: str) -> bool:
    if not path.exists():
        return False
    info_path = path / "meta" / "info.json"
    manifest_path = path / ".hw3_prepare_manifest.json"
    if not info_path.exists() or not manifest_path.exists():
        return False
    try:
        info = read_json(info_path)
        manifest = read_json(manifest_path)
    except Exception:
        return False
    return info.get("codebase_version") == "v3.0" and manifest.get("input_hash") == expected_hash


def read_v3_episode_lengths(dataset_root: Path) -> dict[int, int]:
    import pandas as pd

    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"no v3 episode metadata parquet files found under {dataset_root / 'meta' / 'episodes'}")
    frames = [pd.read_parquet(path, columns=["episode_index", "length"]) for path in episode_files]
    episodes = pd.concat(frames, ignore_index=True)
    required = {"episode_index", "length"}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"v3 episode metadata missing columns {missing}")
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes.to_dict("records")}
    if len(lengths) != len(episodes):
        raise ValueError("v3 episode metadata contains duplicate episode_index values")
    return lengths


def selected_v3_episode_lengths(dataset_root: Path, selected: list[int]) -> dict[int, int]:
    lengths = read_v3_episode_lengths(dataset_root)
    missing = [idx for idx in selected if idx not in lengths]
    if missing:
        raise ValueError(f"episodes file references missing v3 episodes; first missing={missing[:10]}")
    return {idx: lengths[idx] for idx in selected}


def materialize_canonical_v21_subset(
    *,
    dataset_root: Path,
    selected: list[int],
    prepared_root: Path,
    expected_hash: str,
    rebuild: bool,
) -> None:
    import pandas as pd
    import pyarrow as pa
    from datasets import Dataset, Features, Image

    info = read_json(dataset_root / "meta" / "info.json")
    modality = read_json(dataset_root / "meta" / "modality.json")
    tasks = read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    chunks_size = int(info.get("chunks_size") or DEFAULT_CHUNK_SIZE)
    data_path_template = str(info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    episodes_by_idx, stats_by_idx = selected_episode_metadata(dataset_root, selected)

    if prepared_root.exists():
        if rebuild:
            safe_remove_tree(prepared_root, allowed_parent=prepared_root.parent)
        else:
            raise RuntimeError(f"prepared root already exists but is not reusable: {prepared_root}")

    prepared_root.mkdir(parents=True, exist_ok=True)
    new_episodes: list[dict[str, Any]] = []
    new_stats: list[dict[str, Any]] = []
    total_frames = 0

    for new_idx, old_idx in enumerate(selected):
        old_episode = episodes_by_idx[old_idx]
        length = int(old_episode["length"])
        src = source_episode_path(dataset_root, data_path_template, old_idx, chunks_size)
        if not src.exists():
            raise FileNotFoundError(f"episode parquet not found: {src}")

        chunk_idx = new_idx // chunks_size
        dst = prepared_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{new_idx:06d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)

        frame = pd.read_parquet(src).rename(columns=FIELD_MAPPING)
        if len(frame) != length:
            raise ValueError(f"{src} has {len(frame)} rows but episodes.jsonl says {length}")
        frame["episode_index"] = new_idx
        frame["index"] = range(total_frames, total_frames + length)
        if "source_episode_index" not in frame.columns:
            frame["source_episode_index"] = old_idx

        table = pa.Table.from_pandas(frame, preserve_index=False)
        features = Features.from_arrow_schema(table.schema)
        features["observation.images.image"] = Image()
        features["observation.images.wrist_image"] = Image()
        Dataset.from_pandas(frame, features=features, preserve_index=False).to_parquet(dst)

        new_episode = dict(old_episode)
        new_episode.update(
            {
                "episode_index": new_idx,
                "source_episode_index": old_idx,
                "source_split_episode_index": old_idx,
                "source_scene": old_episode.get("scene"),
            }
        )
        new_episodes.append(new_episode)

        stat_row = dict(stats_by_idx[old_idx])
        stat_row["episode_index"] = new_idx
        stat_row["stats"] = canonicalize_stats(dict(stat_row.get("stats") or {}), length)
        new_stats.append(stat_row)
        total_frames += length

    new_info = dict(info)
    new_info.update(
        {
            "codebase_version": "v2.1",
            "total_episodes": len(selected),
            "total_frames": total_frames,
            "total_chunks": math.ceil(len(selected) / chunks_size),
            "splits": {"train": f"0:{len(selected)}"},
            "features": canonicalize_features(info),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "hw3_source_root": str(dataset_root),
            "hw3_selected_episode_count": len(selected),
            "hw3_input_hash": expected_hash,
        }
    )

    new_modality = dict(modality)
    new_modality["state"] = {"state": {"start": 0, "end": 15, "original_key": "observation.state"}}
    new_modality["action"] = {"action": {"start": 0, "end": 7, "original_key": "action"}}
    new_modality["video"] = {
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    }

    write_json(prepared_root / "meta" / "info.json", new_info)
    write_json(prepared_root / "meta" / "modality.json", new_modality)
    write_jsonl(prepared_root / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(prepared_root / "meta" / "episodes_stats.jsonl", new_stats)
    write_jsonl(prepared_root / "meta" / "tasks.jsonl", tasks)
    write_json(
        prepared_root / "meta" / "conversion.json",
        {
            "source_dataset": "xiaoma26/calvin-lerobot",
            "source_root": str(dataset_root),
            "selected_episode_count": len(selected),
            "input_hash": expected_hash,
            "field_mapping": FIELD_MAPPING,
        },
    )


def convert_prepared_subset_to_v30(prepared_root: Path) -> None:
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    convert_dataset(
        repo_id=prepared_root.name,
        root=prepared_root.parent,
        push_to_hub=False,
        force_conversion=True,
    )


def prepare_dataset(
    *,
    dataset_root: Path,
    episodes_file: Path,
    prepared_root: Path,
    rebuild: bool,
) -> tuple[Path, list[int], dict[str, Any]]:
    selected = [int(item) for item in read_json(episodes_file)]
    if not selected:
        raise ValueError("episodes file is empty")

    source_info = read_json(dataset_root / "meta" / "info.json")
    source_codebase_version = str(source_info.get("codebase_version") or "")
    input_hash = stable_hash(
        {
            "script_version": SCRIPT_VERSION,
            "dataset_root": str(dataset_root),
            "episodes_file": str(episodes_file),
            "episodes": selected,
            "source_info_hash": stable_hash(source_info),
            "field_mapping": FIELD_MAPPING,
        }
    )
    prepared_is_source = same_resolved_path(dataset_root, prepared_root)

    if source_codebase_version == "v3.0":
        if not prepared_is_source:
            raise RuntimeError(
                "v3 dataset roots must be used with --prepared-dataset-root equal to --dataset-root; "
                f"got dataset_root={dataset_root} prepared_root={prepared_root}"
            )
        if rebuild:
            raise RuntimeError("refusing --rebuild-prepared-dataset for v3 passthrough input")
        selected_lengths = selected_v3_episode_lengths(dataset_root, selected)
        manifest = {
            "generated_at": utc_now(),
            "script_version": SCRIPT_VERSION,
            "input_hash": input_hash,
            "source_dataset_root": str(dataset_root),
            "source_codebase_version": source_codebase_version,
            "episodes_file": str(episodes_file),
            "selected_episode_count": len(selected),
            "selected_frame_count": sum(selected_lengths.values()),
            "prepared_root": str(prepared_root),
            "prepared_dataset_passthrough": True,
            "field_mapping": FIELD_MAPPING,
            "elapsed_seconds": 0.0,
        }
        write_json(prepared_root / ".hw3_prepare_manifest.json", manifest)
        print(f"ok: using existing LeRobot v3 dataset via passthrough at {prepared_root}", flush=True)
        return prepared_root, selected, manifest

    if prepared_is_source:
        raise RuntimeError(
            "refusing to prepare a non-v3 dataset in place; choose a separate --prepared-dataset-root "
            f"for dataset_root={dataset_root}"
        )

    if manifest_matches(prepared_root, input_hash) and not rebuild:
        manifest = read_json(prepared_root / ".hw3_prepare_manifest.json")
        print(f"ok: reusing prepared canonical v3 dataset at {prepared_root}", flush=True)
        return prepared_root, selected, manifest

    for suffix in ("", "_old", "_v30"):
        candidate = prepared_root.with_name(prepared_root.name + suffix)
        if candidate.exists():
            safe_remove_tree(candidate, allowed_parent=prepared_root.parent)

    print(f"materializing canonical v2.1 subset: {prepared_root}", flush=True)
    start = time.monotonic()
    materialize_canonical_v21_subset(
        dataset_root=dataset_root,
        selected=selected,
        prepared_root=prepared_root,
        expected_hash=input_hash,
        rebuild=rebuild,
    )
    print("converting canonical subset to LeRobot v3.0 with official converter", flush=True)
    convert_prepared_subset_to_v30(prepared_root)
    elapsed = time.monotonic() - start

    manifest = {
        "generated_at": utc_now(),
        "script_version": SCRIPT_VERSION,
        "input_hash": input_hash,
        "source_dataset_root": str(dataset_root),
        "source_codebase_version": source_codebase_version,
        "episodes_file": str(episodes_file),
        "selected_episode_count": len(selected),
        "prepared_root": str(prepared_root),
        "prepared_dataset_passthrough": False,
        "field_mapping": FIELD_MAPPING,
        "elapsed_seconds": elapsed,
    }
    write_json(prepared_root / ".hw3_prepare_manifest.json", manifest)
    print(f"ok: prepared canonical v3 dataset in {parse_seconds(elapsed)}s", flush=True)
    return prepared_root, selected, manifest


def compute_episode_frames(dataset_root: Path, selected: list[int]) -> tuple[int, dict[int, int]]:
    source_info = read_json(dataset_root / "meta" / "info.json")
    if source_info.get("codebase_version") == "v3.0":
        lengths = selected_v3_episode_lengths(dataset_root, selected)
        return sum(lengths.values()), lengths
    episodes_by_idx, _ = selected_episode_metadata(dataset_root, selected)
    lengths = {idx: int(episodes_by_idx[idx]["length"]) for idx in selected}
    return sum(lengths.values()), lengths


def parse_compact_number(raw_value: str) -> int | float:
    suffix = raw_value[-1:].upper()
    multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix)
    number_text = raw_value[:-1] if multiplier else raw_value
    value = float(number_text) if any(char in number_text.lower() for char in (".", "e")) else int(number_text)
    if multiplier:
        value = value * multiplier
        return int(value) if float(value).is_integer() else value
    return value


def parse_metric_line(line: str) -> dict[str, Any] | None:
    stripped = strip_ansi(line)
    if "loss" not in stripped or "step" not in stripped:
        return None
    metrics: dict[str, Any] = {"raw": stripped}
    for key, raw_value in re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*):\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[KMBkmb]?)",
        stripped,
    ):
        normalized = {"epch": "epoch", "grdn": "grad_norm"}.get(key, key)
        try:
            value = parse_compact_number(raw_value)
        except ValueError:
            continue
        metrics[normalized] = value
    return metrics if "loss" in metrics else None


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def parse_wandb_url(line: str) -> str | None:
    clean = strip_ansi(line)
    match = re.search(r"https://wandb\.ai/\S+", clean)
    if not match:
        return None
    return match.group(0).rstrip(").,")


def scalar_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "mean"):
            value = value.mean()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return None


def worker_config_to_train_cfg(config: dict[str, Any]):
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    policy = ACTConfig(
        device="cuda",
        push_to_hub=False,
        chunk_size=int(config["chunk_size"]),
        n_action_steps=int(config["n_action_steps"]),
        optimizer_lr=float(config["learning_rate"]),
        optimizer_lr_backbone=float(config["learning_rate"]),
        optimizer_weight_decay=float(config["weight_decay"]),
    )
    dataset = DatasetConfig(
        repo_id=str(config["repo_id"]),
        root=str(config["prepared_dataset_root"]),
        episodes=None,
        use_imagenet_stats=True,
    )
    wandb = WandBConfig(
        enable=bool(config["wandb_enabled"]),
        project=str(config["wandb_project"]),
        disable_artifact=True,
    )
    return TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        output_dir=Path(config["train_output_dir"]),
        job_name=str(config["run_name"]),
        seed=int(config["seed"]),
        num_workers=int(config["num_workers"]),
        batch_size=int(config["batch_size"]),
        steps=int(config["steps"]),
        eval_freq=0,
        log_freq=int(config["log_freq"]),
        save_checkpoint=True,
        save_freq=int(config["save_freq"]),
        use_policy_training_preset=True,
        wandb=wandb,
        rename_map={},
    )


def install_dataloader_audit_patch(config: dict[str, Any]):
    import torch

    original_dataloader = torch.utils.data.DataLoader
    audit_path = Path(config["dataloader_audit_path"]) if config.get("dataloader_audit_path") else None
    manifest_path = Path(config["manifest_path"]) if config.get("manifest_path") else None
    prefetch_factor = int(config.get("dataloader_prefetch_factor", 2))
    use_persistent_workers = bool(config.get("dataloader_persistent_workers", False))
    audits: list[dict[str, Any]] = []

    class AuditedDataLoader(original_dataloader):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            num_workers = int(kwargs.get("num_workers") or 0)
            if num_workers > 0:
                kwargs["prefetch_factor"] = prefetch_factor
                if use_persistent_workers:
                    kwargs["persistent_workers"] = True
            audit = {
                "event": "dataloader_monkeypatch_audit",
                "captured_at": utc_now(),
                "num_workers": num_workers,
                "batch_size": kwargs.get("batch_size"),
                "prefetch_factor": kwargs.get("prefetch_factor"),
                "persistent_workers": bool(kwargs.get("persistent_workers", False)),
                "pin_memory": bool(kwargs.get("pin_memory", False)),
                "drop_last": bool(kwargs.get("drop_last", False)),
                "sampler_type": type(kwargs.get("sampler")).__name__ if kwargs.get("sampler") is not None else None,
                "shuffle": kwargs.get("shuffle"),
            }
            audits.append(audit)
            print(f"dataloader_monkeypatch_audit: {json.dumps(audit, ensure_ascii=False, sort_keys=True)}", flush=True)
            if audit_path:
                append_jsonl(audit_path, audit)
            if manifest_path:
                merge_manifest_fields(manifest_path, {"dataloader_monkeypatch_audits": audits})
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = AuditedDataLoader
    return torch, original_dataloader


def install_action_component_audit_patch(config: dict[str, Any]):
    from lerobot.policies.act.modeling_act import ACTPolicy

    original_forward = ACTPolicy.forward
    audit_path = Path(config["loss_components_path"]) if config.get("loss_components_path") else None
    manifest_path = Path(config["manifest_path"]) if config.get("manifest_path") else None
    log_freq = max(1, int(config.get("log_freq") or 1))
    action_l1_source = "ACTPolicy.forward loss_dict['l1_loss']"
    step_counter = {"value": 0}
    audit_handle = None

    if audit_path:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_handle = audit_path.open("a", encoding="utf-8", buffering=1)

    if manifest_path:
        merge_manifest_fields(
            manifest_path,
            {
                "loss_components_path": str(audit_path) if audit_path else None,
                "action_l1_source": action_l1_source,
            },
        )

    def record_action_l1(output_dict: dict[str, Any], loss_value: Any, forward_return_type: str) -> None:
        step_counter["value"] += 1
        raw_l1 = output_dict.get("l1_loss")
        action_l1 = scalar_metric(raw_l1)
        total_loss = scalar_metric(loss_value)
        if total_loss is None:
            total_loss = scalar_metric(output_dict.get("loss"))
        if raw_l1 is not None:
            output_dict.setdefault("action_l1", raw_l1)

        row = {
            "event": "action_component_metrics",
            "captured_at": utc_now(),
            "step": step_counter["value"],
            "loss": total_loss,
            "action_l1": action_l1,
            "source": action_l1_source,
            "forward_return_type": forward_return_type,
        }
        if audit_handle is not None:
            audit_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if step_counter["value"] == 1 or step_counter["value"] % log_freq == 0:
            print(f"action_component_metrics: {json.dumps(row, ensure_ascii=False, sort_keys=True)}", flush=True)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_forward(self, *args, **kwargs)
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            loss, output_dict = output
            record_action_l1(output_dict, loss, "tuple")
            return output
        if isinstance(output, dict):
            record_action_l1(output, output.get("loss"), "dict")
            return output
        return output

    ACTPolicy.forward = patched_forward

    def restore() -> None:
        ACTPolicy.forward = original_forward
        if audit_handle is not None:
            audit_handle.close()

    return restore


def run_worker(config_path: Path) -> int:
    runtime_env = apply_runtime_env()
    config = read_json(config_path)
    allow_disk_bypass = bool(config.get("allow_datasets_disk_check_bypass", False))
    print(
        "starting LeRobot worker from "
        f"{config_path}; runtime_env={json.dumps(runtime_env, sort_keys=True)}; "
        f"datasets_disk_check_bypass={allow_disk_bypass}",
        flush=True,
    )

    from lerobot.scripts.lerobot_train import train

    train_cfg = worker_config_to_train_cfg(config)
    torch_module, original_dataloader = install_dataloader_audit_patch(config)
    restore_action_audit = install_action_component_audit_patch(config)
    try:
        with datasets_disk_check_bypass(allow_disk_bypass) as bypassed:
            if bypassed:
                print("warning: HuggingFace Datasets disk-space precheck bypass enabled", flush=True)
            train(train_cfg)
    finally:
        restore_action_audit()
        torch_module.utils.data.DataLoader = original_dataloader
    return 0


def stream_worker(worker_config_path: Path, metrics_path: Path, manifest_path: Path, manifest: dict[str, Any]) -> int:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker-config", str(worker_config_path)]
    child_env = os.environ.copy()
    child_runtime_env = apply_runtime_env(child_env)
    manifest["worker_command"] = cmd
    manifest["worker_runtime_env"] = child_runtime_env
    write_json(manifest_path, manifest)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    wandb_url: str | None = None
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            metric = parse_metric_line(line)
            if metric:
                metric["captured_at"] = utc_now()
                metrics_file.write(json.dumps(metric, ensure_ascii=False) + "\n")
                metrics_file.flush()
            url = parse_wandb_url(line)
            if url:
                wandb_url = url
                manifest["wandb_url"] = wandb_url
                merge_manifest_fields(manifest_path, {"wandb_url": wandb_url})

        return_code = proc.wait()

    manifest["wandb_url"] = wandb_url or manifest.get("wandb_url")
    return return_code


def find_checkpoints(run_dir: Path) -> list[str]:
    candidates = []
    for pattern in (
        "lerobot_train/checkpoints/**/*",
        "checkpoints/**/*",
    ):
        for path in run_dir.glob(pattern):
            if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
                candidates.append(str(path))
    return sorted(set(candidates))


def main() -> int:
    args = parse_args()
    if args._worker_config:
        return run_worker(Path(args._worker_config))

    runtime_env = apply_runtime_env()

    if args.epochs is None and args.steps is None and not args.dry_run_batches:
        raise ValueError("one of --epochs, --steps, or --dry-run-batches must be provided")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.save_freq is not None and args.save_freq <= 0:
        raise ValueError("--save-freq must be positive")
    if args.persistent_workers and args.num_workers <= 0:
        raise ValueError("--persistent-workers requires --num-workers > 0")
    if args.dry_run_batches < 0:
        raise ValueError("--dry-run-batches cannot be negative")

    dataset_root = Path(args.dataset_root)
    episodes_file = Path(args.episodes_file)
    run_dir = Path(args.output_dir)
    train_output_dir = run_dir / "lerobot_train"
    prepared_root = Path(args.prepared_dataset_root) if args.prepared_dataset_root else default_prepared_root(dataset_root, episodes_file)

    if train_output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"training output already exists; pass --overwrite to rebuild: {train_output_dir}")
    if run_dir.exists() and args.overwrite:
        safe_remove_tree(run_dir, allowed_parent=run_dir.parent)
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_root, selected, prepared_manifest = prepare_dataset(
        dataset_root=dataset_root,
        episodes_file=episodes_file,
        prepared_root=prepared_root,
        rebuild=args.rebuild_prepared_dataset,
    )

    frame_count, episode_lengths = compute_episode_frames(dataset_root, selected)
    steps_per_epoch = math.ceil(frame_count / args.batch_size)
    if args.dry_run_batches:
        planned_steps = args.dry_run_batches
        step_budget_source = "dry_run_batches"
    elif args.steps is not None:
        planned_steps = args.steps
        step_budget_source = "steps"
    else:
        planned_steps = steps_per_epoch * int(args.epochs)
        step_budget_source = "epochs"
    if args.save_freq is not None:
        save_freq = min(int(args.save_freq), planned_steps)
    elif step_budget_source == "epochs" and args.save_every_epoch:
        save_freq = steps_per_epoch
    else:
        save_freq = planned_steps
    log_freq = min(max(1, args.log_freq), planned_steps)

    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.jsonl"
    loss_components_path = run_dir / "loss_components.jsonl"
    dataloader_audit_path = run_dir / "dataloader_audit.jsonl"
    worker_config_path = run_dir / "actual_training_config.json"
    worker_config = {
        "run_name": args.run_name,
        "repo_id": args.repo_id,
        "prepared_dataset_root": str(prepared_root),
        "train_output_dir": str(train_output_dir),
        "epochs": args.epochs,
        "requested_steps": args.steps,
        "step_budget_source": step_budget_source,
        "batch_size": args.batch_size,
        "steps": planned_steps,
        "steps_per_epoch": steps_per_epoch,
        "dry_run_batches": args.dry_run_batches,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "num_workers": args.num_workers,
        "dataloader_prefetch_factor": args.prefetch_factor,
        "dataloader_persistent_workers": args.persistent_workers,
        "dataloader_audit_path": str(dataloader_audit_path),
        "loss_components_path": str(loss_components_path),
        "action_l1_source": "ACTPolicy.forward loss_dict['l1_loss']",
        "manifest_path": str(manifest_path),
        "log_freq": log_freq,
        "save_freq": save_freq,
        "wandb_enabled": not args.disable_wandb,
        "wandb_project": args.wandb_project,
        "allow_datasets_disk_check_bypass": args.allow_datasets_disk_check_bypass,
        "runtime_env": runtime_env,
    }
    write_json(worker_config_path, worker_config)

    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "run_name": args.run_name,
        "dataset_root": str(dataset_root),
        "episodes_file": str(episodes_file),
        "source_codebase_version": prepared_manifest.get("source_codebase_version"),
        "selected_episode_count": len(selected),
        "selected_frame_count": frame_count,
        "selected_episode_length_min": min(episode_lengths.values()),
        "selected_episode_length_max": max(episode_lengths.values()),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "requested_steps": args.steps,
        "fixed_step_budget": planned_steps if step_budget_source == "steps" else None,
        "step_budget_source": step_budget_source,
        "steps_per_epoch": steps_per_epoch,
        "planned_steps": planned_steps,
        "dry_run_batches": args.dry_run_batches,
        "num_workers": args.num_workers,
        "dataloader_prefetch_factor": args.prefetch_factor,
        "dataloader_persistent_workers": args.persistent_workers,
        "dataloader_audit_path": str(dataloader_audit_path),
        "loss_components_path": str(loss_components_path),
        "action_l1_source": "ACTPolicy.forward loss_dict['l1_loss']",
        "save_freq": save_freq,
        "prepared_dataset_manifest": prepared_manifest,
        "prepared_dataset_root": str(prepared_root),
        "prepared_dataset_passthrough": bool(prepared_manifest.get("prepared_dataset_passthrough", False)),
        "train_output_dir": str(train_output_dir),
        "metrics_path": str(metrics_path),
        "actual_training_config": str(worker_config_path),
        "field_mapping": FIELD_MAPPING,
        "runtime_env": runtime_env,
        "datasets_disk_check_bypass_requested": args.allow_datasets_disk_check_bypass,
    }
    write_json(manifest_path, manifest)

    started = time.monotonic()
    return_code = stream_worker(worker_config_path, metrics_path, manifest_path, manifest)
    elapsed = time.monotonic() - started

    if manifest_path.exists():
        manifest = read_json(manifest_path)
    manifest.update(
        {
            "finished_at": utc_now(),
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "status": "completed" if return_code == 0 else "failed",
            "checkpoints": find_checkpoints(run_dir),
            "seconds_per_epoch_observed": (
                elapsed / args.epochs if args.epochs and step_budget_source == "epochs" else None
            ),
            "seconds_per_step_observed": elapsed / planned_steps if planned_steps else None,
        }
    )
    write_json(manifest_path, manifest)
    print(f"ok: wrote manifest to {manifest_path}", flush=True)
    print(f"ok: wrote metrics to {metrics_path}", flush=True)
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - server runbook should preserve concise failures.
        traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
