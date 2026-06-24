"""Audit a CALVIN LeRobot shard and optionally dump LeRobot ACT defaults."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENV_FIELD_HINTS = ("env", "environment", "scene", "scene_id", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--scene-info", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--inspect-lerobot-act-defaults", action="store_true")
    parser.add_argument("--output-act-dump", default=None)
    parser.add_argument("--min-frame-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--min-env-count-ratio", type=float, default=0.5)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return scalar(value[0])
    if hasattr(value, "item"):
        return value.item()
    return value


def load_scene_ranges(path: Path) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to read scene_info.npy") from exc

    raw = np.load(path, allow_pickle=True)
    mapping = raw.item() if getattr(raw, "shape", None) == () else raw
    if not isinstance(mapping, dict):
        raise ValueError(f"expected dict-like scene_info.npy, got {type(mapping).__name__}")

    ranges: list[dict[str, Any]] = []
    for name, bounds in mapping.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"invalid scene bounds for {name!r}: {bounds!r}")
        label = str(name).rsplit("_", 1)[-1].upper()
        ranges.append({"name": str(name), "label": label, "start": int(bounds[0]), "end": int(bounds[1])})

    ranges.sort(key=lambda item: item["start"])
    return ranges


def env_candidates(info: dict[str, Any], episodes: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    features = info.get("features", {})
    if isinstance(features, dict):
        keys.update(str(key) for key in features)
    if episodes:
        keys.update(str(key) for key in episodes[0])
    return sorted(key for key in keys if any(hint in key.lower() for hint in ENV_FIELD_HINTS))


def get_episode_frame_records(stats_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in stats_rows:
        episode_index = int(row["episode_index"])
        stats = row.get("stats", {})
        original = stats.get("original_frame_idx") if isinstance(stats, dict) else None
        if not isinstance(original, dict):
            raise ValueError(f"episode {episode_index} is missing stats.original_frame_idx")
        frame_min = int(scalar(original["min"]))
        frame_max = int(scalar(original["max"]))
        count = int(scalar(original.get("count", [0])))
        records.append({"episode_index": episode_index, "frame_min": frame_min, "frame_max": frame_max, "count": count})
    return records


def label_for_frame(frame_idx: int, scene_ranges: list[dict[str, Any]]) -> str | None:
    for scene_range in scene_ranges:
        if scene_range["start"] <= frame_idx <= scene_range["end"]:
            return scene_range["label"]
    return None


def classify_records(
    frame_records: list[dict[str, Any]],
    scene_ranges: list[dict[str, Any]],
    episodes_by_index: dict[int, dict[str, Any]],
    dataset_name: str,
) -> list[dict[str, Any]]:
    labelled: list[dict[str, Any]] = []
    for record in frame_records:
        start_label = label_for_frame(record["frame_min"], scene_ranges)
        end_label = label_for_frame(record["frame_max"], scene_ranges)
        if start_label is None or end_label is None:
            env = "orphan"
        elif start_label != end_label:
            env = "cross"
        else:
            env = start_label

        episode = episodes_by_index.get(record["episode_index"], {})
        labelled.append(
            {
                "dataset": dataset_name,
                "episode_index": record["episode_index"],
                "env": env,
                "frame_min": record["frame_min"],
                "frame_max": record["frame_max"],
                "length": int(episode.get("length") or record["count"] or 0),
                "tasks": episode.get("tasks", []),
            }
        )
    return labelled


def run_summary(labelled: list[dict[str, Any]], *, sort_by_frame: bool) -> dict[str, Any]:
    rows = sorted(labelled, key=lambda row: row["frame_min"]) if sort_by_frame else labelled
    runs: list[dict[str, Any]] = []
    current_label: str | None = None
    current_len = 0
    current_start: int | None = None
    current_end: int | None = None

    for row in rows:
        label = row["env"]
        if label != current_label:
            if current_label is not None:
                runs.append(
                    {
                        "env": current_label,
                        "length": current_len,
                        "start_episode": current_start,
                        "end_episode": current_end,
                    }
                )
            current_label = label
            current_len = 1
            current_start = int(row["episode_index"])
            current_end = int(row["episode_index"])
        else:
            current_len += 1
            current_end = int(row["episode_index"])

    if current_label is not None:
        runs.append(
            {"env": current_label, "length": current_len, "start_episode": current_start, "end_episode": current_end}
        )

    return {
        "run_count": len(runs),
        "transitions": max(len(runs) - 1, 0),
        "longest_run": max((run["length"] for run in runs), default=0),
        "first_runs": runs[:10],
    }


def validate_labels(
    labelled: list[dict[str, Any]],
    scene_ranges: list[dict[str, Any]],
    *,
    min_frame_coverage_ratio: float,
    min_env_count_ratio: float,
) -> dict[str, Any]:
    counts = Counter(row["env"] for row in labelled)
    scene_min = min(item["start"] for item in scene_ranges)
    scene_max = max(item["end"] for item in scene_ranges)
    frame_min = min(row["frame_min"] for row in labelled)
    frame_max = max(row["frame_max"] for row in labelled)
    abc_labels = [item["label"] for item in scene_ranges]
    abc_counts = [counts.get(label, 0) for label in abc_labels]
    failures: list[str] = []
    warnings: list[str] = []

    if counts.get("cross", 0) > 0:
        failures.append(f"cross-scene episodes found: {counts['cross']}")
    if counts.get("orphan", 0) > 0:
        failures.append(f"orphan episodes outside scene_info ranges found: {counts['orphan']}")
    if frame_min < scene_min:
        failures.append(f"global frame_min {frame_min} is below scene_info start {scene_min}")
    if frame_max > scene_max:
        failures.append(f"global frame_max {frame_max} exceeds scene_info end {scene_max}")
    if frame_max < int(scene_max * min_frame_coverage_ratio):
        failures.append(
            f"global frame_max {frame_max} covers less than {min_frame_coverage_ratio:.2f} of scene_info end {scene_max}"
        )
    if any(count <= 0 for count in abc_counts):
        failures.append(f"one or more ABC labels have zero episodes: {dict(zip(abc_labels, abc_counts, strict=True))}")
    if abc_counts and min(abc_counts) / max(abc_counts) < min_env_count_ratio:
        failures.append(
            "ABC episode counts are too imbalanced: "
            f"{dict(zip(abc_labels, abc_counts, strict=True))}, min/max ratio={min(abc_counts) / max(abc_counts):.3f}"
        )

    order_runs = run_summary(labelled, sort_by_frame=False)
    frame_runs = run_summary(labelled, sort_by_frame=True)
    if order_runs["transitions"] > 2:
        warnings.append(
            "episode_index order is interleaved across environments; this is expected for this converted shard, "
            "and split logic must not depend on episode order"
        )
    if frame_runs["transitions"] > 2:
        warnings.append("frame-sorted labels are not contiguous; inspect original_frame_idx mapping before training")

    return {
        "counts": dict(counts),
        "scene_frame_min": scene_min,
        "scene_frame_max": scene_max,
        "dataset_frame_min": frame_min,
        "dataset_frame_max": frame_max,
        "order_runs": order_runs,
        "frame_sorted_runs": frame_runs,
        "failures": failures,
        "warnings": warnings,
        "ok": not failures,
    }


def parquet_probe(dataset_root: Path) -> dict[str, Any]:
    parquet = next(iter(sorted(dataset_root.glob("data/**/*.parquet"))), None)
    if parquet is None:
        return {"available": False, "reason": "no parquet files found"}
    try:
        import pandas as pd
    except ImportError as exc:
        return {"available": False, "path": str(parquet), "reason": f"pandas import failed: {exc}"}

    try:
        frame = pd.read_parquet(parquet)
    except Exception as exc:  # noqa: BLE001 - audit should preserve the exact import/backend failure.
        return {"available": False, "path": str(parquet), "reason": f"read_parquet failed: {exc}"}

    columns: list[dict[str, Any]] = []
    for column in frame.columns:
        value = frame[column].iloc[0] if len(frame) else None
        columns.append(
            {
                "name": str(column),
                "dtype": str(frame[column].dtype),
                "sample_type": type(value).__name__,
                "sample_shape": list(getattr(value, "shape", []) or []),
            }
        )
    return {"available": True, "path": str(parquet), "rows": len(frame), "columns": columns}


def feature_summary(info: dict[str, Any], modality: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features", {})
    camera_features = {
        key: value
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    }
    return {
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
        "fps": info.get("fps"),
        "state": features.get("observation.state"),
        "action": features.get("action"),
        "cameras": camera_features,
        "action_modality": modality.get("action"),
        "video_modality": modality.get("video"),
    }


def dataset_format_warnings(info: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    codebase_version = str(info.get("codebase_version") or "")
    if codebase_version == "v2.1":
        warnings.append(
            "Dataset metadata is LeRobot codebase v2.1. Day 2 metadata audit and episode-list splitting are valid, "
            "but direct training with lerobot==0.4.0 may require confirming the v2.1 loader path or converting to v3.0."
        )
    return warnings


def render_markdown(audit: dict[str, Any]) -> str:
    feature = audit["features"]
    label = audit.get("labels")
    parquet = audit["parquet_probe"]
    lines = [
        "# Day 2 CALVIN Data Audit",
        "",
        f"- Generated at: `{audit['generated_at']}`",
        f"- Dataset root: `{audit['dataset_root']}`",
        f"- Dataset name: `{audit['dataset_name']}`",
        f"- LeRobot codebase version: `{feature.get('codebase_version')}`",
        f"- Robot type: `{feature.get('robot_type')}`",
        f"- Total episodes: `{feature.get('total_episodes')}`",
        f"- Total frames: `{feature.get('total_frames')}`",
        f"- FPS: `{feature.get('fps')}`",
        "",
        "## Format Compatibility",
        "",
    ]
    if audit["format_warnings"]:
        lines.extend(f"- Warning: {warning}" for warning in audit["format_warnings"])
    else:
        lines.append("- No dataset format warning was detected.")

    lines.extend(
        [
            "",
            "## Schema",
            "",
            f"- `observation.state`: `{feature.get('state')}`",
            f"- `action`: `{feature.get('action')}`",
            f"- Action modality: `{json.dumps(feature.get('action_modality'), ensure_ascii=False)}`",
            f"- Video modality: `{json.dumps(feature.get('video_modality'), ensure_ascii=False)}`",
            "",
            "## Environment Label Source",
            "",
        ]
    )

    if audit["explicit_env_field_candidates"]:
        lines.append(f"- Explicit environment-like fields found: `{audit['explicit_env_field_candidates']}`")
    else:
        lines.append("- No explicit `env`, `scene_id`, or `environment` metadata field was found.")
    lines.append(
        "- Environment labels are derived from `episodes_stats.original_frame_idx` intersected with the fixed "
        "`scene_info.npy` ranges."
    )

    if label:
        lines.extend(
            [
                f"- Scene info: `{audit['scene_info']}`",
                f"- Scene ranges: `{json.dumps(label['scene_ranges'], ensure_ascii=False)}`",
                f"- Label counts: `{json.dumps(label['validation']['counts'], ensure_ascii=False)}`",
                f"- Dataset frame range: `{label['validation']['dataset_frame_min']}..{label['validation']['dataset_frame_max']}`",
                f"- Scene frame range: `{label['validation']['scene_frame_min']}..{label['validation']['scene_frame_max']}`",
                f"- Validation ok: `{label['validation']['ok']}`",
                f"- Validation failures: `{label['validation']['failures']}`",
                f"- Validation warnings: `{label['validation']['warnings']}`",
                f"- Episode-order transitions: `{label['validation']['order_runs']['transitions']}`",
                f"- Frame-sorted transitions: `{label['validation']['frame_sorted_runs']['transitions']}`",
            ]
        )

    lines.extend(["", "## Parquet Probe", ""])
    if parquet["available"]:
        lines.append(f"- Sample parquet: `{parquet['path']}`")
        lines.append(f"- Rows: `{parquet['rows']}`")
        for column in parquet["columns"]:
            lines.append(
                f"- `{column['name']}`: dtype=`{column['dtype']}`, "
                f"type=`{column['sample_type']}`, shape=`{column['sample_shape']}`"
            )
    else:
        lines.append(f"- Parquet probe unavailable: `{parquet.get('reason')}`")

    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Do not infer A/B/C/D from `episode_index` order or shard order.",
            "- Use the generated episode index lists for training filters; do not copy large video/parquet data.",
            "- D offline episodes are not required for the core zero-shot rollout path; CALVIN simulator evaluation will handle environment D later.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    meta_root = dataset_root / "meta"
    info = read_json(meta_root / "info.json")
    modality = read_json(meta_root / "modality.json")
    episodes = read_jsonl(meta_root / "episodes.jsonl")
    stats_rows = read_jsonl(meta_root / "episodes_stats.jsonl")
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}

    audit: dict[str, Any] = {
        "generated_at": utc_now(),
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_root.name,
        "features": feature_summary(info, modality),
        "format_warnings": dataset_format_warnings(info),
        "explicit_env_field_candidates": env_candidates(info, episodes),
        "episodes_jsonl_count": len(episodes),
        "episodes_stats_jsonl_count": len(stats_rows),
        "parquet_probe": parquet_probe(dataset_root),
    }

    if args.scene_info:
        scene_info = Path(args.scene_info)
        scene_ranges = load_scene_ranges(scene_info)
        frame_records = get_episode_frame_records(stats_rows)
        labelled = classify_records(frame_records, scene_ranges, episodes_by_index, dataset_root.name)
        validation = validate_labels(
            labelled,
            scene_ranges,
            min_frame_coverage_ratio=args.min_frame_coverage_ratio,
            min_env_count_ratio=args.min_env_count_ratio,
        )
        audit["scene_info"] = str(scene_info)
        audit["labels"] = {
            "scene_ranges": scene_ranges,
            "validation": validation,
            "labelled_episode_count": len(labelled),
        }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as file:
            json.dump(audit, file, ensure_ascii=False, indent=2)
            file.write("\n")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(audit), encoding="utf-8")

    return audit


def to_plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        try:
            return to_plain(value.to_dict())
        except Exception:  # noqa: BLE001 - fallback to repr for audit dumps.
            pass
    return repr(value)


def import_first(candidates: list[str]) -> tuple[str, Any]:
    failures: list[str] = []
    for name in candidates:
        try:
            return name, importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - audit should show all import failures.
            failures.append(f"{name}: {exc!r}")
    raise RuntimeError("all imports failed:\n" + "\n".join(failures))


def inspect_lerobot_defaults(output_path: Path) -> None:
    config_module_name, config_module = import_first(
        [
            "lerobot.policies.act.configuration_act",
            "lerobot.common.policies.act.configuration_act",
        ]
    )
    act_config_cls = getattr(config_module, "ACTConfig")
    act_config = act_config_cls()

    dataset_probe: dict[str, Any] = {}
    try:
        dataset_module_name, dataset_module = import_first(
            [
                "lerobot.datasets.lerobot_dataset",
                "lerobot.common.datasets.lerobot_dataset",
            ]
        )
        dataset_cls = getattr(dataset_module, "LeRobotDataset")
        dataset_probe = {
            "module": dataset_module_name,
            "signature": str(inspect.signature(dataset_cls)),
            "has_episodes_parameter": "episodes" in inspect.signature(dataset_cls).parameters,
        }
    except Exception as exc:  # noqa: BLE001 - this is diagnostic metadata, not the core dump.
        dataset_probe = {"error": repr(exc)}

    train_probe: dict[str, Any] = {}
    try:
        train_module_name, train_module = import_first(["lerobot.configs.train", "lerobot.common.configs.train"])
        train_probe["module"] = train_module_name
        for attr in ("TrainPipelineConfig", "DatasetConfig"):
            if hasattr(train_module, attr):
                obj = getattr(train_module, attr)
                train_probe[attr] = str(inspect.signature(obj))
    except Exception as exc:  # noqa: BLE001 - this is diagnostic metadata.
        train_probe = {"error": repr(exc)}

    version = "unknown"
    try:
        lerobot_module = importlib.import_module("lerobot")
        version = getattr(lerobot_module, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - keep dump useful if version attr import fails.
        pass

    payload = {
        "generated_at": utc_now(),
        "lerobot_version": version,
        "act_config_module": config_module_name,
        "act_config_signature": str(inspect.signature(act_config_cls)),
        "act_config": to_plain(act_config),
        "dataset_episode_filter_probe": dataset_probe,
        "train_config_probe": train_probe,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("# LeRobot ACT defaults dump\n\n")
        file.write(json.dumps(payload, ensure_ascii=False, indent=2))
        file.write("\n")
    print(f"ok: wrote ACT defaults dump to {output_path}")


def main() -> int:
    args = parse_args()

    if args.inspect_lerobot_act_defaults:
        if not args.output_act_dump:
            raise ValueError("--output-act-dump is required with --inspect-lerobot-act-defaults")
        inspect_lerobot_defaults(Path(args.output_act_dump))
        if not args.dataset_root:
            return 0

    if not args.dataset_root:
        raise ValueError("--dataset-root is required unless only --inspect-lerobot-act-defaults is used")

    audit = audit_dataset(args)
    for warning in audit["format_warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    labels = audit.get("labels")
    if labels:
        validation = labels["validation"]
        print("label_counts:", json.dumps(validation["counts"], ensure_ascii=False, sort_keys=True))
        for warning in validation["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
        if not validation["ok"]:
            for failure in validation["failures"]:
                print(f"error: {failure}", file=sys.stderr)
            return 1
    print("ok: dataset audit completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
