"""Create CALVIN A/ABC episode index lists from LeRobot metadata.

The split source of truth is original_frame_idx in episodes_stats.jsonl matched
against the fixed CALVIN task_ABC_D training scene_info.npy. The script never
uses episode order or shard order to infer environments.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-info", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-frame-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--min-env-count-ratio", type=float, default=0.5)
    parser.add_argument(
        "--allow-label-failures",
        action="store_true",
        help="Write outputs even if hard label validation fails. Do not use for training.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def label_for_frame(frame_idx: int, scene_ranges: list[dict[str, Any]]) -> str | None:
    for scene_range in scene_ranges:
        if scene_range["start"] <= frame_idx <= scene_range["end"]:
            return scene_range["label"]
    return None


def episode_frame_range(row: dict[str, Any]) -> dict[str, int]:
    episode_index = int(row["episode_index"])
    original = row.get("stats", {}).get("original_frame_idx")
    if not isinstance(original, dict):
        raise ValueError(f"episode {episode_index} is missing stats.original_frame_idx")
    return {
        "episode_index": episode_index,
        "frame_min": int(scalar(original["min"])),
        "frame_max": int(scalar(original["max"])),
        "count": int(scalar(original.get("count", [0]))),
    }


def classify_episode(
    dataset_name: str,
    frame_record: dict[str, int],
    scene_ranges: list[dict[str, Any]],
    episodes_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    start_label = label_for_frame(frame_record["frame_min"], scene_ranges)
    end_label = label_for_frame(frame_record["frame_max"], scene_ranges)
    if start_label is None or end_label is None:
        env = "orphan"
    elif start_label != end_label:
        env = "cross"
    else:
        env = start_label

    episode = episodes_by_index.get(frame_record["episode_index"], {})
    return {
        "dataset": dataset_name,
        "episode_index": frame_record["episode_index"],
        "env": env,
        "frame_min": frame_record["frame_min"],
        "frame_max": frame_record["frame_max"],
        "length": int(episode.get("length") or frame_record["count"] or 0),
        "tasks": episode.get("tasks", []),
    }


def run_summary(labelled: list[dict[str, Any]], *, sort_by_frame: bool) -> dict[str, Any]:
    rows = sorted(labelled, key=lambda row: row["frame_min"]) if sort_by_frame else labelled
    runs: list[dict[str, Any]] = []
    current_label: str | None = None
    current_len = 0

    for row in rows:
        label = row["env"]
        if label != current_label:
            if current_label is not None:
                runs.append({"env": current_label, "length": current_len})
            current_label = label
            current_len = 1
        else:
            current_len += 1

    if current_label is not None:
        runs.append({"env": current_label, "length": current_len})

    return {
        "run_count": len(runs),
        "transitions": max(len(runs) - 1, 0),
        "longest_run": max((run["length"] for run in runs), default=0),
        "first_runs": runs[:10],
    }


def validate(
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
    labels = [item["label"] for item in scene_ranges]
    abc_counts = [counts.get(label, 0) for label in labels]
    failures: list[str] = []
    warnings: list[str] = []

    if counts.get("cross", 0):
        failures.append(f"cross-scene episodes found: {counts['cross']}")
    if counts.get("orphan", 0):
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
        failures.append(f"one or more ABC labels have zero episodes: {dict(zip(labels, abc_counts, strict=True))}")
    if abc_counts and min(abc_counts) / max(abc_counts) < min_env_count_ratio:
        failures.append(
            "ABC episode counts are too imbalanced: "
            f"{dict(zip(labels, abc_counts, strict=True))}, min/max ratio={min(abc_counts) / max(abc_counts):.3f}"
        )

    order_runs = run_summary(labelled, sort_by_frame=False)
    frame_runs = run_summary(labelled, sort_by_frame=True)
    if order_runs["transitions"] > 2:
        warnings.append(
            "episode_index order is interleaved; split logic did not use episode order and this is not a failure"
        )
    if frame_runs["transitions"] > 2:
        warnings.append("frame-sorted labels are not contiguous; inspect scene_info mapping before training")

    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "counts": dict(counts),
        "scene_frame_min": scene_min,
        "scene_frame_max": scene_max,
        "dataset_frame_min": frame_min,
        "dataset_frame_max": frame_max,
        "episode_order_runs": order_runs,
        "frame_sorted_runs": frame_runs,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    dataset_name = dataset_root.name
    meta_root = dataset_root / "meta"
    output_root = Path(args.output_dir) / dataset_name

    episodes = read_jsonl(meta_root / "episodes.jsonl")
    stats_rows = read_jsonl(meta_root / "episodes_stats.jsonl")
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    scene_ranges = load_scene_ranges(Path(args.scene_info))

    if len(episodes) != len(stats_rows):
        raise RuntimeError(f"episodes/stat count mismatch: episodes={len(episodes)}, stats={len(stats_rows)}")

    labelled = [
        classify_episode(dataset_name, episode_frame_range(row), scene_ranges, episodes_by_index) for row in stats_rows
    ]
    validation = validate(
        labelled,
        scene_ranges,
        min_frame_coverage_ratio=args.min_frame_coverage_ratio,
        min_env_count_ratio=args.min_env_count_ratio,
    )

    print("label_counts:", json.dumps(validation["counts"], ensure_ascii=False, sort_keys=True))
    print("episode_order_transitions:", validation["episode_order_runs"]["transitions"])
    print("frame_sorted_transitions:", validation["frame_sorted_runs"]["transitions"])
    for warning in validation["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    if not validation["ok"] and not args.allow_label_failures:
        for failure in validation["failures"]:
            print(f"error: {failure}", file=sys.stderr)
        return 1

    by_env: dict[str, list[int]] = {}
    for row in labelled:
        by_env.setdefault(row["env"], []).append(int(row["episode_index"]))

    episodes_a = sorted(by_env.get("A", []))
    episodes_b = sorted(by_env.get("B", []))
    episodes_c = sorted(by_env.get("C", []))
    episodes_abc = sorted(episodes_a + episodes_b + episodes_c)
    episodes_d: list[int] = []

    write_jsonl(output_root / "env_label_map.jsonl", labelled)
    write_json(output_root / "episodes_A.json", episodes_a)
    write_json(output_root / "episodes_B.json", episodes_b)
    write_json(output_root / "episodes_C.json", episodes_c)
    write_json(output_root / "episodes_ABC.json", episodes_abc)
    write_json(output_root / "episodes_D.json", episodes_d)
    write_json(
        output_root / "summary.json",
        {
            "generated_at": utc_now(),
            "dataset_root": str(dataset_root),
            "dataset": dataset_name,
            "scene_info": str(Path(args.scene_info)),
            "scene_ranges": scene_ranges,
            "validation": validation,
            "outputs": {
                "env_label_map": str(output_root / "env_label_map.jsonl"),
                "episodes_A": str(output_root / "episodes_A.json"),
                "episodes_ABC": str(output_root / "episodes_ABC.json"),
                "episodes_D": str(output_root / "episodes_D.json"),
            },
            "notes": [
                "episodes_D is empty because task_ABC_D training scene_info only labels ABC.",
                "Do not infer environments from episode order or shard order.",
            ],
        },
    )

    print(f"ok: wrote split files to {output_root}")
    print(f"episodes_A: {len(episodes_a)}")
    print(f"episodes_ABC: {len(episodes_abc)}")
    print("episodes_D: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
