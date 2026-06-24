"""Summarize Week 3 Day 5 Topic 2 evaluation artifacts.

The script combines full splitD open-loop Action L1 JSON files, optional
direct-cameras smoke logs, and training action-L1 component logs into compact
tables and report figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAL_SUCCESS_NOTE = (
    "not run as a formal closed-loop benchmark; strict EGL was limited by the "
    "system graphics stack; see direct-cameras smoke evidence"
)


@dataclass(frozen=True)
class ModelPaths:
    name: str
    training_data: str
    steps: int
    action_l1_path: Path
    smoke_log_path: Path
    loss_components_path: Path
    smoke_json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Day 5 A-only vs ABC result tables and figures.")
    parser.add_argument("--results-dir", default="topic2_act/eval/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--a-only-result", default="topic2_act/eval/results/a_only_splitD_action_l1.json")
    parser.add_argument("--abc-result", default="topic2_act/eval/results/abc_splitD_action_l1.json")
    parser.add_argument("--a-only-smoke-log", default="logs/Week3_Day4/day4_a_only_closed_loop_direct_cameras.log")
    parser.add_argument("--abc-smoke-log", default="logs/Week3_Day5/day5_abc_closed_loop_direct_cameras.log")
    parser.add_argument(
        "--a-only-loss-components",
        default="topic2_act/outputs/act_calvin/a_only_full_150k/loss_components.jsonl",
    )
    parser.add_argument(
        "--abc-loss-components",
        default="topic2_act/outputs/act_calvin/abc_full_150k/loss_components.jsonl",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def require_keys(payload: dict[str, Any], path: Path, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")


def validate_action_l1_pair(a_only: dict[str, Any], abc: dict[str, Any]) -> None:
    required = (
        "status",
        "action_l1_valid_mean",
        "raw_action_l1_valid_mean",
        "episode_count",
        "frame_count",
        "missing_parquet_count",
        "action_dim",
        "chunk_size",
        "per_chunk_l1",
        "raw_per_chunk_l1",
        "metric_space",
        "raw_metric_space",
        "checkpoint",
        "dataset_root",
    )
    require_keys(a_only, Path("a_only_result"), required)
    require_keys(abc, Path("abc_result"), required)

    for label, payload in (("A-only", a_only), ("ABC", abc)):
        if payload["status"] != "completed":
            raise ValueError(f"{label} result status must be completed, got {payload['status']!r}")
        if payload["missing_parquet_count"] != 0:
            raise ValueError(f"{label} result has missing parquet files: {payload['missing_parquet_count']}")

    comparable_keys = ("episode_count", "frame_count", "action_dim", "chunk_size", "metric_space", "raw_metric_space")
    mismatches = {
        key: (a_only.get(key), abc.get(key))
        for key in comparable_keys
        if a_only.get(key) != abc.get(key)
    }
    if mismatches:
        raise ValueError(f"A-only and ABC Action L1 results are not comparable: {mismatches}")


def extract_labeled_json(text: str, label: str) -> dict[str, Any]:
    label_index = text.rfind(label)
    if label_index < 0:
        raise ValueError(f"could not find label {label!r} in smoke log")

    brace_start = text.find("{", label_index)
    if brace_start < 0:
        raise ValueError(f"could not find JSON object after label {label!r}")

    depth = 0
    in_string = False
    escape = False
    for index in range(brace_start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[brace_start : index + 1])
    raise ValueError(f"unterminated JSON object after label {label!r}")


def smoke_summary_from_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    payload = extract_labeled_json(text, "single_rollout_smoke_result:")
    required = ("steps", "wrapper_step_count", "last_action_shape", "moved", "max_tcp_delta", "gripper_values")
    require_keys(payload, path, required)
    return payload


def finite_pairs(values: list[Any]) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            pairs.append((index, number))
    return pairs


def smoke_evidence(model: str, smoke: dict[str, Any], log_path: Path, json_path: Path) -> str:
    return (
        f"{model} direct-cameras smoke: steps={smoke['steps']}, "
        f"moved={smoke['moved']}, max_tcp_delta={float(smoke['max_tcp_delta']):.6f}, "
        f"last_action_shape={smoke['last_action_shape']}; json={json_path.as_posix()}, log={log_path.as_posix()}"
    )


def build_row(paths: ModelPaths, result: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": paths.name,
        "training_data": paths.training_data,
        "steps": paths.steps,
        "d_normalized_action_l1": float(result["action_l1_valid_mean"]),
        "d_raw_action_l1": float(result["raw_action_l1_valid_mean"]),
        "formal_closed_loop_success_rate": FORMAL_SUCCESS_NOTE,
        "direct_cameras_smoke_evidence": smoke_evidence(paths.name, smoke, paths.smoke_log_path, paths.smoke_json_path),
        "source_log_result_path": paths.action_l1_path.as_posix(),
        "checkpoint": result["checkpoint"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "training_data",
        "steps",
        "d_normalized_action_l1",
        "d_raw_action_l1",
        "formal_closed_loop_success_rate",
        "direct_cameras_smoke_evidence",
        "source_log_result_path",
        "checkpoint",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Week3 Day5 Topic2 A-only vs ABC",
        "",
        "| Model | Training data | Steps | D normalized Action L1 | D raw Action L1 | Formal closed-loop success rate | Direct-cameras smoke | Source |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {training_data} | {steps} | {norm:.6f} | {raw:.6f} | {success} | {smoke} | {source} |".format(
                model=row["model"],
                training_data=row["training_data"],
                steps=row["steps"],
                norm=row["d_normalized_action_l1"],
                raw=row["d_raw_action_l1"],
                success=row["formal_closed_loop_success_rate"],
                smoke=row["direct_cameras_smoke_evidence"],
                source=row["source_log_result_path"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_action_l1_series(path: Path, max_points: int = 2000) -> tuple[list[int], list[float]]:
    rows = read_jsonl(path)
    points: list[tuple[int, float]] = []
    for row in rows:
        if "step" not in row or "action_l1" not in row:
            continue
        points.append((int(row["step"]), float(row["action_l1"])))
    if len(points) <= max_points:
        return [step for step, _value in points], [value for _step, value in points]

    stride = max(1, len(points) // max_points)
    sampled = points[::stride]
    if sampled[-1][0] != points[-1][0]:
        sampled.append(points[-1])
    return [step for step, _value in sampled], [value for _step, value in sampled]


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curve(path: Path, a_only_loss: Path, abc_loss: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, source in (("A-only", a_only_loss), ("ABC", abc_loss)):
        steps, values = load_action_l1_series(source)
        ax.plot(steps, values, linewidth=1.3, label=label)
    ax.set_title("Topic 2 training action L1")
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("Training action L1")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_splitd_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    plt = setup_matplotlib()
    labels = [row["model"] for row in rows]
    normalized = [row["d_normalized_action_l1"] for row in rows]
    raw = [row["d_raw_action_l1"] for row in rows]
    x_positions = list(range(len(rows)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar([x - width / 2 for x in x_positions], normalized, width, label="Normalized")
    ax.bar([x + width / 2 for x in x_positions], raw, width, label="Raw")
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Mean Action L1 on splitD")
    ax.set_title("Environment D open-loop Action L1")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_chunk(path: Path, a_only: dict[str, Any], abc: dict[str, Any]) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, payload in (("A-only", a_only), ("ABC", abc)):
        pairs = finite_pairs(payload["per_chunk_l1"])
        ax.plot([index for index, _value in pairs], [value for _index, value in pairs], linewidth=1.4, label=label)
    ax.set_xlabel("Chunk offset")
    ax.set_ylabel("Normalized Action L1")
    ax.set_title("splitD Action L1 by ACT chunk offset")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    a_only_paths = ModelPaths(
        name="A-only",
        training_data="splitA",
        steps=150000,
        action_l1_path=Path(args.a_only_result),
        smoke_log_path=Path(args.a_only_smoke_log),
        loss_components_path=Path(args.a_only_loss_components),
        smoke_json_path=results_dir / "a_only_direct_cameras_smoke.json",
    )
    abc_paths = ModelPaths(
        name="ABC",
        training_data="splitA+splitB+splitC",
        steps=150000,
        action_l1_path=Path(args.abc_result),
        smoke_log_path=Path(args.abc_smoke_log),
        loss_components_path=Path(args.abc_loss_components),
        smoke_json_path=results_dir / "abc_direct_cameras_smoke.json",
    )

    a_only_result = read_json(a_only_paths.action_l1_path)
    abc_result = read_json(abc_paths.action_l1_path)
    validate_action_l1_pair(a_only_result, abc_result)

    a_only_smoke = smoke_summary_from_log(a_only_paths.smoke_log_path)
    abc_smoke = smoke_summary_from_log(abc_paths.smoke_log_path)
    write_json(a_only_paths.smoke_json_path, a_only_smoke)
    write_json(abc_paths.smoke_json_path, abc_smoke)

    rows = [
        build_row(a_only_paths, a_only_result, a_only_smoke),
        build_row(abc_paths, abc_result, abc_smoke),
    ]
    comparison = {
        "status": "completed",
        "metric_primary": "open-loop Action L1 on official splitD",
        "metric_note": "Lower is better. Normalized action space is the main comparison; raw action units are included for interpretability.",
        "formal_success_rate_note": FORMAL_SUCCESS_NOTE,
        "validation": {
            "episode_count": a_only_result["episode_count"],
            "frame_count": a_only_result["frame_count"],
            "metric_space": a_only_result["metric_space"],
            "raw_metric_space": a_only_result["raw_metric_space"],
            "action_dim": a_only_result["action_dim"],
            "chunk_size": a_only_result["chunk_size"],
        },
        "rows": rows,
    }
    write_json(results_dir / "day5_topic2_a_only_vs_abc.json", comparison)
    write_csv(results_dir / "day5_topic2_a_only_vs_abc.csv", rows)
    write_markdown(results_dir / "day5_topic2_a_only_vs_abc.md", rows)

    plot_training_curve(figures_dir / "topic2_train_action_l1_curve.png", a_only_paths.loss_components_path, abc_paths.loss_components_path)
    plot_splitd_bar(figures_dir / "topic2_splitD_action_l1_bar.png", rows)
    plot_per_chunk(figures_dir / "topic2_splitD_per_chunk_l1.png", a_only_result, abc_result)
    return comparison


def main() -> int:
    args = parse_args()
    comparison = build_outputs(args)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
