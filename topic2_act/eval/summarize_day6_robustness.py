"""Build Day 6 Topic 2 robustness tables and figures."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from topic2_act.eval.summarize_day5_results import (  # noqa: E402
    FORMAL_SUCCESS_NOTE,
    ModelPaths,
    build_row,
    plot_per_chunk,
    plot_splitd_bar,
    plot_training_curve,
    read_json,
    smoke_summary_from_log,
    validate_action_l1_pair,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Day 6 action chunking robustness outputs.")
    parser.add_argument("--results-dir", default="topic2_act/eval/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument(
        "--a-only-result",
        default="topic2_act/eval/results/a_only_splitD_action_l1_day6_distribution.json",
    )
    parser.add_argument(
        "--abc-result",
        default="topic2_act/eval/results/abc_splitD_action_l1_day6_distribution.json",
    )
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
    parser.add_argument(
        "--a-only-manifest",
        default="topic2_act/outputs/act_calvin/a_only_full_150k/run_manifest.json",
    )
    parser.add_argument("--abc-manifest", default="topic2_act/outputs/act_calvin/abc_full_150k/run_manifest.json")
    parser.add_argument("--week3-config", default="topic2_act/configs/act_calvin_week3_150k.yaml")
    parser.add_argument("--act-defaults", default="topic2_act/configs/act_default_v0.4.0.dump.txt")
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    clean = value.strip()
    if clean in {"true", "True"}:
        return True
    if clean in {"false", "False"}:
        return False
    if clean in {"null", "None", "~"}:
        return None
    try:
        return ast.literal_eval(clean)
    except (SyntaxError, ValueError):
        pass
    try:
        if any(token in clean for token in (".", "e", "E")):
            return float(clean)
        return int(clean)
    except ValueError:
        return clean.strip("'\"")


def read_yaml_scalars(path: Path) -> dict[str, Any]:
    scalars: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parts = [item for _indent, item in stack] + [key]
        value = value.strip()
        if value:
            scalars[".".join(parts)] = parse_scalar(value)
        else:
            stack.append((indent, key))
    return scalars


def read_act_defaults(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"could not find JSON payload in {path}")
    payload = json.loads(text[start:])
    return payload["act_config"]


def finite_values(values: list[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            output.append(number)
    return output


def mean_window(values: list[Any], start: int, end: int) -> float | None:
    finite = finite_values(values[start:end])
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def last_finite_index(values: list[Any]) -> int | None:
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value is not None and math.isfinite(float(value)):
            return index
    return None


def relative_reduction(before: float, after: float) -> float:
    return float((before - after) / before) if before else float("nan")


def distribution_summary(result: dict[str, Any], key: str) -> dict[str, Any]:
    required = ("count", "mean", "std", "p50", "p75", "p95", "histogram_bin_edges", "histogram_counts")
    payload = result.get(key)
    if not isinstance(payload, dict):
        raise ValueError(f"missing distribution summary: {key}")
    missing = [item for item in required if item not in payload]
    if missing:
        raise ValueError(f"{key} is missing required fields: {missing}")
    return payload


def build_metric_rows(a_only: dict[str, Any], abc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, result in (("A-only", a_only), ("ABC", abc)):
        dist = distribution_summary(result, "frame_l1_distribution")
        raw_dist = distribution_summary(result, "raw_frame_l1_distribution")
        tail_index = last_finite_index(result["per_chunk_l1"])
        rows.append(
            {
                "model": model,
                "normalized_action_l1": float(result["action_l1_valid_mean"]),
                "raw_action_l1": float(result["raw_action_l1_valid_mean"]),
                "frame_l1_mean": float(dist["mean"]),
                "frame_l1_median": float(dist["p50"]),
                "frame_l1_p75": float(dist["p75"]),
                "frame_l1_p95": float(dist["p95"]),
                "raw_frame_l1_median": float(raw_dist["p50"]),
                "chunk_offset_0_l1": float(result["per_chunk_l1"][0]),
                "chunk_offset_tail": tail_index,
                "chunk_offset_tail_l1": float(result["per_chunk_l1"][tail_index]) if tail_index is not None else None,
                "chunk_early_0_9_l1": mean_window(result["per_chunk_l1"], 0, 10),
                "chunk_mid_25_34_l1": mean_window(result["per_chunk_l1"], 25, 35),
                "chunk_late_55_64_l1": mean_window(result["per_chunk_l1"], 55, 65),
                "distribution_count": int(dist["count"]),
            }
        )
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hyperparam_rows(config: dict[str, Any], defaults: dict[str, Any], a_manifest: dict[str, Any], abc_manifest: dict[str, Any]) -> list[dict[str, str]]:
    def row(group: str, parameter: str, value: Any, source: str) -> dict[str, str]:
        return {"group": group, "parameter": parameter, "value": str(value), "source": source}

    rows = [
        row("model", "policy", config.get("policy.type", "act"), "act_calvin_week3_150k.yaml"),
        row("model", "vision_backbone", defaults["vision_backbone"], "act_default_v0.4.0.dump.txt"),
        row("model", "pretrained_backbone_weights", defaults["pretrained_backbone_weights"], "act_default_v0.4.0.dump.txt"),
        row("model", "dim_model", defaults["dim_model"], "act_default_v0.4.0.dump.txt"),
        row("model", "n_heads", defaults["n_heads"], "act_default_v0.4.0.dump.txt"),
        row("model", "n_encoder_layers", defaults["n_encoder_layers"], "act_default_v0.4.0.dump.txt"),
        row("model", "n_decoder_layers", defaults["n_decoder_layers"], "act_default_v0.4.0.dump.txt"),
        row("model", "dim_feedforward", defaults["dim_feedforward"], "act_default_v0.4.0.dump.txt"),
        row("model", "use_vae", defaults["use_vae"], "act_default_v0.4.0.dump.txt"),
        row("model", "latent_dim", defaults["latent_dim"], "act_default_v0.4.0.dump.txt"),
        row("model", "dropout", defaults["dropout"], "act_default_v0.4.0.dump.txt"),
        row("model", "kl_weight", defaults["kl_weight"], "act_default_v0.4.0.dump.txt"),
        row("data", "A-only episodes/frames", f"{a_manifest['selected_episode_count']} / {a_manifest['selected_frame_count']}", "a_only run_manifest.json"),
        row("data", "ABC episodes/frames", f"{abc_manifest['selected_episode_count']} / {abc_manifest['selected_frame_count']}", "abc run_manifest.json"),
        row("data", "state/action shape", f"{config.get('data.schema.state_shape')} / {config.get('data.schema.action_shape')}", "act_calvin_week3_150k.yaml"),
        row("data", "static/wrist image shape", f"{config.get('data.schema.image_shape')} / {config.get('data.schema.wrist_image_shape')}", "act_calvin_week3_150k.yaml"),
        row("training", "optimizer", config.get("optimization.optimizer"), "act_calvin_week3_150k.yaml"),
        row("training", "learning_rate", config.get("optimization.learning_rate"), "act_calvin_week3_150k.yaml"),
        row("training", "weight_decay", config.get("optimization.weight_decay"), "act_calvin_week3_150k.yaml"),
        row("training", "batch_size", a_manifest["batch_size"], "run_manifest.json"),
        row("training", "gradient_steps", a_manifest["planned_steps"], "run_manifest.json"),
        row("training", "seed", config.get("optimization.seed"), "act_calvin_week3_150k.yaml"),
        row("training", "chunk_size", config.get("policy.chunk_size"), "act_calvin_week3_150k.yaml"),
        row("training", "n_action_steps", config.get("policy.n_action_steps"), "act_calvin_week3_150k.yaml"),
        row("training", "num_workers", a_manifest["num_workers"], "run_manifest.json"),
        row("training", "prefetch_factor", a_manifest["dataloader_prefetch_factor"], "run_manifest.json"),
        row("training", "persistent_workers", a_manifest["dataloader_persistent_workers"], "run_manifest.json"),
        row("training", "save_freq", a_manifest["save_freq"], "run_manifest.json"),
        row("logging", "A-only WandB", a_manifest.get("wandb_url"), "a_only run_manifest.json"),
        row("logging", "ABC WandB", abc_manifest.get("wandb_url"), "abc run_manifest.json"),
    ]
    return rows


def plot_distribution(path: Path, a_only: dict[str, Any], abc: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, payload in (("A-only", a_only), ("ABC", abc)):
        dist = distribution_summary(payload, "frame_l1_distribution")
        counts = dist["histogram_counts"]
        edges = dist["histogram_bin_edges"]
        total = sum(counts)
        density = [count / total if total else 0 for count in counts]
        ax.stairs(density, edges, linewidth=1.6, label=label)
    ax.set_title("splitD per-frame action-chunk L1 distribution")
    ax.set_xlabel("Per-frame normalized Action L1")
    ax.set_ylabel("Fraction of frames")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    a_only_result = read_json(Path(args.a_only_result))
    abc_result = read_json(Path(args.abc_result))
    validate_action_l1_pair(a_only_result, abc_result)
    distribution_summary(a_only_result, "frame_l1_distribution")
    distribution_summary(abc_result, "frame_l1_distribution")

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
    a_smoke = smoke_summary_from_log(a_only_paths.smoke_log_path)
    abc_smoke = smoke_summary_from_log(abc_paths.smoke_log_path)
    comparison_rows = [
        build_row(a_only_paths, a_only_result, a_smoke),
        build_row(abc_paths, abc_result, abc_smoke),
    ]
    metric_rows = build_metric_rows(a_only_result, abc_result)
    config = read_yaml_scalars(Path(args.week3_config))
    defaults = read_act_defaults(Path(args.act_defaults))
    a_manifest = read_json(Path(args.a_only_manifest))
    abc_manifest = read_json(Path(args.abc_manifest))
    hp_rows = hyperparam_rows(config, defaults, a_manifest, abc_manifest)

    normalized_reduction = relative_reduction(
        float(a_only_result["action_l1_valid_mean"]),
        float(abc_result["action_l1_valid_mean"]),
    )
    raw_reduction = relative_reduction(
        float(a_only_result["raw_action_l1_valid_mean"]),
        float(abc_result["raw_action_l1_valid_mean"]),
    )
    p95_reduction = relative_reduction(
        float(distribution_summary(a_only_result, "frame_l1_distribution")["p95"]),
        float(distribution_summary(abc_result, "frame_l1_distribution")["p95"]),
    )
    payload = {
        "status": "completed",
        "metric_primary": "open-loop Action L1 on official splitD",
        "formal_success_rate_note": FORMAL_SUCCESS_NOTE,
        "validation": {
            "episode_count": a_only_result["episode_count"],
            "frame_count": a_only_result["frame_count"],
            "metric_space": a_only_result["metric_space"],
            "raw_metric_space": a_only_result["raw_metric_space"],
            "action_dim": a_only_result["action_dim"],
            "chunk_size": a_only_result["chunk_size"],
        },
        "relative_reduction": {
            "normalized_action_l1": normalized_reduction,
            "raw_action_l1": raw_reduction,
            "frame_l1_p95": p95_reduction,
        },
        "rows": comparison_rows,
        "chunking_rows": metric_rows,
        "sources": {
            "a_only_result": Path(args.a_only_result).as_posix(),
            "abc_result": Path(args.abc_result).as_posix(),
            "week3_config": Path(args.week3_config).as_posix(),
            "act_defaults": Path(args.act_defaults).as_posix(),
            "a_only_manifest": Path(args.a_only_manifest).as_posix(),
            "abc_manifest": Path(args.abc_manifest).as_posix(),
        },
    }

    write_json(results_dir / "day6_action_chunking_robustness.json", payload)
    write_csv_rows(results_dir / "day6_action_chunking_robustness.csv", metric_rows)
    write_markdown_table(results_dir / "day6_action_chunking_robustness.md", "Week3 Day6 Topic2 Action Chunking Robustness", metric_rows)
    write_json(results_dir / "day6_topic2_hyperparams.json", {"status": "completed", "rows": hp_rows})
    write_csv_rows(results_dir / "day6_topic2_hyperparams.csv", hp_rows)
    write_markdown_table(results_dir / "day6_topic2_hyperparams.md", "Week3 Day6 Topic2 Hyperparameters", hp_rows)

    plot_training_curve(figures_dir / "topic2_train_action_l1_curve.png", a_only_paths.loss_components_path, abc_paths.loss_components_path)
    plot_splitd_bar(figures_dir / "topic2_splitD_action_l1_bar.png", comparison_rows)
    plot_per_chunk(figures_dir / "topic2_splitD_per_chunk_l1.png", a_only_result, abc_result)
    plot_distribution(figures_dir / "topic2_splitD_error_distribution.png", a_only_result, abc_result)
    return payload


def main() -> int:
    payload = build_outputs(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
