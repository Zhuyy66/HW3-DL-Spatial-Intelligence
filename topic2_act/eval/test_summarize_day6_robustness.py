"""Unit tests for Day 6 robustness summarization."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from topic2_act.eval.summarize_day6_robustness import (
    build_metric_rows,
    build_outputs,
    read_yaml_scalars,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def result_payload(model: str, normalized: float, raw: float) -> dict:
    return {
        "status": "completed",
        "action_l1_valid_mean": normalized,
        "raw_action_l1_valid_mean": raw,
        "episode_count": 5124,
        "frame_count": 308918,
        "missing_parquet_count": 0,
        "action_dim": 7,
        "chunk_size": 5,
        "metric_space": "normalized_action",
        "raw_metric_space": "dataset_action_units",
        "checkpoint": f"/tmp/{model}/pretrained_model",
        "dataset_root": "/tmp/splitD",
        "per_chunk_l1": [normalized, normalized + 0.1, normalized + 0.2, None, None],
        "raw_per_chunk_l1": [raw, raw + 0.1, raw + 0.2, None, None],
        "per_dim_l1": [normalized] * 7,
        "raw_per_dim_l1": [raw] * 7,
        "frame_l1_distribution": {
            "count": 4,
            "mean": normalized,
            "std": 0.1,
            "min": normalized - 0.1,
            "p05": normalized - 0.09,
            "p25": normalized - 0.05,
            "p50": normalized,
            "p75": normalized + 0.05,
            "p95": normalized + 0.09,
            "max": normalized + 0.1,
            "histogram_bin_edges": [0.0, 0.5, 1.0],
            "histogram_counts": [1, 3],
        },
        "raw_frame_l1_distribution": {
            "count": 4,
            "mean": raw,
            "std": 0.1,
            "min": raw - 0.1,
            "p05": raw - 0.09,
            "p25": raw - 0.05,
            "p50": raw,
            "p75": raw + 0.05,
            "p95": raw + 0.09,
            "max": raw + 0.1,
            "histogram_bin_edges": [0.0, 0.5, 1.0],
            "histogram_counts": [2, 2],
        },
    }


def smoke_log(max_tcp_delta: float) -> str:
    payload = {
        "steps": 60,
        "wrapper_step_count": 60,
        "last_action_shape": [7],
        "moved": True,
        "max_tcp_delta": max_tcp_delta,
        "gripper_values": [-1.0, 1.0],
    }
    return "single_rollout_smoke_result:\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def manifest_payload(name: str, episodes: int, frames: int) -> dict:
    return {
        "run_name": name,
        "selected_episode_count": episodes,
        "selected_frame_count": frames,
        "batch_size": 8,
        "planned_steps": 150000,
        "num_workers": 16,
        "dataloader_prefetch_factor": 4,
        "dataloader_persistent_workers": True,
        "save_freq": 50000,
        "wandb_url": f"https://wandb.ai/example/{name}",
    }


def week3_yaml() -> str:
    return """
policy:
  type: act
  chunk_size: 100
  n_action_steps: 100
optimization:
  optimizer: adamw
  learning_rate: 1.0e-5
  weight_decay: 1.0e-4
  batch_size: 8
  seed: 20260529
data:
  schema:
    state_shape: [15]
    action_shape: [7]
    image_shape: [200, 200, 3]
    wrist_image_shape: [84, 84, 3]
"""


def act_defaults() -> dict:
    return {
        "act_config": {
            "vision_backbone": "resnet18",
            "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
            "dim_model": 512,
            "n_heads": 8,
            "n_encoder_layers": 4,
            "n_decoder_layers": 1,
            "dim_feedforward": 3200,
            "use_vae": True,
            "latent_dim": 32,
            "dropout": 0.1,
            "kl_weight": 10.0,
        }
    }


class Day6RobustnessSummaryTest(unittest.TestCase):
    def test_read_yaml_scalars_handles_nested_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(week3_yaml(), encoding="utf-8")

            scalars = read_yaml_scalars(path)

            self.assertEqual(scalars["policy.chunk_size"], 100)
            self.assertEqual(scalars["data.schema.action_shape"], [7])

    def test_build_metric_rows_uses_distribution_and_chunk_windows(self) -> None:
        rows = build_metric_rows(result_payload("a", 0.6, 0.2), result_payload("b", 0.5, 0.1))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["distribution_count"], 4)
        self.assertEqual(rows[0]["chunk_offset_tail"], 2)

    def test_build_outputs_writes_day6_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "results"
            figures_dir = root / "figures"
            a_only_result = root / "a_only.json"
            abc_result = root / "abc.json"
            a_only_smoke = root / "a_only_smoke.log"
            abc_smoke = root / "abc_smoke.log"
            a_only_loss = root / "a_only_loss.jsonl"
            abc_loss = root / "abc_loss.jsonl"
            a_manifest = root / "a_manifest.json"
            abc_manifest = root / "abc_manifest.json"
            yaml_path = root / "week3.yaml"
            defaults_path = root / "defaults.txt"

            write_json(a_only_result, result_payload("a_only", 0.6, 0.2))
            write_json(abc_result, result_payload("abc", 0.5, 0.1))
            a_only_smoke.write_text(smoke_log(0.2), encoding="utf-8")
            abc_smoke.write_text(smoke_log(0.3), encoding="utf-8")
            write_jsonl(a_only_loss, [{"step": 1, "action_l1": 0.3}, {"step": 2, "action_l1": 0.2}])
            write_jsonl(abc_loss, [{"step": 1, "action_l1": 0.4}, {"step": 2, "action_l1": 0.1}])
            write_json(a_manifest, manifest_payload("a", 6089, 366693))
            write_json(abc_manifest, manifest_payload("abc", 17870, 1071743))
            yaml_path.write_text(week3_yaml(), encoding="utf-8")
            defaults_path.write_text("# dump\n" + json.dumps(act_defaults()), encoding="utf-8")

            payload = build_outputs(
                Namespace(
                    results_dir=str(results_dir),
                    figures_dir=str(figures_dir),
                    a_only_result=str(a_only_result),
                    abc_result=str(abc_result),
                    a_only_smoke_log=str(a_only_smoke),
                    abc_smoke_log=str(abc_smoke),
                    a_only_loss_components=str(a_only_loss),
                    abc_loss_components=str(abc_loss),
                    a_only_manifest=str(a_manifest),
                    abc_manifest=str(abc_manifest),
                    week3_config=str(yaml_path),
                    act_defaults=str(defaults_path),
                )
            )

            self.assertEqual(payload["status"], "completed")
            self.assertTrue((results_dir / "day6_action_chunking_robustness.json").is_file())
            self.assertTrue((results_dir / "day6_action_chunking_robustness.csv").is_file())
            self.assertTrue((results_dir / "day6_topic2_hyperparams.md").is_file())
            self.assertTrue((figures_dir / "topic2_splitD_error_distribution.png").is_file())


if __name__ == "__main__":
    unittest.main()
