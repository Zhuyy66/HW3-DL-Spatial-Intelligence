"""Unit tests for Week 3 Day 5 result summarization."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from topic2_act.eval.summarize_day5_results import (
    build_outputs,
    extract_labeled_json,
    validate_action_l1_pair,
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
        "chunk_size": 3,
        "metric_space": "normalized_action",
        "raw_metric_space": "dataset_action_units",
        "checkpoint": f"/tmp/{model}/pretrained_model",
        "dataset_root": "/tmp/splitD",
        "per_chunk_l1": [normalized, normalized + 0.1, None],
        "raw_per_chunk_l1": [raw, raw + 0.1, None],
        "per_dim_l1": [normalized] * 7,
        "raw_per_dim_l1": [raw] * 7,
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
    return (
        "prefix\n"
        "single_rollout_smoke_result:\n"
        "Exception ignored during cleanup\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


class Day5SummaryTest(unittest.TestCase):
    def test_extract_labeled_json_tolerates_cleanup_noise(self) -> None:
        payload = extract_labeled_json(smoke_log(0.25), "single_rollout_smoke_result:")

        self.assertEqual(payload["last_action_shape"], [7])
        self.assertTrue(payload["moved"])

    def test_validate_action_l1_pair_rejects_mismatch(self) -> None:
        a_only = result_payload("a_only", 0.6, 0.2)
        abc = result_payload("abc", 0.5, 0.1)
        abc["chunk_size"] = 100

        with self.assertRaises(ValueError):
            validate_action_l1_pair(a_only, abc)

    def test_build_outputs_writes_tables_and_figures(self) -> None:
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

            write_json(a_only_result, result_payload("a_only", 0.6, 0.2))
            write_json(abc_result, result_payload("abc", 0.5, 0.1))
            a_only_smoke.write_text(smoke_log(0.2), encoding="utf-8")
            abc_smoke.write_text(smoke_log(0.3), encoding="utf-8")
            write_jsonl(a_only_loss, [{"step": 1, "action_l1": 0.3}, {"step": 2, "action_l1": 0.2}])
            write_jsonl(abc_loss, [{"step": 1, "action_l1": 0.4}, {"step": 2, "action_l1": 0.1}])

            comparison = build_outputs(
                Namespace(
                    results_dir=str(results_dir),
                    figures_dir=str(figures_dir),
                    a_only_result=str(a_only_result),
                    abc_result=str(abc_result),
                    a_only_smoke_log=str(a_only_smoke),
                    abc_smoke_log=str(abc_smoke),
                    a_only_loss_components=str(a_only_loss),
                    abc_loss_components=str(abc_loss),
                )
            )

            self.assertEqual(comparison["status"], "completed")
            self.assertEqual(len(comparison["rows"]), 2)
            self.assertTrue((results_dir / "day5_topic2_a_only_vs_abc.json").is_file())
            self.assertTrue((results_dir / "day5_topic2_a_only_vs_abc.csv").is_file())
            self.assertTrue((results_dir / "day5_topic2_a_only_vs_abc.md").is_file())
            self.assertTrue((results_dir / "a_only_direct_cameras_smoke.json").is_file())
            self.assertTrue((results_dir / "abc_direct_cameras_smoke.json").is_file())
            self.assertTrue((figures_dir / "topic2_train_action_l1_curve.png").is_file())
            self.assertTrue((figures_dir / "topic2_splitD_action_l1_bar.png").is_file())
            self.assertTrue((figures_dir / "topic2_splitD_per_chunk_l1.png").is_file())


if __name__ == "__main__":
    unittest.main()
