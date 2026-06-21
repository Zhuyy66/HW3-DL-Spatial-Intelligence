from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_act_train as train  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class RunActTrainPrepareTest(unittest.TestCase):
    def install_fake_act_policy(self, forward: object):
        lerobot = types.ModuleType("lerobot")
        policies = types.ModuleType("lerobot.policies")
        act = types.ModuleType("lerobot.policies.act")
        modeling_act = types.ModuleType("lerobot.policies.act.modeling_act")

        class FakeACTPolicy:
            pass

        FakeACTPolicy.forward = forward  # type: ignore[method-assign]
        modeling_act.ACTPolicy = FakeACTPolicy
        lerobot.policies = policies
        policies.act = act
        act.modeling_act = modeling_act
        patcher = mock.patch.dict(
            sys.modules,
            {
                "lerobot": lerobot,
                "lerobot.policies": policies,
                "lerobot.policies.act": act,
                "lerobot.policies.act.modeling_act": modeling_act,
            },
        )
        return FakeACTPolicy, patcher

    def test_datasets_disk_check_bypass_restores_original(self) -> None:
        package = types.ModuleType("datasets")
        builder = types.ModuleType("datasets.builder")

        def original_check(*_args: object, **_kwargs: object) -> bool:
            return False

        builder.has_sufficient_disk_space = original_check
        package.builder = builder

        with mock.patch.dict(sys.modules, {"datasets": package, "datasets.builder": builder}):
            with train.datasets_disk_check_bypass(True) as bypassed:
                self.assertTrue(bypassed)
                self.assertIsNot(builder.has_sufficient_disk_space, original_check)
                self.assertTrue(builder.has_sufficient_disk_space())

            self.assertIs(builder.has_sufficient_disk_space, original_check)

            with train.datasets_disk_check_bypass(False) as bypassed:
                self.assertFalse(bypassed)
                self.assertIs(builder.has_sufficient_disk_space, original_check)

    def test_action_component_audit_records_tuple_forward_output(self) -> None:
        def original_forward(_self: object, _batch: dict[str, object]) -> tuple[float, dict[str, float]]:
            return 12.5, {"l1_loss": 3.25}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_path = root / "loss_components.jsonl"
            manifest_path = root / "run_manifest.json"
            write_json(manifest_path, {})
            FakeACTPolicy, patcher = self.install_fake_act_policy(original_forward)

            with patcher:
                restore = train.install_action_component_audit_patch(
                    {
                        "loss_components_path": str(audit_path),
                        "manifest_path": str(manifest_path),
                        "log_freq": 1,
                    }
                )
                try:
                    output = FakeACTPolicy().forward({})
                finally:
                    restore()

                self.assertIs(FakeACTPolicy.forward, original_forward)

            self.assertEqual(output[0], 12.5)
            self.assertEqual(output[1]["action_l1"], 3.25)
            row = json.loads(audit_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["loss"], 12.5)
            self.assertEqual(row["action_l1"], 3.25)
            self.assertEqual(row["source"], "ACTPolicy.forward loss_dict['l1_loss']")
            self.assertEqual(row["forward_return_type"], "tuple")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["action_l1_source"], "ACTPolicy.forward loss_dict['l1_loss']")

    def test_action_component_audit_keeps_dict_forward_behavior(self) -> None:
        def original_forward(_self: object, _batch: dict[str, object]) -> dict[str, float]:
            return {"loss": 9.0, "l1_loss": 1.5}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_path = root / "loss_components.jsonl"
            manifest_path = root / "run_manifest.json"
            write_json(manifest_path, {})
            FakeACTPolicy, patcher = self.install_fake_act_policy(original_forward)

            with patcher:
                restore = train.install_action_component_audit_patch(
                    {
                        "loss_components_path": str(audit_path),
                        "manifest_path": str(manifest_path),
                        "log_freq": 1,
                    }
                )
                try:
                    output = FakeACTPolicy().forward({})
                finally:
                    restore()

            self.assertEqual(output["action_l1"], 1.5)
            row = json.loads(audit_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["loss"], 9.0)
            self.assertEqual(row["action_l1"], 1.5)
            self.assertEqual(row["forward_return_type"], "dict")

    def test_rejects_in_place_v21_prepare_without_deleting_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "splitA"
            write_json(root / "meta" / "info.json", {"codebase_version": "v2.1"})
            episodes = Path(tmp) / "episodes.json"
            write_json(episodes, [0])
            marker = root / "do_not_delete.txt"
            marker.write_text("source", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "refusing to prepare"):
                train.prepare_dataset(
                    dataset_root=root,
                    episodes_file=episodes,
                    prepared_root=root,
                    rebuild=False,
                )

            self.assertTrue(marker.exists())

    def test_accepts_in_place_v3_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "abc_joint_canonical_v3"
            write_json(root / "meta" / "info.json", {"codebase_version": "v3.0"})
            episodes = Path(tmp) / "episodes_ABC_full.json"
            write_json(episodes, [0, 1])

            with mock.patch.object(train, "read_v3_episode_lengths", return_value={0: 4, 1: 6}):
                prepared_root, selected, manifest = train.prepare_dataset(
                    dataset_root=root,
                    episodes_file=episodes,
                    prepared_root=root,
                    rebuild=False,
                )

            self.assertEqual(prepared_root, root)
            self.assertEqual(selected, [0, 1])
            self.assertTrue(manifest["prepared_dataset_passthrough"])
            self.assertEqual(manifest["source_codebase_version"], "v3.0")
            self.assertEqual(manifest["selected_episode_count"], 2)
            self.assertEqual(manifest["selected_frame_count"], 10)
            self.assertTrue((root / ".hw3_prepare_manifest.json").exists())

    def test_compute_episode_frames_uses_v3_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "abc_joint_canonical_v3"
            write_json(root / "meta" / "info.json", {"codebase_version": "v3.0"})

            with mock.patch.object(train, "read_v3_episode_lengths", return_value={0: 4, 1: 6}):
                frame_count, lengths = train.compute_episode_frames(root, [0, 1])

            self.assertEqual(frame_count, 10)
            self.assertEqual(lengths, {0: 4, 1: 6})

    def test_disk_bypass_flag_written_to_worker_config_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "splitA"
            episodes = root / "episodes_A_full.json"
            run_dir = root / "run"
            prepared_root = root / "prepared_v3"
            write_json(episodes, [0])

            argv = [
                "run_act_train.py",
                "--dataset-root",
                str(dataset_root),
                "--episodes-file",
                str(episodes),
                "--output-dir",
                str(run_dir),
                "--run-name",
                "test_disk_bypass",
                "--steps",
                "1",
                "--allow-datasets-disk-check-bypass",
            ]
            prepared_manifest = {
                "source_codebase_version": "v2.1",
                "prepared_dataset_passthrough": False,
            }

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(train, "prepare_dataset", return_value=(prepared_root, [0], prepared_manifest)),
                mock.patch.object(train, "compute_episode_frames", return_value=(8, {0: 8})),
                mock.patch.object(train, "stream_worker", return_value=0),
            ):
                return_code = train.main()

            self.assertEqual(return_code, 0)
            worker_config = json.loads((run_dir / "actual_training_config.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(worker_config["allow_datasets_disk_check_bypass"])
            self.assertTrue(manifest["datasets_disk_check_bypass_requested"])


if __name__ == "__main__":
    unittest.main()
