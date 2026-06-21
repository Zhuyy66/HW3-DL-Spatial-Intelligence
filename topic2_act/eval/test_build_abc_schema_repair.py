from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_abc_canonical_dataset as abc  # noqa: E402

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None


FEATURE_COLUMNS = [
    "observation.images.image",
    "observation.images.wrist_image",
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "source_frame_index",
    "source_episode_index",
]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@unittest.skipUnless(pa is not None and pq is not None, "pyarrow is required for parquet repair tests")
class BuildABCSchemaRepairTest(unittest.TestCase):
    def write_dataset(self, root: Path, *, extra_columns: dict[str, object]) -> Path:
        info = {
            "codebase_version": "v3.0",
            "total_frames": 2,
            "features": {column: {"dtype": "int64", "shape": [1]} for column in FEATURE_COLUMNS},
        }
        write_json(root / "meta" / "info.json", info)
        parquet_path = root / "data" / "chunk-000" / "file-000.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        columns: dict[str, object] = {column: pa.array([1, 2]) for column in FEATURE_COLUMNS}
        for key, values in extra_columns.items():
            columns[key] = pa.array(values)
        pq.write_table(pa.table(columns), parquet_path)
        return parquet_path

    def test_repair_drops_expected_audit_columns_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "abc_joint_canonical_v3"
            parquet_path = self.write_dataset(
                root,
                extra_columns={
                    "source_task_index": [3, 3],
                    "source_split": ["splitA", "splitA"],
                    "source_scene": ["A", "A"],
                    "source_split_episode_index": [7, 7],
                },
            )

            summary = abc.repair_v3_train_schema(root)

            self.assertEqual(summary["total_rows"], 2)
            self.assertEqual(summary["repaired_files"], 1)
            self.assertEqual(summary["skipped_files"], 0)
            self.assertEqual(summary["unexpected_extra_columns"], [])
            self.assertEqual(summary["missing_feature_columns"], [])
            self.assertTrue(summary["schema_matches_info_features"])
            self.assertEqual(pq.read_table(parquet_path).column_names, FEATURE_COLUMNS)

            resumed = abc.repair_v3_train_schema(root)
            self.assertEqual(resumed["repaired_files"], 0)
            self.assertEqual(resumed["skipped_files"], 1)

    def test_repair_rejects_unknown_extra_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "abc_joint_canonical_v3"
            self.write_dataset(root, extra_columns={"unexpected_column": [1, 2]})

            with self.assertRaisesRegex(ValueError, "unexpected extra columns"):
                abc.repair_v3_train_schema(root)


if __name__ == "__main__":
    unittest.main()
