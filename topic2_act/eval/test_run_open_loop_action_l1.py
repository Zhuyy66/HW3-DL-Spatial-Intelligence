"""Unit tests for open-loop ACT Action L1 aggregation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from topic2_act.eval.run_open_loop_action_l1 import L1Accumulator, build_action_chunk


class ActionChunkTest(unittest.TestCase):
    def test_build_action_chunk_clamps_tail_and_marks_padding(self) -> None:
        actions = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)

        chunk, is_pad = build_action_chunk(actions, frame_index=1, chunk_size=4)

        self.assertEqual(chunk.tolist(), [[2.0, 20.0], [3.0, 30.0], [3.0, 30.0], [3.0, 30.0]])
        self.assertEqual(is_pad.tolist(), [False, False, True, True])

    def test_build_action_chunk_rejects_bad_frame_index(self) -> None:
        actions = np.zeros((2, 7), dtype=np.float32)

        with self.assertRaises(IndexError):
            build_action_chunk(actions, frame_index=2, chunk_size=3)


class L1AccumulatorTest(unittest.TestCase):
    def test_valid_mean_and_forward_equivalent_ignore_padded_values_differently(self) -> None:
        accumulator = L1Accumulator(chunk_size=3, action_dim=2)
        abs_error = np.asarray(
            [
                [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
                [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
            ],
            dtype=np.float32,
        )
        is_pad = np.asarray([[False, False, True], [False, True, True]])

        accumulator.update(abs_error, is_pad)
        summary = accumulator.summary()

        self.assertAlmostEqual(summary["action_l1_valid_mean"], (1 + 3 + 5 + 7 + 2 + 4) / 6)
        self.assertAlmostEqual(summary["forward_l1_loss_equivalent"], (1 + 3 + 5 + 7 + 2 + 4) / 12)
        self.assertEqual(summary["valid_action_element_count"], 6)
        self.assertEqual(summary["forward_action_element_count"], 12)
        self.assertEqual(summary["per_dim_l1"], [8 / 3, 14 / 3])
        self.assertEqual(summary["per_chunk_counts"], [4, 2, 0])
        self.assertEqual(summary["per_chunk_l1"][2], None)
        self.assertEqual(summary["frame_l1_distribution"]["count"], 2)
        self.assertAlmostEqual(summary["frame_l1_distribution"]["mean"], ((1 + 3 + 5 + 7) / 4 + (2 + 4) / 2) / 2)
        self.assertEqual(len(summary["frame_l1_distribution"]["histogram_counts"]), 40)

    def test_raw_metrics_are_optional(self) -> None:
        accumulator = L1Accumulator(chunk_size=1, action_dim=2)
        abs_error = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
        raw_abs_error = np.asarray([[[10.0, 20.0]]], dtype=np.float32)
        is_pad = np.asarray([[False]])

        accumulator.update(abs_error, is_pad, raw_abs_error=raw_abs_error)
        summary = accumulator.summary()

        self.assertAlmostEqual(summary["raw_action_l1_valid_mean"], 15.0)
        self.assertEqual(summary["raw_per_dim_l1"], [10.0, 20.0])
        self.assertEqual(summary["raw_frame_l1_distribution"]["count"], 1)
        self.assertAlmostEqual(summary["raw_frame_l1_distribution"]["p50"], 15.0)


if __name__ == "__main__":
    unittest.main()
