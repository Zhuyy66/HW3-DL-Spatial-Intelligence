"""Local static tests for the Day 5 LeRobot ACT CALVIN wrapper skeleton."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from topic2_act.eval.lerobot_act_wrapper import LeRobotACTWrapper


class LeRobotACTWrapperTest(unittest.TestCase):
    def test_missing_checkpoint_raises_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing"
            with self.assertRaises(FileNotFoundError):
                LeRobotACTWrapper(missing)

    def test_resolves_latest_lerobot_checkpoint_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            for step in (10, 2):
                ckpt = run_dir / "lerobot_train" / "checkpoints" / str(step) / "pretrained_model"
                ckpt.mkdir(parents=True)
                (ckpt / "model.safetensors").write_bytes(b"placeholder")

            wrapper = LeRobotACTWrapper(run_dir, load_weights=False)

            self.assertEqual(wrapper.checkpoint_info.checkpoint_step, 10)
            model_file = Path(str(wrapper.checkpoint_info.model_file))
            self.assertEqual(model_file.name, "model.safetensors")
            self.assertEqual(model_file.parent.name, "pretrained_model")
            self.assertEqual(model_file.parent.parent.name, "10")
            self.assertFalse(wrapper.checkpoint_info.loaded)

    def test_reset_and_step_return_zero_arm_open_gripper_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "pretrained_model"
            model_dir.mkdir()
            (model_dir / "model.safetensors").write_bytes(b"placeholder")

            wrapper = LeRobotACTWrapper(model_dir, action_dim=7, load_weights=False)
            wrapper.step({}, "goal")
            wrapper.reset()
            action = wrapper.step({"obs": 1}, "goal")

            self.assertEqual(wrapper.step_count, 1)
            self.assertEqual(action.shape, (7,))
            self.assertEqual(action.dtype, np.float32)
            self.assertTrue(np.all(action[:6] == 0.0))
            self.assertEqual(float(action[-1]), 1.0)

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") and importlib.util.find_spec("torch"),
        "safetensors and torch are required for tensor loading",
    )
    def test_loads_small_safetensors_checkpoint(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "pretrained_model"
            model_dir.mkdir()
            save_file({"linear.weight": torch.zeros((1, 2))}, str(model_dir / "model.safetensors"))

            wrapper = LeRobotACTWrapper(model_dir)

            self.assertTrue(wrapper.checkpoint_info.loaded)
            self.assertEqual(wrapper.checkpoint_info.key_count, 1)
            self.assertEqual(wrapper.checkpoint_info.tensor_count, 1)
            self.assertEqual(wrapper.checkpoint_info.loader, "safetensors.torch.load_file")


if __name__ == "__main__":
    unittest.main()
