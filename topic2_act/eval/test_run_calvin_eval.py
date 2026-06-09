"""Local static tests for CALVIN eval launcher policy handling."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from topic2_act.eval.run_calvin_eval import audit_egl, clear_cameras_for_direct_policy


class RunCalvinEvalPolicyTest(unittest.TestCase):
    def test_direct_egl_policy_skips_mapping(self) -> None:
        audit = audit_egl(cuda_device=0, egl_policy="direct")

        self.assertEqual(audit["egl_policy"], "direct")
        self.assertIsNone(audit["egl_device"])
        self.assertIsNone(audit["egl_mapping_error"])
        self.assertIn("direct policy", str(audit["fallback"]))

    def test_clear_cameras_for_direct_policy(self) -> None:
        render_conf = SimpleNamespace(
            cameras={"static": object(), "gripper": object(), "tactile": object()},
            env=SimpleNamespace(cameras={"static": object(), "gripper": object(), "tactile": object()}),
        )

        clear_cameras_for_direct_policy(render_conf)

        self.assertEqual(render_conf.cameras, {})
        self.assertEqual(render_conf.env.cameras, {})


if __name__ == "__main__":
    unittest.main()
