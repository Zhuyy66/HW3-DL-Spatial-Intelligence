"""Tests for the Day 6 bridge framing protocol."""

from __future__ import annotations

import io
import struct
import unittest

import numpy as np

from topic2_act.eval.bridge_protocol import BridgeProtocolError, read_frame, write_frame
from topic2_act.eval.lerobot_act_worker import LeRobotACTWorkerCore


class BridgeProtocolTest(unittest.TestCase):
    def test_roundtrip_plain_action_payload(self) -> None:
        stream = io.BytesIO()
        payload = {"type": "action", "action": [float(v) for v in range(7)]}

        write_frame(stream, payload)
        stream.seek(0)
        decoded = read_frame(stream)

        self.assertEqual(decoded["type"], "action")
        self.assertEqual(decoded["action"], payload["action"])

    def test_worker_action_response_uses_plain_python_list(self) -> None:
        action = np.arange(7, dtype=np.float32)
        payload = LeRobotACTWorkerCore.action_response(action, {"step_count": 1})

        self.assertEqual(payload["type"], "action")
        self.assertIsInstance(payload["action"], list)
        self.assertNotIsInstance(payload["action"], np.ndarray)
        self.assertEqual(payload["action"], [float(v) for v in action])

    def test_rejects_invalid_frame_size(self) -> None:
        stream = io.BytesIO(struct.pack(">I", 0))

        with self.assertRaises(BridgeProtocolError):
            read_frame(stream)


if __name__ == "__main__":
    unittest.main()
