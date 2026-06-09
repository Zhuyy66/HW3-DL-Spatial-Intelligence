"""Persistent LeRobot ACT worker for the Day 6 CALVIN bridge.

Run this script with the Python environment that contains LeRobot 0.4.0.  The
CALVIN wrapper talks to it through length-prefixed pickle frames on
stdin/stdout.  Ordinary logs must stay on stderr or a file so stdout remains a
clean binary protocol stream.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from topic2_act.eval.bridge_protocol import read_frame, write_frame


LOGGER = logging.getLogger(__name__)
STATIC_KEY = "observation.images.image"
GRIPPER_KEY = "observation.images.wrist_image"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


class LeRobotACTWorkerCore:
    """Load and run a LeRobot ACT policy in the robot training environment."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        action_dim: int = 7,
        strict_policy: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = str(device)
        self.action_dim = int(action_dim)
        self.strict_policy = bool(strict_policy)
        self.step_count = 0
        self.reset_count = 0
        self.policy: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.torch: Any = None
        self.policy_config: Any = None
        self.input_features: dict[str, Any] = {}
        self.output_features: dict[str, Any] = {}

        self._load_policy()

    def _load_policy(self) -> None:
        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"model.safetensors not found in checkpoint directory: {self.checkpoint}")

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        torch.backends.cudnn.benchmark = False

        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        if not isinstance(config, ACTConfig):
            raise TypeError(
                f"checkpoint policy config must be ACTConfig, got {type(config).__name__}; "
                f"checkpoint={self.checkpoint}"
            )
        config.device = self.device
        self.policy = ACTPolicy.from_pretrained(
            self.checkpoint,
            config=config,
            strict=self.strict_policy,
        )
        self.policy.eval()
        self.policy_config = self.policy.config
        self.input_features = dict(self.policy.config.input_features)
        self.output_features = dict(self.policy.config.output_features)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": self.device}},
        )
        LOGGER.info("loaded ACT policy checkpoint=%s device=%s", self.checkpoint, self.device)
        LOGGER.info("policy input_features=%s", self._feature_summary(self.input_features))
        LOGGER.info("policy output_features=%s", self._feature_summary(self.output_features))

    def ready_payload(self) -> dict[str, Any]:
        return {
            "type": "ready",
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "action_dim": self.action_dim,
            "input_features": self._feature_summary(self.input_features),
            "output_features": self._feature_summary(self.output_features),
            "chunk_size": int(getattr(self.policy.config, "chunk_size", -1)),
            "n_action_steps": int(getattr(self.policy.config, "n_action_steps", -1)),
        }

    def reset(self) -> dict[str, Any]:
        self.step_count = 0
        self.reset_count += 1
        if self.policy is not None:
            self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if processor is not None and hasattr(processor, "reset"):
                processor.reset()
        LOGGER.info("worker reset reset_count=%s", self.reset_count)
        return {"type": "reset_ok", "reset_count": self.reset_count}

    def step(self, obs: dict[str, Any], goal: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
        if self.policy is None:
            raise RuntimeError("policy is not loaded")

        self.step_count += 1
        observation = self._build_lerobot_observation(obs, goal)
        with self.torch.inference_mode():
            observation = self._prepare_observation_for_inference(observation)
            batch = self.preprocessor(observation)
            action_tensor = self.policy.select_action(batch)
            action_tensor = self.postprocessor(action_tensor)

        raw_action = self._tensor_to_numpy(action_tensor)
        action = self._format_calvin_action(raw_action)
        meta = {
            "step_count": self.step_count,
            "action_shape": list(action.shape),
            "action_norm": float(np.linalg.norm(action[:6])),
            "raw_action_norm": float(np.linalg.norm(raw_action.reshape(-1)[: min(6, raw_action.size)])),
            "gripper": float(action[-1]),
            "raw_min": float(np.nanmin(raw_action)),
            "raw_max": float(np.nanmax(raw_action)),
        }
        if self.step_count <= 5 or self.step_count % 20 == 0:
            LOGGER.info("worker step meta=%s", json.dumps(meta, sort_keys=True))
        return action, meta

    @staticmethod
    def action_response(action: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
        """Build a pickle-safe action response for the CALVIN Python 3.8 side."""

        action_list = np.asarray(action, dtype=np.float32).reshape(-1).astype(np.float32).tolist()
        return {"type": "action", "action": action_list, "meta": meta}

    def _build_lerobot_observation(self, obs: dict[str, Any], goal: Any) -> dict[str, np.ndarray]:
        if not isinstance(obs, dict):
            raise TypeError(f"CALVIN obs must be a dict, got {type(obs).__name__}")

        observation: dict[str, np.ndarray] = {}
        if STATIC_KEY in self.input_features:
            observation[STATIC_KEY] = self._prepare_image_feature(
                self._extract_image(obs, canonical_key=STATIC_KEY, calvin_key="rgb_static"),
                STATIC_KEY,
                fallback_hw=(200, 200),
            )
        if GRIPPER_KEY in self.input_features:
            observation[GRIPPER_KEY] = self._prepare_image_feature(
                self._extract_image(obs, canonical_key=GRIPPER_KEY, calvin_key="rgb_gripper"),
                GRIPPER_KEY,
                fallback_hw=(84, 84),
            )
        if STATE_KEY in self.input_features:
            expected_len = self._feature_length(STATE_KEY, fallback=15)
            observation[STATE_KEY] = self._prepare_state_feature(self._extract_state(obs), expected_len)

        if goal is not None:
            observation["task"] = str(goal)
        return observation

    def _prepare_observation_for_inference(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        from lerobot.policies.utils import prepare_observation_for_inference

        task = str(observation.pop("task", ""))
        return prepare_observation_for_inference(observation, self.torch.device(self.device), task=task)

    def _extract_image(self, obs: dict[str, Any], canonical_key: str, calvin_key: str) -> Any:
        if canonical_key in obs:
            return obs[canonical_key]
        if calvin_key in obs:
            return obs[calvin_key]
        rgb_obs = obs.get("rgb_obs")
        if isinstance(rgb_obs, dict) and calvin_key in rgb_obs:
            return rgb_obs[calvin_key]
        raise KeyError(f"missing CALVIN image key {calvin_key!r} / canonical key {canonical_key!r}")

    def _extract_state(self, obs: dict[str, Any]) -> Any:
        if STATE_KEY in obs:
            return obs[STATE_KEY]
        if "robot_obs" in obs:
            return obs["robot_obs"]
        state_obs = obs.get("state_obs")
        if isinstance(state_obs, dict) and "robot_obs" in state_obs:
            return state_obs["robot_obs"]
        raise KeyError("missing CALVIN robot_obs / canonical observation.state")

    def _prepare_image_feature(self, image: Any, feature_key: str, fallback_hw: tuple[int, int]) -> np.ndarray:
        arr = self._as_hwc_uint8(image)
        expected_h, expected_w = self._feature_image_hw(feature_key, fallback_hw)
        if arr.shape[0] != expected_h or arr.shape[1] != expected_w:
            arr = self._resize_hwc_uint8(arr, expected_h, expected_w)
        return arr

    def _as_hwc_uint8(self, image: Any) -> np.ndarray:
        if self.torch is not None and isinstance(image, self.torch.Tensor):
            image = image.detach().cpu().numpy()
        arr = np.asarray(image)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"image must be 3D or batch-1 4D, got shape {arr.shape}")

        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] != 3:
            raise ValueError(f"image last dimension must be 1, 3, or 4 channels, got shape {arr.shape}")

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _resize_hwc_uint8(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        try:
            import cv2

            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001 - Pillow is a safe fallback in env_hw3_robot.
            from PIL import Image

            resized = np.asarray(Image.fromarray(image).resize((width, height), resample=Image.BILINEAR))
        return np.ascontiguousarray(resized.astype(np.uint8))

    def _prepare_state_feature(self, robot_obs: Any, expected_len: int) -> np.ndarray:
        if self.torch is not None and isinstance(robot_obs, self.torch.Tensor):
            robot_obs = robot_obs.detach().cpu().numpy()
        flat = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
        if flat.size == expected_len:
            return flat.astype(np.float32)
        if flat.size == 16 and expected_len == 15:
            euler = np.asarray(_quat_xyzw_to_euler(flat[3:7]), dtype=np.float32)
            converted = np.concatenate([flat[:3], euler, flat[7:8], flat[8:15], flat[15:16]])
            return converted.astype(np.float32)
        raise ValueError(f"robot_obs has length {flat.size}, expected {expected_len}")

    def _tensor_to_numpy(self, action: Any) -> np.ndarray:
        if self.torch is not None and isinstance(action, self.torch.Tensor):
            action = action.detach().cpu().numpy()
        arr = np.asarray(action, dtype=np.float32)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.reshape(-1).astype(np.float32)

    def _format_calvin_action(self, raw_action: np.ndarray) -> np.ndarray:
        if raw_action.size != self.action_dim:
            raise ValueError(f"policy action has {raw_action.size} dims, expected {self.action_dim}")
        if not np.isfinite(raw_action).all():
            raise ValueError(f"policy action contains non-finite values: {raw_action}")
        action = raw_action.astype(np.float32).copy()
        if self.action_dim >= 7:
            action[:6] = np.clip(action[:6], -1.0, 1.0)
            action[-1] = 1.0 if action[-1] >= 0.0 else -1.0
        return action.astype(np.float32)

    def _feature_image_hw(self, key: str, fallback: tuple[int, int]) -> tuple[int, int]:
        shape = self._feature_shape(key)
        if len(shape) == 3:
            if shape[0] in (1, 3, 4):
                return int(shape[1]), int(shape[2])
            return int(shape[0]), int(shape[1])
        return fallback

    def _feature_length(self, key: str, fallback: int) -> int:
        shape = self._feature_shape(key)
        if len(shape) == 1:
            return int(shape[0])
        return fallback

    def _feature_shape(self, key: str) -> tuple[int, ...]:
        feature = self.input_features.get(key) or self.output_features.get(key)
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if shape is None:
            return ()
        return tuple(int(dim) for dim in shape)

    def _feature_summary(self, features: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key, feature in features.items():
            feature_type = getattr(getattr(feature, "type", None), "value", getattr(feature, "type", None))
            summary[key] = {
                "shape": list(self._feature_shape(key)),
                "type": str(feature_type),
            }
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent LeRobot ACT bridge worker.")
    parser.add_argument("--checkpoint", required=True, help="LeRobot pretrained_model checkpoint directory.")
    parser.add_argument("--device", default="cpu", help="LeRobot inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--strict-policy", action="store_true", help="Use strict weight loading.")
    parser.add_argument("--log-file", default=None, help="Optional worker log file.")
    return parser.parse_args()


def configure_logging(log_file: str | None) -> None:
    handlers: list[logging.Handler] = []
    if log_file:
        Path(log_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def worker_loop(core: LeRobotACTWorkerCore) -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    write_frame(stdout, core.ready_payload())

    while True:
        try:
            message = read_frame(stdin)
        except EOFError:
            LOGGER.info("bridge stdin closed")
            return 0

        msg_type = message.get("type")
        try:
            if msg_type == "reset":
                write_frame(stdout, core.reset())
            elif msg_type == "step":
                action, meta = core.step(message.get("obs"), goal=message.get("goal"))
                write_frame(stdout, core.action_response(action, meta))
            elif msg_type == "close":
                write_frame(stdout, {"type": "closed"})
                return 0
            else:
                raise ValueError(f"unknown bridge message type: {msg_type!r}")
        except Exception as exc:  # noqa: BLE001 - send exact worker failure to CALVIN side.
            LOGGER.exception("worker failed to handle message type=%s", msg_type)
            write_frame(
                stdout,
                {
                    "type": "error",
                    "message": str(exc),
                    "repr": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file)
    LOGGER.info("starting worker checkpoint=%s device=%s", args.checkpoint, args.device)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            core = LeRobotACTWorkerCore(
                args.checkpoint,
                device=args.device,
                action_dim=args.action_dim,
                strict_policy=args.strict_policy,
            )
    except Exception as exc:  # noqa: BLE001 - startup failures should be visible in worker logs.
        LOGGER.exception("worker startup failed")
        print(f"worker startup failed: {exc}", file=sys.stderr)
        return 1
    return worker_loop(core)


def _quat_xyzw_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = [float(v) for v in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


if __name__ == "__main__":
    raise SystemExit(main())
