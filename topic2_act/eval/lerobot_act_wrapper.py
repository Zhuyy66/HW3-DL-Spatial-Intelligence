"""LeRobot ACT wrapper skeleton for CALVIN evaluation.

Day 5 only verifies the CALVIN CustomModel-style interface. The wrapper loads
checkpoint metadata, resets internal state, and returns a zero-arm/open-gripper
placeholder until the real CALVIN observation -> ACT inference path is wired.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # CALVIN is only installed in env_hw3_calvin_eval on the server.
    from calvin_agent.models.calvin_base_model import CalvinBaseModel
except Exception:  # noqa: BLE001 - keep local static tests independent.
    CalvinBaseModel = object  # type: ignore[assignment,misc]


LOGGER = logging.getLogger(__name__)
SUPPORTED_WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin")


@dataclass
class CheckpointInfo:
    """Resolved checkpoint metadata used by the Day 5 smoke tests."""

    input_path: str
    checkpoint_dir: str | None
    model_file: str | None
    checkpoint_step: int | None
    loader: str | None = None
    loaded: bool = False
    key_count: int = 0
    tensor_count: int = 0
    load_error: str | None = None


class LeRobotACTWrapper(CalvinBaseModel):  # type: ignore[misc,valid-type]
    """Minimal CALVIN policy adapter for a LeRobot ACT checkpoint.

    Parameters mirror the Day 5 plan. `step` intentionally returns zero arm
    motion with an open gripper; this file only proves that CALVIN can call our
    policy wrapper.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        action_dim: int = 7,
        strict_checkpoint: bool = True,
        load_weights: bool = True,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.device = str(device)
        self.action_dim = int(action_dim)
        self.strict_checkpoint = bool(strict_checkpoint)
        self.load_weights = bool(load_weights)
        self.step_count = 0
        self.action_chunk: np.ndarray | None = None

        self.checkpoint_info = self._resolve_checkpoint(Path(checkpoint))
        if self.load_weights:
            self._load_checkpoint_weights()

        LOGGER.info("initialized LeRobotACTWrapper: %s", self.checkpoint_summary())

    def reset(self) -> None:
        """Reset per-rollout state before a CALVIN subtask starts."""

        self.step_count = 0
        self.action_chunk = None
        LOGGER.info("LeRobotACTWrapper.reset called")

    def step(self, obs: Any, goal: Any) -> np.ndarray:  # noqa: ARG002 - placeholder interface.
        """Return a zero-arm/open-gripper placeholder action."""

        self.step_count += 1
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[-1] = 1.0
        if self.step_count <= 3:
            LOGGER.info(
                "LeRobotACTWrapper.step called: step_count=%s action_shape=%s goal_type=%s",
                self.step_count,
                action.shape,
                type(goal).__name__,
            )
        return action

    def checkpoint_summary(self) -> dict[str, Any]:
        """Return JSON-serializable checkpoint and wrapper state."""

        payload = asdict(self.checkpoint_info)
        payload.update(
            {
                "device": self.device,
                "action_dim": self.action_dim,
                "strict_checkpoint": self.strict_checkpoint,
                "load_weights": self.load_weights,
                "step_count": self.step_count,
            }
        )
        return payload

    def _resolve_checkpoint(self, checkpoint: Path) -> CheckpointInfo:
        input_path = checkpoint.expanduser()
        model_file: Path | None = None
        checkpoint_dir: Path | None = None
        checkpoint_step: int | None = None

        if input_path.is_file():
            model_file = input_path
            checkpoint_dir = input_path.parent
            checkpoint_step = _infer_step_from_path(input_path)
        elif input_path.is_dir():
            model_file = _find_model_file(input_path)
            if model_file is not None:
                checkpoint_dir = model_file.parent
                checkpoint_step = _infer_step_from_path(model_file)
        elif self.strict_checkpoint:
            raise FileNotFoundError(f"checkpoint path does not exist: {input_path}")

        if model_file is None and self.strict_checkpoint:
            raise FileNotFoundError(
                "could not find model weights under checkpoint path: "
                f"{input_path}; expected model.safetensors or one of {SUPPORTED_WEIGHT_SUFFIXES}"
            )

        return CheckpointInfo(
            input_path=str(input_path),
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
            model_file=str(model_file) if model_file else None,
            checkpoint_step=checkpoint_step,
        )

    def _load_checkpoint_weights(self) -> None:
        model_path = Path(self.checkpoint_info.model_file) if self.checkpoint_info.model_file else None
        if model_path is None:
            if self.strict_checkpoint:
                raise FileNotFoundError("load_weights=True but no model file was resolved")
            return

        try:
            if model_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(model_path), device="cpu")
                loader = "safetensors.torch.load_file"
            else:
                import torch

                state = torch.load(str(model_path), map_location="cpu")
                loader = "torch.load"
        except Exception as exc:  # noqa: BLE001 - expose exact dependency/checkpoint failure.
            self.checkpoint_info.load_error = repr(exc)
            if self.strict_checkpoint:
                raise RuntimeError(f"failed to load checkpoint weights from {model_path}: {exc}") from exc
            LOGGER.warning("failed to load checkpoint weights from %s: %r", model_path, exc)
            return

        tensor_count = 0
        key_count = 0
        if isinstance(state, dict):
            key_count = len(state)
            tensor_count = sum(1 for value in state.values() if _looks_like_tensor(value))
        else:
            key_count = 1
            tensor_count = 1 if _looks_like_tensor(state) else 0

        self.checkpoint_info.loader = loader
        self.checkpoint_info.loaded = True
        self.checkpoint_info.key_count = key_count
        self.checkpoint_info.tensor_count = tensor_count


def _find_model_file(path: Path) -> Path | None:
    direct_candidates = [
        path / "model.safetensors",
        path / "pretrained_model" / "model.safetensors",
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    checkpoint_roots = [path / "lerobot_train" / "checkpoints", path / "checkpoints", path]
    candidates: list[Path] = []
    for root in checkpoint_roots:
        if root.is_dir():
            candidates.extend(root.glob("*/pretrained_model/model.safetensors"))
            for suffix in SUPPORTED_WEIGHT_SUFFIXES:
                candidates.extend(root.glob(f"*/pretrained_model/*{suffix}"))

    if not candidates:
        return None
    return sorted(set(candidates), key=_checkpoint_sort_key)[-1]


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    step = _infer_step_from_path(path)
    return (step if step is not None else -1, str(path))


def _infer_step_from_path(path: Path) -> int | None:
    for parent in [path.parent, *path.parents]:
        name = parent.name
        if name.isdigit():
            return int(name)
    return None


def _looks_like_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")
