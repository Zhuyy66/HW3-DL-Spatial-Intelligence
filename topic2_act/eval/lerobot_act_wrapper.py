"""LeRobot ACT wrapper for CALVIN evaluation.

The CALVIN evaluation environment is Python 3.8, while LeRobot 0.4.0 runs in the
robot training environment.  This wrapper keeps CALVIN's ``reset`` / ``step``
interface and delegates real ACT inference to one persistent worker subprocess.
Without a worker it preserves the Day 5 zero-action smoke-test behavior.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from typing import Any, BinaryIO

import numpy as np

try:  # CALVIN is only installed in env_hw3_calvin_eval on the server.
    from calvin_agent.models.calvin_base_model import CalvinBaseModel
except Exception:  # noqa: BLE001 - keep local static tests independent.
    CalvinBaseModel = object  # type: ignore[assignment,misc]


LOGGER = logging.getLogger(__name__)
SUPPORTED_WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin")
DEFAULT_WORKER_TIMEOUT_SECONDS = 180.0


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


class LeRobotACTWorkerClient:
    """Length-prefixed subprocess client for the LeRobot ACT worker."""

    def __init__(
        self,
        worker_python: str | Path,
        checkpoint_dir: str | Path,
        worker_device: str,
        action_dim: int,
        worker_log: str | Path | None = None,
        timeout: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    ) -> None:
        self.worker_python = str(worker_python)
        self.checkpoint_dir = str(checkpoint_dir)
        self.worker_device = str(worker_device)
        self.action_dim = int(action_dim)
        self.worker_log = str(worker_log) if worker_log else None
        self.timeout = float(timeout)
        self._log_handle: BinaryIO | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.ready: dict[str, Any] | None = None

        self._start()

    def _start(self) -> None:
        from topic2_act.eval.bridge_protocol import read_frame

        worker_script = Path(__file__).with_name("lerobot_act_worker.py").resolve()
        command = [
            self.worker_python,
            str(worker_script),
            "--checkpoint",
            self.checkpoint_dir,
            "--device",
            self.worker_device,
            "--action-dim",
            str(self.action_dim),
        ]
        if self.worker_log:
            command.extend(["--log-file", self.worker_log])
            Path(self.worker_log).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.worker_log, "ab")  # noqa: SIM115 - closed in close().

        LOGGER.info("starting LeRobot ACT worker: %s", command)
        stderr_target = self._log_handle if self._log_handle else None
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            bufsize=0,
        )
        self.ready = self._read_message_with_timeout(read_frame, self.timeout)
        if self.ready.get("type") != "ready":
            raise RuntimeError(f"worker did not send ready message: {self.ready}")
        LOGGER.info("LeRobot ACT worker ready: %s", self.ready)

    def reset(self) -> dict[str, Any]:
        self._write({"type": "reset"})
        response = self._read()
        if response.get("type") != "reset_ok":
            self._raise_for_response(response, expected="reset_ok")
        return response

    def step(self, obs: Any, goal: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._write({"type": "step", "obs": obs, "goal": goal})
        response = self._read()
        if response.get("type") != "action":
            self._raise_for_response(response, expected="action")
        action = np.asarray(response.get("action"), dtype=np.float32).reshape(-1)
        meta = response.get("meta")
        return action, meta if isinstance(meta, dict) else {}

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self._write({"type": "close"})
                    self._read()
                except Exception as exc:  # noqa: BLE001 - shutdown should not hide main result.
                    LOGGER.warning("worker close handshake failed: %r", exc)
                process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        finally:
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
            if self._log_handle:
                self._log_handle.close()
            self.process = None

    def summary(self) -> dict[str, Any]:
        process = self.process
        return {
            "worker_python": self.worker_python,
            "checkpoint_dir": self.checkpoint_dir,
            "worker_device": self.worker_device,
            "worker_log": self.worker_log,
            "pid": process.pid if process else None,
            "returncode": process.poll() if process else None,
            "ready": self.ready,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        from topic2_act.eval.bridge_protocol import write_frame

        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError("worker stdin is not available")
        write_frame(process.stdin, payload)

    def _read(self) -> dict[str, Any]:
        from topic2_act.eval.bridge_protocol import read_frame

        return self._read_message_with_timeout(read_frame, self.timeout)

    def _read_message_with_timeout(self, reader: Any, timeout: float) -> dict[str, Any]:
        process = self._require_process()
        if process.stdout is None:
            raise RuntimeError("worker stdout is not available")

        queue: Queue[tuple[str, Any]] = Queue(maxsize=1)

        def target() -> None:
            try:
                queue.put(("ok", reader(process.stdout)))
            except Exception as exc:  # noqa: BLE001 - return exact protocol failure.
                queue.put(("error", exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            process.kill()
            raise TimeoutError(f"timed out waiting {timeout:.1f}s for LeRobot ACT worker response")
        kind, value = queue.get()
        if kind == "error":
            if process.poll() is not None:
                raise RuntimeError(
                    f"LeRobot ACT worker exited with code {process.returncode}; "
                    f"worker_log={self.worker_log}"
                ) from value
            raise value
        return value

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self.process is None:
            raise RuntimeError("LeRobot ACT worker is closed")
        if self.process.poll() is not None:
            raise RuntimeError(
                f"LeRobot ACT worker already exited with code {self.process.returncode}; "
                f"worker_log={self.worker_log}"
            )
        return self.process

    def _raise_for_response(self, response: dict[str, Any], expected: str) -> None:
        if response.get("type") == "error":
            traceback_text = response.get("traceback")
            message = response.get("message") or response.get("repr") or response
            raise RuntimeError(f"worker error while waiting for {expected}: {message}\n{traceback_text}")
        raise RuntimeError(f"expected worker response {expected!r}, got {response}")


class LeRobotACTWrapper(CalvinBaseModel):  # type: ignore[misc,valid-type]
    """CALVIN policy adapter for a LeRobot ACT checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        action_dim: int = 7,
        strict_checkpoint: bool = True,
        load_weights: bool = True,
        worker_python: str | Path | None = None,
        worker_device: str | None = None,
        worker_log: str | Path | None = None,
        worker_timeout: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
        worker_transport: Any | None = None,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.device = str(device)
        self.action_dim = int(action_dim)
        self.strict_checkpoint = bool(strict_checkpoint)
        self.load_weights = bool(load_weights)
        self.worker_python = str(worker_python) if worker_python else None
        self.worker_device = str(worker_device or device)
        self.worker_log = str(worker_log) if worker_log else None
        self.worker_timeout = float(worker_timeout)
        self.worker_transport = worker_transport
        self.step_count = 0
        self.action_chunk: np.ndarray | None = None

        self.checkpoint_info = self._resolve_checkpoint(Path(checkpoint))
        if self.load_weights and not self._worker_requested:
            self._load_checkpoint_weights()
        if self._worker_requested:
            self._start_worker_if_needed()

        LOGGER.info("initialized LeRobotACTWrapper: %s", self.checkpoint_summary())

    def reset(self) -> None:
        """Reset per-rollout state before a CALVIN subtask starts."""

        self.step_count = 0
        self.action_chunk = None
        if self.worker_transport is not None:
            self.worker_transport.reset()
        LOGGER.info("LeRobotACTWrapper.reset called")

    def step(self, obs: Any, goal: Any) -> np.ndarray:
        """Return a CALVIN-compatible action."""

        self.step_count += 1
        if self.worker_transport is not None:
            action, meta = self.worker_transport.step(obs, goal)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size != self.action_dim:
                raise ValueError(f"worker action has {action.size} dims, expected {self.action_dim}")
            if self.step_count <= 5 or self.step_count % 20 == 0:
                LOGGER.info(
                    "LeRobotACTWrapper.worker_step step_count=%s action_shape=%s meta=%s",
                    self.step_count,
                    action.shape,
                    meta,
                )
            return action

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

    def close(self) -> None:
        """Close the persistent worker if this wrapper owns one."""

        if self.worker_transport is not None and hasattr(self.worker_transport, "close"):
            self.worker_transport.close()
        self.worker_transport = None

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
                "worker_enabled": self.worker_transport is not None,
                "worker_python": self.worker_python,
                "worker_device": self.worker_device,
                "worker_log": self.worker_log,
                "worker_timeout": self.worker_timeout,
            }
        )
        if self.worker_transport is not None and hasattr(self.worker_transport, "summary"):
            payload["worker_summary"] = self.worker_transport.summary()
        return payload

    @property
    def _worker_requested(self) -> bool:
        return self.worker_transport is not None or self.worker_python is not None

    def _start_worker_if_needed(self) -> None:
        if self.worker_transport is not None:
            return
        if self.worker_python is None:
            return
        checkpoint_dir = self.checkpoint_info.checkpoint_dir or self.checkpoint_info.input_path
        if checkpoint_dir is None:
            raise FileNotFoundError("worker requested but no checkpoint directory was resolved")
        self.worker_transport = LeRobotACTWorkerClient(
            self.worker_python,
            checkpoint_dir,
            worker_device=self.worker_device,
            action_dim=self.action_dim,
            worker_log=self.worker_log,
            timeout=self.worker_timeout,
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - destructors must not raise.
            pass

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
