"""Evaluate ACT open-loop action L1 on raw CALVIN split parquet files.

The official course splitD is stored in the LeRobot v2.1 schema:
``image``, ``wrist_image``, ``state`` and ``actions``.  This evaluator streams
those parquet episodes directly, builds the same 100-step action chunks used by
ACT training, and computes L1 against the checkpoint prediction without
materializing another canonical dataset copy.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


STATIC_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_PAD_KEY = "action_is_pad"
RAW_COLUMNS = ("image", "wrist_image", "state", "actions")


@dataclass
class L1Accumulator:
    """Streaming L1 aggregates over valid action chunk positions."""

    chunk_size: int
    action_dim: int

    def __post_init__(self) -> None:
        self.valid_abs_sum = 0.0
        self.valid_count = 0
        self.forward_abs_sum = 0.0
        self.forward_count = 0
        self.per_dim_sum = np.zeros(self.action_dim, dtype=np.float64)
        self.per_dim_count = np.zeros(self.action_dim, dtype=np.int64)
        self.per_chunk_sum = np.zeros(self.chunk_size, dtype=np.float64)
        self.per_chunk_count = np.zeros(self.chunk_size, dtype=np.int64)
        self.raw_valid_abs_sum = 0.0
        self.raw_valid_count = 0
        self.raw_per_dim_sum = np.zeros(self.action_dim, dtype=np.float64)
        self.raw_per_dim_count = np.zeros(self.action_dim, dtype=np.int64)
        self.raw_per_chunk_sum = np.zeros(self.chunk_size, dtype=np.float64)
        self.raw_per_chunk_count = np.zeros(self.chunk_size, dtype=np.int64)

    def update(
        self,
        abs_error: np.ndarray,
        action_is_pad: np.ndarray,
        raw_abs_error: np.ndarray | None = None,
    ) -> None:
        """Add one batch of ``[batch, chunk, action_dim]`` errors."""

        err = np.asarray(abs_error, dtype=np.float64)
        pad = np.asarray(action_is_pad, dtype=bool)
        if err.ndim != 3:
            raise ValueError(f"abs_error must be 3D, got shape {err.shape}")
        if pad.shape != err.shape[:2]:
            raise ValueError(f"action_is_pad shape {pad.shape} does not match error shape {err.shape[:2]}")
        if err.shape[1] != self.chunk_size or err.shape[2] != self.action_dim:
            raise ValueError(
                f"error shape {err.shape} does not match chunk_size={self.chunk_size}, "
                f"action_dim={self.action_dim}"
            )

        valid = (~pad)[..., None]
        expanded_valid = np.broadcast_to(valid, err.shape)
        valid_err = np.where(expanded_valid, err, 0.0)

        self.valid_abs_sum += float(valid_err.sum())
        self.valid_count += int(expanded_valid.sum())
        self.forward_abs_sum += float(valid_err.sum())
        self.forward_count += int(err.size)
        self.per_dim_sum += valid_err.sum(axis=(0, 1))
        self.per_dim_count += expanded_valid.sum(axis=(0, 1)).astype(np.int64)
        self.per_chunk_sum += valid_err.sum(axis=(0, 2))
        self.per_chunk_count += expanded_valid.sum(axis=(0, 2)).astype(np.int64)

        if raw_abs_error is None:
            return
        raw_err = np.asarray(raw_abs_error, dtype=np.float64)
        if raw_err.shape != err.shape:
            raise ValueError(f"raw_abs_error shape {raw_err.shape} does not match error shape {err.shape}")
        raw_valid_err = np.where(expanded_valid, raw_err, 0.0)
        self.raw_valid_abs_sum += float(raw_valid_err.sum())
        self.raw_valid_count += int(expanded_valid.sum())
        self.raw_per_dim_sum += raw_valid_err.sum(axis=(0, 1))
        self.raw_per_dim_count += expanded_valid.sum(axis=(0, 1)).astype(np.int64)
        self.raw_per_chunk_sum += raw_valid_err.sum(axis=(0, 2))
        self.raw_per_chunk_count += expanded_valid.sum(axis=(0, 2)).astype(np.int64)

    @staticmethod
    def _means(sums: np.ndarray, counts: np.ndarray) -> list[float | None]:
        values: list[float | None] = []
        for total, count in zip(sums.tolist(), counts.tolist(), strict=True):
            values.append(float(total / count) if count else None)
        return values

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action_l1_valid_mean": safe_div(self.valid_abs_sum, self.valid_count),
            "forward_l1_loss_equivalent": safe_div(self.forward_abs_sum, self.forward_count),
            "valid_action_element_count": self.valid_count,
            "forward_action_element_count": self.forward_count,
            "per_dim_l1": self._means(self.per_dim_sum, self.per_dim_count),
            "per_dim_counts": self.per_dim_count.tolist(),
            "per_chunk_l1": self._means(self.per_chunk_sum, self.per_chunk_count),
            "per_chunk_counts": self.per_chunk_count.tolist(),
            "metric_space": "normalized_action",
        }
        if self.raw_valid_count:
            payload.update(
                {
                    "raw_action_l1_valid_mean": safe_div(self.raw_valid_abs_sum, self.raw_valid_count),
                    "raw_per_dim_l1": self._means(self.raw_per_dim_sum, self.raw_per_dim_count),
                    "raw_per_dim_counts": self.raw_per_dim_count.tolist(),
                    "raw_per_chunk_l1": self._means(self.raw_per_chunk_sum, self.raw_per_chunk_count),
                    "raw_per_chunk_counts": self.raw_per_chunk_count.tolist(),
                    "raw_metric_space": "dataset_action_units",
                }
            )
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ACT open-loop Action L1 on raw CALVIN split data.")
    parser.add_argument("--dataset-root", required=True, help="Raw v2.1 split root, e.g. .../splitD.")
    parser.add_argument("--checkpoint", required=True, help="LeRobot pretrained_model checkpoint directory.")
    parser.add_argument("--output", required=True, help="Path to write JSON summary.")
    parser.add_argument("--device", default="cuda:0", help="Inference device passed to LeRobot.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_div(total: float, count: int) -> float | None:
    return float(total / count) if count else None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_image_hwc_uint8(value: Any) -> np.ndarray:
    """Decode a LeRobot image cell into contiguous HWC uint8."""

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            from PIL import Image

            with Image.open(io.BytesIO(value["bytes"])) as image:
                return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        if value.get("path"):
            from PIL import Image

            with Image.open(value["path"]) as image:
                return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"image must be 3D, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"image must have 3 channels after conversion, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.array(arr, dtype=np.uint8, copy=True, order="C")


def image_to_chw_float_tensor(value: Any):
    import torch

    arr = load_image_hwc_uint8(value)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
    return tensor / 255.0


def build_action_chunk(actions: np.ndarray, frame_index: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Match LeRobot future-action chunking with end-of-episode padding."""

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2:
        raise ValueError(f"actions must be 2D, got shape {action_array.shape}")
    if frame_index < 0 or frame_index >= action_array.shape[0]:
        raise IndexError(f"frame_index {frame_index} out of range for episode length {action_array.shape[0]}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    chunk = np.empty((chunk_size, action_array.shape[1]), dtype=np.float32)
    is_pad = np.zeros(chunk_size, dtype=bool)
    last_index = action_array.shape[0] - 1
    for offset in range(chunk_size):
        src_index = frame_index + offset
        if src_index > last_index:
            src_index = last_index
            is_pad[offset] = True
        chunk[offset] = action_array[src_index]
    return chunk, is_pad


def make_frame_sample(row: Any, actions: np.ndarray, local_index: int, chunk_size: int) -> dict[str, Any]:
    import torch

    chunk, is_pad = build_action_chunk(actions, local_index, chunk_size)
    return {
        STATIC_KEY: image_to_chw_float_tensor(row["image"]),
        WRIST_KEY: image_to_chw_float_tensor(row["wrist_image"]),
        STATE_KEY: torch.as_tensor(np.array(row["state"], dtype=np.float32, copy=True).reshape(-1), dtype=torch.float32),
        ACTION_KEY: torch.as_tensor(chunk, dtype=torch.float32),
        ACTION_PAD_KEY: torch.as_tensor(is_pad, dtype=torch.bool),
    }


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate

    return default_collate(samples)


def episode_parquet_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    template = str(info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    episode_chunk = episode_index // chunks_size
    return dataset_root / template.format(episode_chunk=episode_chunk, episode_index=episode_index)


def load_policy(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, ACTConfig):
        raise TypeError(f"checkpoint config must be ACTConfig, got {type(config).__name__}")
    config.device = device
    policy = ACTPolicy.from_pretrained(checkpoint, config=config)
    policy.eval()
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor


def predict_action_chunk(policy: Any, batch: dict[str, Any]):
    from lerobot.policies.act import modeling_act

    model_batch = dict(batch)
    image_key = getattr(modeling_act, "OBS_IMAGES", "observation.images")
    action_key = getattr(modeling_act, "ACTION", ACTION_KEY)
    if policy.config.image_features:
        model_batch[image_key] = [model_batch[key] for key in policy.config.image_features]
    actions_hat, _latent = policy.model(model_batch)
    return actions_hat, model_batch[action_key]


def load_action_normalizer(checkpoint: Path):
    try:
        import torch
        from safetensors.torch import load_file

        state_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        state = load_file(str(state_path), device="cpu")
        mean = state.get("action.mean")
        std = state.get("action.std")
        if mean is None or std is None:
            return None
        return mean.to(dtype=torch.float32), std.to(dtype=torch.float32)
    except Exception:
        return None


def unnormalize_action(normalizer: Any, tensor: Any):
    if normalizer is None:
        return None
    mean, std = normalizer
    return tensor.detach().cpu() * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def validate_dataset_root(dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    info = read_json(dataset_root / "meta" / "info.json")
    missing = [column for column in RAW_COLUMNS if column not in info.get("features", {})]
    if missing:
        raise ValueError(f"dataset features missing required raw columns: {missing}")
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    return info, episodes


def summarize_dataset(info: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "codebase_version": info.get("codebase_version"),
        "scene": info.get("scene"),
        "source_dataset": info.get("source_dataset"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "episode_metadata_rows": len(episodes),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd
    import torch

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive when provided")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided")

    started = time.monotonic()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    info, episodes = validate_dataset_root(dataset_root)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    policy, preprocessor = load_policy(checkpoint, args.device)
    chunk_size = int(getattr(policy.config, "chunk_size", 100))
    output_features = getattr(policy.config, "output_features", {})
    action_feature = output_features.get(ACTION_KEY)
    action_dim = int(getattr(action_feature, "shape", (7,))[0])
    accumulator = L1Accumulator(chunk_size=chunk_size, action_dim=action_dim)
    normalizer = load_action_normalizer(checkpoint)

    processed_episodes = 0
    processed_frames = 0
    batch: list[dict[str, Any]] = []
    missing_parquets: list[str] = []

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        collated = collate_samples(batch)
        with torch.inference_mode():
            processed = preprocessor(collated)
            pred_norm, target_norm = predict_action_chunk(policy, processed)
            pad = processed[ACTION_PAD_KEY].detach().cpu().numpy().astype(bool)
            err = (target_norm - pred_norm).abs().detach().cpu().numpy()
            pred_raw = unnormalize_action(normalizer, pred_norm)
            target_raw = unnormalize_action(normalizer, target_norm)
            raw_err = None
            if pred_raw is not None and target_raw is not None:
                raw_err = (target_raw - pred_raw).abs().numpy()
        accumulator.update(err, pad, raw_abs_error=raw_err)
        batch = []

    for episode_pos, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        parquet_path = episode_parquet_path(dataset_root, info, episode_index)
        if not parquet_path.is_file():
            missing_parquets.append(str(parquet_path))
            continue
        frame = pd.read_parquet(parquet_path)
        actions = np.stack(frame["actions"].to_numpy()).astype(np.float32)
        for local_index, (_row_index, row) in enumerate(frame.iterrows()):
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break
            batch.append(make_frame_sample(row, actions, local_index, chunk_size))
            processed_frames += 1
            if len(batch) >= args.batch_size:
                flush_batch()
        processed_episodes += 1
        if args.log_every and (processed_episodes == 1 or processed_episodes % args.log_every == 0):
            print(
                json.dumps(
                    {
                        "event": "open_loop_action_l1_progress",
                        "processed_episodes": processed_episodes,
                        "processed_frames": processed_frames,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.max_frames is not None and processed_frames >= args.max_frames:
            break

    flush_batch()
    elapsed = time.monotonic() - started
    metric_summary = accumulator.summary()
    finite_metric = metric_summary["action_l1_valid_mean"]
    payload = {
        "status": "completed" if finite_metric is not None and math.isfinite(float(finite_metric)) else "failed",
        "generated_at": utc_now(),
        "elapsed_seconds": elapsed,
        "dataset_root": str(dataset_root),
        "dataset": summarize_dataset(info, episodes),
        "checkpoint": str(checkpoint),
        "device": args.device,
        "batch_size": args.batch_size,
        "chunk_size": chunk_size,
        "action_dim": action_dim,
        "episode_count": processed_episodes,
        "frame_count": processed_frames,
        "missing_parquet_count": len(missing_parquets),
        "missing_parquets_sample": missing_parquets[:10],
        "max_episodes": args.max_episodes,
        "max_frames": args.max_frames,
        "normalizer_loaded_for_raw_metrics": normalizer is not None,
        **metric_summary,
    }
    return payload


def main() -> int:
    args = parse_args()
    payload = evaluate(args)
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if payload["status"] != "completed":
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - leave concise evidence in tee logs.
        print(f"error: {exc}", file=sys.stderr)
        raise
