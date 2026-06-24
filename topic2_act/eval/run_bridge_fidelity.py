"""Compare direct LeRobot ACT inference with the subprocess bridge.

This script is intended to run inside ``env_hw3_robot``.  It loads one real
LeRobot training-set observation, runs the worker core directly in the current
process, then runs the same observation through the length-prefixed subprocess
bridge.  Day 6 acceptance requires the formatted CALVIN action to match exactly.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from topic2_act.eval.lerobot_act_worker import GRIPPER_KEY, STATE_KEY, STATIC_KEY, LeRobotACTWorkerCore
from topic2_act.eval.lerobot_act_wrapper import LeRobotACTWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 6 ACT bridge fidelity check.")
    parser.add_argument("--checkpoint", required=True, help="LeRobot pretrained_model checkpoint directory.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobot dataset root.")
    parser.add_argument("--worker-python", required=True, help="Python executable used for the subprocess bridge.")
    parser.add_argument("--device", default="cuda:0", help="Inference device for direct and bridge workers.")
    parser.add_argument("--repo-id", default=None, help="LeRobot repo_id; inferred from meta/info.json when omitted.")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset frame index to compare.")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--worker-timeout", type=float, default=180.0)
    parser.add_argument("--atol", type=float, default=0.0, help="Allowed absolute tolerance; 0 requires exact match.")
    return parser.parse_args()


def print_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label}:")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def infer_repo_id(dataset_root: Path, override: str | None) -> str:
    if override:
        return override
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            repo_id = info.get("repo_id") or info.get("dataset_repo_id")
            if repo_id:
                return str(repo_id)
        except Exception:
            pass
    return "hw3/calvin-splitA-canonical"


def load_training_observation(dataset_root: Path, repo_id: str, sample_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample index {sample_index} out of range for dataset length {len(dataset)}")
    item = dataset[sample_index]
    obs = {
        "rgb_obs": {
            "rgb_static": _to_hwc_uint8(item[STATIC_KEY]),
            "rgb_gripper": _to_hwc_uint8(item[GRIPPER_KEY]),
        },
        "robot_obs": _to_numpy(item[STATE_KEY]).astype(np.float32).reshape(-1),
    }
    meta = {
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "sample_index": sample_index,
        "dataset_len": len(dataset),
        "static_shape": list(obs["rgb_obs"]["rgb_static"].shape),
        "gripper_shape": list(obs["rgb_obs"]["rgb_gripper"].shape),
        "state_shape": list(obs["robot_obs"].shape),
    }
    return obs, meta


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _to_hwc_uint8(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"image sample must be 3D or batch-1 4D, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def run_direct(checkpoint: str, device: str, action_dim: int, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    core = LeRobotACTWorkerCore(checkpoint, device=device, action_dim=action_dim)
    core.reset()
    action, meta = core.step(obs, goal="day6 bridge fidelity")
    del core
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return action, meta


def run_bridge(args: argparse.Namespace, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    wrapper = LeRobotACTWrapper(
        args.checkpoint,
        device=args.device,
        action_dim=args.action_dim,
        load_weights=False,
        worker_python=args.worker_python,
        worker_device=args.device,
        worker_timeout=args.worker_timeout,
    )
    try:
        wrapper.reset()
        action = wrapper.step(obs, goal="day6 bridge fidelity")
        return action, wrapper.checkpoint_summary()
    finally:
        wrapper.close()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    repo_id = infer_repo_id(dataset_root, args.repo_id)
    obs, sample_meta = load_training_observation(dataset_root, repo_id, args.sample_index)
    print_json("bridge_fidelity_sample", sample_meta)

    direct_action, direct_meta = run_direct(args.checkpoint, args.device, args.action_dim, obs)
    bridge_action, bridge_summary = run_bridge(args, obs)
    diff = np.abs(direct_action.astype(np.float32) - bridge_action.astype(np.float32))
    exact = bool(np.array_equal(direct_action, bridge_action))
    allclose = bool(np.allclose(direct_action, bridge_action, atol=args.atol, rtol=0.0))
    result = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "device": args.device,
        "action_dim": args.action_dim,
        "exact": exact,
        "atol": args.atol,
        "allclose": allclose,
        "max_abs_diff": float(diff.max()) if diff.size else None,
        "direct_action": direct_action.astype(float).tolist(),
        "bridge_action": bridge_action.astype(float).tolist(),
        "direct_meta": direct_meta,
        "bridge_summary": bridge_summary,
    }
    print_json("bridge_fidelity_result", result)
    if args.atol == 0.0 and not exact:
        raise RuntimeError(f"direct-vs-bridge action mismatch: max_abs_diff={result['max_abs_diff']}")
    if args.atol != 0.0 and not allclose:
        raise RuntimeError(f"direct-vs-bridge action mismatch above atol={args.atol}: {result['max_abs_diff']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - leave concise failure evidence in tee log.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
