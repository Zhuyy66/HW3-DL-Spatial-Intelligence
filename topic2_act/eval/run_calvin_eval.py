"""CALVIN evaluation launcher for the Day 6 LeRobot ACT bridge."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CALVIN with the HW3 LeRobot ACT bridge.")
    parser.add_argument("--calvin-root", default=None, help="Path to the official CALVIN clone.")
    parser.add_argument("--dataset-path", default=None, help="Path to a downloaded CALVIN dataset split/debug dataset.")
    parser.add_argument("--checkpoint", required=True, help="ACT run dir, checkpoint dir, or model weights file.")
    parser.add_argument("--eval-log-dir", default=None, help="Directory for official CALVIN evaluation logs.")
    parser.add_argument("--cuda-device", type=int, default=0, help="Logical CUDA device exposed to the eval env.")
    parser.add_argument("--num-sequences", type=int, default=1, help="Monkeypatched CALVIN sequence count for smoke.")
    parser.add_argument("--ep-len", type=int, default=2, help="Monkeypatched rollout length for smoke.")
    parser.add_argument("--dry-run-wrapper-only", action="store_true", help="Only instantiate/reset/step the wrapper.")
    parser.add_argument("--debug", action="store_true", help="Pass debug=True into official evaluate_policy.")
    parser.add_argument("--no-load-weights", action="store_true", help="Resolve checkpoint path without reading tensors.")
    parser.add_argument("--worker-python", default=None, help="Python executable for env_hw3_robot / LeRobot worker.")
    parser.add_argument("--worker-device", default=None, help="Inference device used inside the LeRobot worker.")
    parser.add_argument("--worker-log", default=None, help="File for worker stderr/logging.")
    parser.add_argument("--worker-timeout", type=float, default=180.0, help="Seconds to wait for worker protocol replies.")
    parser.add_argument(
        "--single-rollout-smoke",
        action="store_true",
        help="Run one manual rollout episode and log action/movement evidence instead of official eval.",
    )
    parser.add_argument("--rollout-steps", type=int, default=60, help="Step count for --single-rollout-smoke.")
    parser.add_argument(
        "--egl-policy",
        choices=("strict", "fallback0", "direct", "direct-cameras"),
        default="strict",
        help=(
            "EGL handling for CALVIN env creation. strict requires CUDA->EGL mapping; "
            "fallback0 sets EGL_VISIBLE_DEVICES=0 if mapping fails; direct disables "
            "the CALVIN EGL plugin and removes cameras; direct-cameras keeps static "
            "and gripper cameras under PyBullet DIRECT for Day 6 real image inference."
        ),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def add_calvin_paths(calvin_root: str | Path | None) -> Path | None:
    if calvin_root is None:
        return None

    root = Path(calvin_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"CALVIN root does not exist: {root}")
    for rel in ("calvin_models", "calvin_env"):
        candidate = root / rel
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    sys.path.insert(0, str(root))
    return root


def require_existing_dir(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    return resolved


def audit_egl(cuda_device: int, egl_policy: str) -> dict[str, Any]:
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    audit: dict[str, Any] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "EGL_VISIBLE_DEVICES_before": os.environ.get("EGL_VISIBLE_DEVICES"),
        "DISPLAY": os.environ.get("DISPLAY"),
        "LD_LIBRARY_PATH_first_entries": [entry for entry in ld_library_path.split(":") if entry][:8],
        "cuda_device": cuda_device,
        "egl_policy": egl_policy,
        "egl_device": None,
        "egl_mapping_error": None,
        "fallback": None,
    }

    if egl_policy == "direct":
        audit["fallback"] = "skipped CUDA/EGL mapping because direct policy disables CALVIN use_egl"
        audit["EGL_VISIBLE_DEVICES_after"] = os.environ.get("EGL_VISIBLE_DEVICES")
        return audit
    if egl_policy == "direct-cameras":
        audit["fallback"] = "skipped CUDA/EGL mapping because direct-cameras policy disables CALVIN use_egl"
        audit["EGL_VISIBLE_DEVICES_after"] = os.environ.get("EGL_VISIBLE_DEVICES")
        return audit

    try:
        from calvin_env.utils.utils import get_egl_device_id

        egl_device = int(get_egl_device_id(cuda_device))
        audit["egl_device"] = egl_device
        if "EGL_VISIBLE_DEVICES" not in os.environ:
            os.environ["EGL_VISIBLE_DEVICES"] = str(egl_device)
    except Exception as exc:  # noqa: BLE001 - log exact CALVIN/EGL failure.
        audit["egl_mapping_error"] = repr(exc)
        LOGGER.warning("could not resolve EGL device for CUDA device %s: %r", cuda_device, exc)
        if egl_policy == "fallback0":
            os.environ["EGL_VISIBLE_DEVICES"] = "0"
            audit["fallback"] = "set EGL_VISIBLE_DEVICES=0 after mapping failure"

    audit["EGL_VISIBLE_DEVICES_after"] = os.environ.get("EGL_VISIBLE_DEVICES")
    return audit


def clear_cameras_for_direct_policy(render_conf: Any) -> None:
    """Disable camera instantiation for Day 5 state-only direct smoke."""

    render_conf.cameras = {}
    render_conf.env.cameras = {}


def restrict_cameras_for_direct_policy(render_conf: Any) -> None:
    """Keep only static and gripper cameras for Day 6 direct image smoke."""

    cameras: dict[str, Any] = {}
    existing = getattr(render_conf, "cameras", {}) or {}
    for name in ("static", "gripper"):
        if name in existing:
            cameras[name] = existing[name]
        else:
            cameras[name] = _load_calvin_camera_conf(name)
    render_conf.cameras = cameras
    render_conf.env.cameras = cameras


def _load_calvin_camera_conf(name: str) -> Any:
    import calvin_env
    from omegaconf import OmegaConf

    camera_path = Path(calvin_env.__file__).resolve().parents[1] / "conf" / "cameras" / "cameras" / f"{name}.yaml"
    if not camera_path.is_file():
        raise FileNotFoundError(f"CALVIN camera config not found: {camera_path}")
    return OmegaConf.load(camera_path)


def make_direct_env(dataset_path: Path, camera_mode: str = "none") -> Any:
    """Create a CALVIN env with PyBullet DIRECT and use_egl=False."""

    validation_dir = dataset_path / "validation"
    merged_config = validation_dir / ".hydra" / "merged_config.yaml"
    if not validation_dir.is_dir():
        raise FileNotFoundError(f"CALVIN validation directory does not exist: {validation_dir}")
    if not merged_config.is_file():
        raise FileNotFoundError(f"CALVIN merged Hydra config does not exist: {merged_config}")

    import hydra
    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(merged_config)
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    if camera_mode == "none":
        clear_cameras_for_direct_policy(render_conf)
    elif camera_mode == "static_gripper":
        restrict_cameras_for_direct_policy(render_conf)
    else:
        raise ValueError(f"unknown direct camera_mode: {camera_mode!r}")
    LOGGER.info("creating CALVIN env with use_egl=False and camera_mode=%s", camera_mode)
    return hydra.utils.instantiate(
        render_conf.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
        use_egl=False,
    )


def make_env_for_policy(eval_mod: Any, dataset_path: Path, egl_policy: str) -> Any:
    if egl_policy == "direct":
        return make_direct_env(dataset_path, camera_mode="none")
    if egl_policy == "direct-cameras":
        return make_direct_env(dataset_path, camera_mode="static_gripper")
    return eval_mod.make_env(str(dataset_path))


def print_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label}:")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def load_wrapper_class():
    from topic2_act.eval.lerobot_act_wrapper import LeRobotACTWrapper

    return LeRobotACTWrapper


def make_wrapper(args: argparse.Namespace) -> Any:
    LeRobotACTWrapper = load_wrapper_class()
    load_weights = (not args.no_load_weights) and not args.worker_python
    return LeRobotACTWrapper(
        args.checkpoint,
        device=f"cuda:{args.cuda_device}",
        load_weights=load_weights,
        worker_python=args.worker_python,
        worker_device=args.worker_device or f"cuda:{args.cuda_device}",
        worker_log=args.worker_log,
        worker_timeout=args.worker_timeout,
    )


def dry_run_wrapper(args: argparse.Namespace) -> int:
    model = make_wrapper(args)
    print_json("wrapper_checkpoint_summary", model.checkpoint_summary())
    model.reset()
    if args.worker_python:
        print_json(
            "wrapper_dry_run",
            {
                "worker_enabled": True,
                "step_skipped": "worker inference needs a real CALVIN observation",
            },
        )
        model.close()
        return 0
    action = model.step(obs={}, goal="day6 wrapper smoke")
    print_json(
        "wrapper_dry_run",
        {
            "step_count": model.step_count,
            "action_shape": list(action.shape),
            "action_dtype": str(action.dtype),
            "action_sum": float(action.sum()),
        },
    )
    model.close()
    return 0


def extract_tcp_pos(obs: Any, info: dict[str, Any] | None = None) -> list[float] | None:
    """Extract TCP xyz from CALVIN info first, then robot_obs fallback."""

    robot_info = info.get("robot_info") if isinstance(info, dict) else None
    if isinstance(robot_info, dict):
        for key in ("tcp_pos", "tcp_position", "tcp_pos_world"):
            if key in robot_info:
                return np.asarray(robot_info[key], dtype=np.float64).reshape(-1)[:3].tolist()

    robot_obs = None
    if isinstance(obs, dict):
        if "robot_obs" in obs:
            robot_obs = obs["robot_obs"]
        else:
            state_obs = obs.get("state_obs")
            if isinstance(state_obs, dict) and "robot_obs" in state_obs:
                robot_obs = state_obs["robot_obs"]
    if robot_obs is None:
        return None
    flat = np.asarray(robot_obs, dtype=np.float64).reshape(-1)
    if flat.size < 3:
        return None
    return flat[:3].tolist()


def single_rollout_smoke(args: argparse.Namespace) -> int:
    if args.calvin_root is None:
        raise ValueError("--calvin-root is required for --single-rollout-smoke")
    if args.dataset_path is None:
        raise ValueError("--dataset-path is required for --single-rollout-smoke")
    if not args.worker_python:
        raise ValueError("--worker-python is required for --single-rollout-smoke Day 6 real ACT inference")
    if args.rollout_steps <= 0:
        raise ValueError(f"--rollout-steps must be positive, got {args.rollout_steps}")

    calvin_root = add_calvin_paths(args.calvin_root)
    dataset_path = require_existing_dir(args.dataset_path, "CALVIN dataset")
    print_json(
        "calvin_eval_start",
        {
            "mode": "single_rollout_smoke",
            "calvin_root": str(calvin_root) if calvin_root else None,
            "dataset_path": str(dataset_path),
            "checkpoint": args.checkpoint,
            "eval_log_dir": args.eval_log_dir,
            "rollout_steps": args.rollout_steps,
            "egl_policy": args.egl_policy,
            "direct_camera_mode": (
                "none" if args.egl_policy == "direct" else "static_gripper" if args.egl_policy == "direct-cameras" else None
            ),
            "worker_python": args.worker_python,
            "worker_device": args.worker_device or f"cuda:{args.cuda_device}",
            "worker_log": args.worker_log,
        },
    )

    egl_audit = audit_egl(args.cuda_device, args.egl_policy)
    print_json("egl_audit", egl_audit)
    if args.egl_policy == "strict" and egl_audit.get("egl_mapping_error"):
        raise RuntimeError(f"EGL device mapping failed: {egl_audit['egl_mapping_error']}")

    eval_mod = None
    if args.egl_policy not in ("direct", "direct-cameras"):
        eval_mod = importlib.import_module("calvin_agent.evaluation.evaluate_policy")
    model = make_wrapper(args)
    print_json("wrapper_checkpoint_summary", model.checkpoint_summary())

    env = None
    action_norms: list[float] = []
    gripper_values: list[float] = []
    action_shapes: list[list[int]] = []
    tcp_trace: list[list[float]] = []
    action_preview: list[dict[str, Any]] = []
    try:
        env = make_env_for_policy(eval_mod, dataset_path, args.egl_policy)
        obs = env.reset()
        model.reset()
        start_tcp = extract_tcp_pos(obs)
        if start_tcp is not None:
            tcp_trace.append(start_tcp)

        for step_idx in range(args.rollout_steps):
            action = np.asarray(model.step(obs, goal="day6 connectivity smoke"), dtype=np.float32).reshape(-1)
            action_shapes.append(list(action.shape))
            action_norms.append(float(np.linalg.norm(action[: min(6, action.size)])))
            gripper_values.append(float(action[-1]) if action.size else float("nan"))
            if step_idx < 5 or step_idx == args.rollout_steps - 1:
                action_preview.append(
                    {
                        "step": step_idx,
                        "shape": list(action.shape),
                        "norm_first6": action_norms[-1],
                        "gripper": gripper_values[-1],
                        "action": action.astype(float).tolist(),
                    }
                )
            obs, _reward, _done, info = env.step(action)
            tcp_pos = extract_tcp_pos(obs, info)
            if tcp_pos is not None:
                tcp_trace.append(tcp_pos)

        start = np.asarray(tcp_trace[0], dtype=np.float64) if tcp_trace else None
        tcp_deltas = [float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - start)) for pos in tcp_trace] if start is not None else []
        max_tcp_delta = max(tcp_deltas) if tcp_deltas else None
        movement_threshold = 1e-4
        moved = bool(max_tcp_delta is not None and max_tcp_delta > movement_threshold)
        result = {
            "steps": args.rollout_steps,
            "wrapper_step_count": model.step_count,
            "action_shapes": action_shapes[:5],
            "last_action_shape": action_shapes[-1] if action_shapes else None,
            "max_action_norm_first6": max(action_norms) if action_norms else None,
            "min_action_norm_first6": min(action_norms) if action_norms else None,
            "gripper_values": sorted(set(gripper_values)),
            "start_tcp_pos": tcp_trace[0] if tcp_trace else None,
            "end_tcp_pos": tcp_trace[-1] if tcp_trace else None,
            "tcp_trace_count": len(tcp_trace),
            "max_tcp_delta": max_tcp_delta,
            "movement_threshold": movement_threshold,
            "moved": moved,
            "action_preview": action_preview,
        }
        print_json("single_rollout_smoke_result", result)
        if not moved:
            raise RuntimeError(f"single rollout did not exceed movement threshold: max_tcp_delta={max_tcp_delta}")
    finally:
        model.close()
        if env is not None and hasattr(env, "close"):
            env.close()
    return 0


def run_official_eval(args: argparse.Namespace) -> int:
    if args.calvin_root is None:
        raise ValueError("--calvin-root is required unless --dry-run-wrapper-only is used")
    if args.dataset_path is None:
        raise ValueError("--dataset-path is required unless --dry-run-wrapper-only is used")

    calvin_root = add_calvin_paths(args.calvin_root)
    dataset_path = require_existing_dir(args.dataset_path, "CALVIN dataset")
    print_json(
        "calvin_eval_start",
        {
            "calvin_root": str(calvin_root) if calvin_root else None,
            "dataset_path": str(dataset_path),
            "checkpoint": args.checkpoint,
            "eval_log_dir": args.eval_log_dir,
            "num_sequences": args.num_sequences,
            "ep_len": args.ep_len,
            "egl_policy": args.egl_policy,
            "direct_camera_mode": (
                "none" if args.egl_policy == "direct" else "static_gripper" if args.egl_policy == "direct-cameras" else None
            ),
        },
    )

    egl_audit = audit_egl(args.cuda_device, args.egl_policy)
    print_json("egl_audit", egl_audit)
    if args.egl_policy == "strict" and egl_audit.get("egl_mapping_error"):
        raise RuntimeError(f"EGL device mapping failed: {egl_audit['egl_mapping_error']}")

    eval_mod = importlib.import_module("calvin_agent.evaluation.evaluate_policy")
    eval_mod.NUM_SEQUENCES = args.num_sequences
    eval_mod.EP_LEN = args.ep_len

    model = make_wrapper(args)
    print_json("wrapper_checkpoint_summary", model.checkpoint_summary())

    env = None
    try:
        env = make_env_for_policy(eval_mod, dataset_path, args.egl_policy)
        results = eval_mod.evaluate_policy(
            model,
            env,
            epoch="hw3_day6_bridge",
            eval_log_dir=args.eval_log_dir,
            debug=args.debug,
            create_plan_tsne=False,
        )
        print_json(
            "calvin_eval_result",
            {
                "result_count": len(results) if hasattr(results, "__len__") else None,
                "results": list(results) if isinstance(results, (list, tuple)) else str(results),
                "wrapper_step_count": model.step_count,
            },
        )
    finally:
        model.close()
        if env is not None and hasattr(env, "close"):
            env.close()
    return 0


def main() -> int:
    configure_logging()
    args = parse_args()
    add_calvin_paths(args.calvin_root)

    if args.dry_run_wrapper_only:
        return dry_run_wrapper(args)
    if args.single_rollout_smoke:
        return single_rollout_smoke(args)
    return run_official_eval(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI should leave concise log evidence.
        LOGGER.exception("CALVIN eval launcher failed")
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
