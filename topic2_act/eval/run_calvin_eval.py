"""Thin CALVIN evaluation launcher for the Day 5 LeRobot ACT wrapper."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CALVIN with the HW3 LeRobot ACT wrapper skeleton.")
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
    parser.add_argument(
        "--egl-policy",
        choices=("strict", "fallback0", "direct"),
        default="strict",
        help=(
            "EGL handling for CALVIN env creation. strict requires CUDA->EGL mapping; "
            "fallback0 sets EGL_VISIBLE_DEVICES=0 if mapping fails; direct disables "
            "the CALVIN EGL plugin and uses PyBullet DIRECT/TinyRenderer for Day 5 smoke."
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


def make_direct_env(dataset_path: Path) -> Any:
    """Create a state-only CALVIN env with PyBullet DIRECT and use_egl=False."""

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
    clear_cameras_for_direct_policy(render_conf)
    LOGGER.info("creating CALVIN env with use_egl=False and cameras=none for Day 5 direct smoke")
    return hydra.utils.instantiate(
        render_conf.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
        use_egl=False,
    )


def make_env_for_policy(eval_mod: Any, dataset_path: Path, egl_policy: str) -> Any:
    if egl_policy == "direct":
        return make_direct_env(dataset_path)
    return eval_mod.make_env(str(dataset_path))


def print_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label}:")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def load_wrapper_class():
    from topic2_act.eval.lerobot_act_wrapper import LeRobotACTWrapper

    return LeRobotACTWrapper


def dry_run_wrapper(args: argparse.Namespace) -> int:
    LeRobotACTWrapper = load_wrapper_class()
    model = LeRobotACTWrapper(
        args.checkpoint,
        device=f"cuda:{args.cuda_device}",
        load_weights=not args.no_load_weights,
    )
    print_json("wrapper_checkpoint_summary", model.checkpoint_summary())
    model.reset()
    action = model.step(obs={}, goal="day5 wrapper smoke")
    print_json(
        "wrapper_dry_run",
        {
            "step_count": model.step_count,
            "action_shape": list(action.shape),
            "action_dtype": str(action.dtype),
            "action_sum": float(action.sum()),
        },
    )
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
            "direct_camera_mode": "none" if args.egl_policy == "direct" else None,
        },
    )

    egl_audit = audit_egl(args.cuda_device, args.egl_policy)
    print_json("egl_audit", egl_audit)
    if args.egl_policy == "strict" and egl_audit.get("egl_mapping_error"):
        raise RuntimeError(f"EGL device mapping failed: {egl_audit['egl_mapping_error']}")

    eval_mod = importlib.import_module("calvin_agent.evaluation.evaluate_policy")
    eval_mod.NUM_SEQUENCES = args.num_sequences
    eval_mod.EP_LEN = args.ep_len

    LeRobotACTWrapper = load_wrapper_class()
    model = LeRobotACTWrapper(
        args.checkpoint,
        device=f"cuda:{args.cuda_device}",
        load_weights=not args.no_load_weights,
    )
    print_json("wrapper_checkpoint_summary", model.checkpoint_summary())

    env = make_env_for_policy(eval_mod, dataset_path, args.egl_policy)
    results = eval_mod.evaluate_policy(
        model,
        env,
        epoch="hw3_day5_smoke",
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
    return 0


def main() -> int:
    configure_logging()
    args = parse_args()
    add_calvin_paths(args.calvin_root)

    if args.dry_run_wrapper_only:
        return dry_run_wrapper(args)
    return run_official_eval(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI should leave concise log evidence.
        LOGGER.exception("CALVIN eval launcher failed")
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
