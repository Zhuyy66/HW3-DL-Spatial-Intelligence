"""Smoke-test the Day 1 LeRobot ACT environment.

The script verifies imports, FFmpeg availability, and optionally CUDA runtime
availability. Use ``--require-cuda`` for the GPU-ready Day 1 gate.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys


ACT_IMPORT_CANDIDATES = (
    "lerobot.policies.act.modeling_act",
    "lerobot.common.policies.act.modeling_act",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if PyTorch cannot initialize CUDA.",
    )
    return parser.parse_args()


def import_required_module(name: str):
    module = importlib.import_module(name)
    print(f"ok: imported {name}")
    return module


def import_act_policy() -> None:
    failures: list[str] = []
    for module_name in ACT_IMPORT_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            getattr(module, "ACTPolicy")
            print(f"ok: ACTPolicy imported from {module_name}")
            return
        except Exception as exc:  # noqa: BLE001 - smoke test should show all failures.
            failures.append(f"{module_name}: {exc!r}")

    joined = "\n".join(failures)
    raise RuntimeError(f"ACTPolicy import failed for all known paths:\n{joined}")


def check_ffmpeg() -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    result = subprocess.run(
        [exe, "-version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    first_line = result.stdout.splitlines()[0] if result.stdout else "ffmpeg"
    print(f"ok: {first_line}")


def check_cuda(torch) -> bool:
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - smoke test should expose CUDA init errors.
        print(f"torch.cuda.is_available error: {exc!r}")
        cuda_available = False

    print(f"torch.cuda.is_available: {cuda_available}")

    try:
        device_count = int(torch.cuda.device_count())
        print(f"torch.cuda.device_count: {device_count}")
        if device_count > 0:
            print(f"torch.cuda.device0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001 - CUDA diagnostics should continue after errors.
        print(f"torch.cuda.device_count error: {exc!r}")

    return cuda_available


def main() -> int:
    args = parse_args()

    torch = import_required_module("torch")
    print(f"torch: {torch.__version__}")
    cuda_available = check_cuda(torch)
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    import_required_module("lerobot")
    import_act_policy()
    import_required_module("torchcodec")
    check_ffmpeg()

    print("ok: env_hw3_robot smoke test passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI smoke test should report concise failure.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
