"""Verify the server-side CALVIN evaluation environment for HW3 Day 5."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test env_hw3_calvin_eval imports and EGL mapping.")
    parser.add_argument("--calvin-root", default=None, help="Path to the official CALVIN clone.")
    parser.add_argument("--cuda-device", type=int, default=0, help="Logical CUDA device used by CALVIN.")
    parser.add_argument(
        "--allow-missing-egl",
        action="store_true",
        help="Report EGL mapping failures without failing the command.",
    )
    parser.add_argument(
        "--skip-raw-egl-diagnostics",
        action="store_true",
        help="Skip GLVND/ldconfig/eglinfo/CALVIN raw EGL diagnostics.",
    )
    parser.add_argument(
        "--egl-diagnostics-dir",
        default=None,
        help="Optional directory for raw EGL diagnostic stdout/stderr files.",
    )
    return parser.parse_args()


def add_calvin_paths(calvin_root: str | Path | None) -> Path | None:
    if calvin_root is None:
        return None
    root = Path(calvin_root).expanduser().resolve()
    for rel in ("calvin_models", "calvin_env"):
        candidate = root / rel
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    sys.path.insert(0, str(root))
    return root


def env_snapshot() -> dict[str, Any]:
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "EGL_VISIBLE_DEVICES": os.environ.get("EGL_VISIBLE_DEVICES"),
        "DISPLAY": os.environ.get("DISPLAY"),
        "LD_LIBRARY_PATH_first_entries": [entry for entry in ld_library_path.split(":") if entry][:8],
    }


def import_check(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - verification should show exact import errors.
        return {"name": name, "ok": False, "error": repr(exc)}
    version = getattr(module, "__version__", None)
    path = getattr(module, "__file__", None)
    return {"name": name, "ok": True, "version": version, "path": path}


def torch_check() -> dict[str, Any]:
    result = import_check("torch")
    if not result["ok"]:
        return result
    import torch

    result.update(
        {
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        result["device0"] = torch.cuda.get_device_name(0)
    return result


def egl_check(cuda_device: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_device": cuda_device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "EGL_VISIBLE_DEVICES_before": os.environ.get("EGL_VISIBLE_DEVICES"),
        "ok": False,
    }
    try:
        from calvin_env.utils.utils import get_egl_device_id

        egl_device = int(get_egl_device_id(cuda_device))
        result["egl_device"] = egl_device
        result["ok"] = True
        if "EGL_VISIBLE_DEVICES" not in os.environ:
            os.environ["EGL_VISIBLE_DEVICES"] = str(egl_device)
    except Exception as exc:  # noqa: BLE001 - expose EGL build/device failures.
        result["error"] = repr(exc)
    result["EGL_VISIBLE_DEVICES_after"] = os.environ.get("EGL_VISIBLE_DEVICES")
    return result


def truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def write_text_if_requested(output_dir: Path | None, name: str, text: str) -> str | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(text, encoding="utf-8", errors="replace")
    return str(path)


def run_shell(command: str, output_dir: Path | None = None, name: str | None = None, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            shell=True,
            text=True,
            capture_output=True,
            executable="/bin/bash" if Path("/bin/bash").exists() else None,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive missing commands.
        return {"command": command, "ok": False, "error": repr(exc)}

    stdout_path = write_text_if_requested(output_dir, f"{name}.stdout.log", completed.stdout) if name else None
    stderr_path = write_text_if_requested(output_dir, f"{name}.stderr.log", completed.stderr) if name else None
    return {
        "command": command,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": truncate_text(completed.stdout),
        "stderr": truncate_text(completed.stderr),
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def collect_glvnd_vendor_files(output_dir: Path | None) -> list[dict[str, Any]]:
    vendor_files: list[dict[str, Any]] = []
    for directory in (Path("/usr/share/glvnd/egl_vendor.d"), Path("/etc/glvnd/egl_vendor.d")):
        if not directory.exists():
            vendor_files.append({"directory": str(directory), "exists": False})
            continue
        json_files = sorted(directory.glob("*.json"))
        vendor_files.append({"directory": str(directory), "exists": True, "json_count": len(json_files)})
        for json_file in json_files:
            try:
                content = json_file.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                vendor_files.append({"path": str(json_file), "ok": False, "error": repr(exc)})
                continue
            written = write_text_if_requested(output_dir, f"glvnd_{json_file.name}", content)
            vendor_files.append({"path": str(json_file), "ok": True, "content": content, "copy_path": written})
    return vendor_files


def collect_raw_egl_diagnostics(calvin_root: Path | None, output_dir_arg: str | None) -> dict[str, Any]:
    output_dir = Path(output_dir_arg).expanduser().resolve() if output_dir_arg else None
    diagnostics: dict[str, Any] = {
        "glvnd_vendor_files": collect_glvnd_vendor_files(output_dir),
        "ldconfig_egl_nvidia": run_shell(
            "ldconfig -p | grep -E 'libEGL|libGLX|libOpenGL|nvidia' || true",
            output_dir,
            "ldconfig_egl_nvidia",
        ),
        "driver_libs": run_shell(
            "roots=(); for root in /usr /lib /opt /run; do [ -e \"$root\" ] && roots+=(\"$root\"); done; "
            "find \"${roots[@]}\" \\( -type f -o -type l \\) "
            "\\( -name 'libEGL_nvidia.so*' -o -name 'libGLX_nvidia.so*' "
            "-o -name 'libnvidia-egl*.so*' -o -name 'libcuda.so*' \\) -print 2>/dev/null || true",
            output_dir,
            "driver_libs",
        ),
        "eglinfo_B": run_shell("eglinfo -B || true", output_dir, "eglinfo_B"),
    }

    if calvin_root is None:
        diagnostics["calvin_egl_checker"] = {"ok": False, "error": "--calvin-root not provided"}
        return diagnostics

    egl_dir = calvin_root / "calvin_env" / "egl_check"
    if not egl_dir.is_dir():
        diagnostics["calvin_egl_checker"] = {"ok": False, "error": f"missing CALVIN egl_check dir: {egl_dir}"}
        return diagnostics

    build = run_shell("bash build.sh", output_dir, "calvin_egl_build", cwd=egl_dir)
    probe = run_shell("./EGL_options.o || true", output_dir, "calvin_EGL_options", cwd=egl_dir)
    diagnostics["calvin_egl_checker"] = {
        "egl_check_dir": str(egl_dir),
        "build": build,
        "probe": probe,
    }
    return diagnostics


def main() -> int:
    args = parse_args()
    calvin_root = add_calvin_paths(args.calvin_root)

    checks = [
        {"name": "python", "ok": True, "executable": sys.executable, "version": sys.version},
        import_check("numpy"),
        torch_check(),
        import_check("git"),
        import_check("pybullet"),
        import_check("egl_probe"),
        import_check("safetensors"),
        import_check("cv2"),
        import_check("hydra"),
        import_check("omegaconf"),
        import_check("pytorch_lightning"),
        import_check("quaternion"),
        import_check("gym"),
        import_check("termcolor"),
        import_check("tqdm"),
        import_check("calvin_agent"),
        import_check("calvin_env"),
    ]
    egl = egl_check(args.cuda_device)

    payload = {
        "calvin_root": str(calvin_root) if calvin_root else None,
        "env": env_snapshot(),
        "checks": checks,
        "egl": egl,
    }
    if not args.skip_raw_egl_diagnostics:
        payload["raw_egl_diagnostics"] = collect_raw_egl_diagnostics(calvin_root, args.egl_diagnostics_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    failed = [check for check in checks if not check.get("ok")]
    if failed:
        print("error: required import checks failed", file=sys.stderr)
        return 1
    if not egl.get("ok") and not args.allow_missing_egl:
        print("error: EGL/CUDA mapping check failed", file=sys.stderr)
        return 1
    print("ok: env_hw3_calvin_eval verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
