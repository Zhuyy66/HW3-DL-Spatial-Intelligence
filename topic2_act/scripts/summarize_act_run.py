"""Summarize and gate HW3 Topic2 ACT training runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ERROR_PATTERNS = (
    "traceback",
    "error:",
    "outofmemoryerror",
    "cuda out of memory",
    "runtimeerror:",
    "valueerror:",
    "keyerror:",
    "the following arguments are required",
    "nan",
)
RECENT_ERROR_LINE_PATTERNS = (
    "error:",
    "traceback",
    "runtimeerror:",
    "valueerror:",
    "keyerror:",
    "typeerror:",
    "outofmemoryerror",
    "cuda out of memory",
    "the following arguments are required",
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--min-epochs", type=float, default=0.0)
    parser.add_argument("--require-healthy-loss", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--require-wandb", action="store_true")
    parser.add_argument("--loss-window", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"raw": line, "parse_error": line_no})
    return rows


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def parse_metric_line(line: str) -> dict[str, Any] | None:
    clean = strip_ansi(line)
    if "loss" not in clean or "step" not in clean:
        return None
    row: dict[str, Any] = {"raw": clean}
    for key, raw_value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*):\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", clean):
        normalized = {"epch": "epoch", "grdn": "grad_norm"}.get(key, key)
        try:
            value: int | float
            value = int(raw_value) if re.fullmatch(r"[-+]?\d+", raw_value) else float(raw_value)
        except ValueError:
            continue
        row[normalized] = value
    return row if "loss" in row else None


def parse_wandb_url(text: str) -> str | None:
    match = re.search(r"https://wandb\.ai/\S+", strip_ansi(text))
    if not match:
        return None
    return match.group(0).rstrip(").,")


def recent_error_lines(text: str, *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        clean = strip_ansi(line).strip()
        lower = clean.lower()
        if clean and any(pattern in lower for pattern in RECENT_ERROR_LINE_PATTERNS):
            lines.append(clean)
    return lines[-limit:]


def pid_running(pid_file: Path | None) -> tuple[bool | None, str]:
    if pid_file is None:
        return None, "pid file not provided"
    if not pid_file.exists():
        return None, f"pid file not found: {pid_file}"
    raw = pid_file.read_text(encoding="utf-8").strip()
    if not raw:
        return None, f"pid file is empty: {pid_file}"
    try:
        pid = int(raw)
    except ValueError:
        return None, f"invalid pid value {raw!r}"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, f"pid {pid} is not running"
    except PermissionError:
        return True, f"pid {pid} exists but permission check was denied"
    except OSError as exc:
        if os.name == "nt":
            return None, f"pid check unavailable on this platform: {exc}"
        return None, f"pid check failed for {pid}: {exc}"
    return True, f"pid {pid} is running"


def checkpoint_files(run_dir: Path) -> list[Path]:
    patterns = (
        "lerobot_train/checkpoints/**/*.safetensors",
        "lerobot_train/checkpoints/**/*.bin",
        "lerobot_train/checkpoints/**/*.pt",
        "lerobot_train/checkpoints/**/*.pth",
        "checkpoints/**/*.safetensors",
        "checkpoints/**/*.bin",
        "checkpoints/**/*.pt",
        "checkpoints/**/*.pth",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in run_dir.glob(pattern) if path.is_file())
    return sorted(set(paths))


def collect_metrics(run_dir: Path, log_text: str) -> list[dict[str, Any]]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    parsed_from_log = [row for row in (parse_metric_line(line) for line in log_text.splitlines()) if row]
    if parsed_from_log and len(parsed_from_log) > len(metrics):
        return parsed_from_log
    return metrics


def finite_losses(metrics: list[dict[str, Any]]) -> list[float]:
    losses: list[float] = []
    for row in metrics:
        value = row.get("loss")
        if isinstance(value, (int, float)):
            losses.append(float(value))
    return losses


def max_epoch(metrics: list[dict[str, Any]], manifest: dict[str, Any]) -> float:
    observed = []
    for row in metrics:
        value = row.get("epoch")
        if isinstance(value, (int, float)):
            observed.append(float(value))
    if observed:
        return max(observed)
    status = manifest.get("status")
    if status == "completed":
        return float(manifest.get("epochs") or 0)
    steps_per_epoch = float(manifest.get("steps_per_epoch") or 0)
    max_step = max((float(row.get("step")) for row in metrics if isinstance(row.get("step"), (int, float))), default=0.0)
    return max_step / steps_per_epoch if steps_per_epoch else 0.0


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def loss_health(losses: list[float], *, window: int) -> tuple[bool, str]:
    if len(losses) < max(4, min(window, 4)):
        return False, f"not enough loss points ({len(losses)})"
    if any(not math.isfinite(value) for value in losses):
        return False, "loss contains NaN or Inf"

    first = losses[: min(window, len(losses) // 3 or 1)]
    last = losses[-min(window, len(losses) // 3 or 1) :]
    first_avg = average(first)
    last_avg = average(last)
    max_loss = max(losses)
    if first_avg <= 0:
        return False, f"first loss window average is non-positive: {first_avg:.6g}"
    if max_loss > first_avg * 10:
        return False, f"loss exploded: max={max_loss:.6g}, first_avg={first_avg:.6g}"
    if last_avg > first_avg * 1.05:
        return False, f"loss did not trend down/stabilize: first_avg={first_avg:.6g}, last_avg={last_avg:.6g}"
    if min(losses[len(losses) // 2 :]) >= first_avg:
        return False, f"no later loss point improved over first_avg={first_avg:.6g}"
    return True, f"loss healthy: first_avg={first_avg:.6g}, last_avg={last_avg:.6g}, min={min(losses):.6g}"


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    log_text = ""
    if args.log_file:
        log_path = Path(args.log_file)
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")

    metrics = collect_metrics(run_dir, log_text)
    losses = finite_losses(metrics)
    checkpoints = checkpoint_files(run_dir)
    wandb_url = manifest.get("wandb_url") or parse_wandb_url(log_text or "")
    running, pid_detail = pid_running(Path(args.pid_file) if args.pid_file else None)

    checks: list[CheckResult] = []
    log_lower = log_text.lower()
    error_hits = [pattern for pattern in ERROR_PATTERNS if pattern in log_lower]
    recent_errors = recent_error_lines(log_text)
    error_detail = (
        f"hits={error_hits}; recent_errors={recent_errors}" if error_hits else "no fatal pattern found"
    )
    checks.append(CheckResult("log_errors", not error_hits, error_detail))
    checks.append(CheckResult("metrics", bool(metrics), f"metric_rows={len(metrics)}", required=False))
    checks.append(CheckResult("loss_finite", bool(losses) and all(math.isfinite(value) for value in losses), f"loss_points={len(losses)}"))

    observed_epoch = max_epoch(metrics, manifest)
    if args.min_epochs:
        checks.append(
            CheckResult(
                "min_epochs",
                observed_epoch >= args.min_epochs,
                f"observed_epoch={observed_epoch:.3f}, required={args.min_epochs}",
            )
        )

    if args.require_checkpoint:
        checks.append(CheckResult("checkpoint", bool(checkpoints), f"checkpoint_files={len(checkpoints)}"))
    else:
        checks.append(CheckResult("checkpoint", bool(checkpoints), f"checkpoint_files={len(checkpoints)}", required=False))

    if args.require_wandb:
        checks.append(CheckResult("wandb", bool(wandb_url), f"wandb_url={wandb_url or 'missing'}"))
    else:
        checks.append(CheckResult("wandb", bool(wandb_url), f"wandb_url={wandb_url or 'missing'}", required=False))

    if args.require_healthy_loss:
        ok, detail = loss_health(losses, window=args.loss_window)
        checks.append(CheckResult("healthy_loss", ok, detail))

    checks.append(CheckResult("pid_status", True, pid_detail, required=False))

    required_failures = [check for check in checks if check.required and not check.ok]
    pending = bool(args.min_epochs and observed_epoch < args.min_epochs and running)
    verdict = "healthy" if not required_failures else ("pending" if pending else "unhealthy")

    summary = {
        "verdict": verdict,
        "run_dir": str(run_dir),
        "manifest_status": manifest.get("status"),
        "pid_running": running,
        "observed_epoch": observed_epoch,
        "metric_rows": len(metrics),
        "loss_points": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "checkpoint_count": len(checkpoints),
        "wandb_url": wandb_url,
        "recent_error_lines": recent_errors,
        "checks": [check.__dict__ for check in checks],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"verdict: {verdict}")
    if verdict == "healthy":
        return 0
    if verdict == "pending":
        return 2
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - runbook diagnostics should be concise.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
