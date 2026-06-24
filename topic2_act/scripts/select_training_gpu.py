"""Select a GPU for HW3 Topic2 ACT training.

The selection policy is intentionally conservative:

1. Keep batch size 8 if any preferred 24GB candidate GPU has enough free VRAM.
2. If none fit, optionally consider the larger 49GB cards.
3. Only downgrade batch size when no allowed GPU can fit the requested batch.

The script writes a shell-sourceable env file for the server runbook and a JSON
sidecar containing the exact nvidia-smi snapshot used for the decision.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_gpu_pct: int


def parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-gpus", default="0,1,2,4,6,7")
    parser.add_argument("--large-gpus", default="3,5")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-free-mib", type=int, default=9000)
    parser.add_argument("--allow-large-gpus", action="store_true")
    parser.add_argument("--downgrade-batches", default="4,2")
    parser.add_argument("--write-env", required=True)
    parser.add_argument(
        "--nvidia-smi-output",
        default=None,
        help="Optional CSV fixture for testing without nvidia-smi.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_nvidia_smi() -> str:
    query = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


def parse_nvidia_smi_csv(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise ValueError(f"invalid nvidia-smi CSV at line {line_no}: {raw_line!r}")
        index, name, total, used, free, util = parts
        gpus.append(
            GpuInfo(
                index=int(index),
                name=name,
                memory_total_mib=int(total),
                memory_used_mib=int(used),
                memory_free_mib=int(free),
                utilization_gpu_pct=int(util),
            )
        )
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return gpus


def by_index(gpus: list[GpuInfo]) -> dict[int, GpuInfo]:
    return {gpu.index: gpu for gpu in gpus}


def enough_threshold(base_min_free_mib: int, requested_batch: int, chosen_batch: int) -> int:
    # Treat the batch-8 threshold as linear for explicit fallback decisions.
    return max(1, int(round(base_min_free_mib * chosen_batch / requested_batch)))


def choose_gpu(
    gpus: list[GpuInfo],
    *,
    candidate_ids: list[int],
    large_ids: list[int],
    requested_batch: int,
    min_free_mib: int,
    allow_large: bool,
    downgrade_batches: list[int],
) -> tuple[GpuInfo, int, str, int]:
    lookup = by_index(gpus)
    normal = [lookup[idx] for idx in candidate_ids if idx in lookup]
    large = [lookup[idx] for idx in large_ids if idx in lookup]
    if not normal and not large:
        raise RuntimeError("none of the requested candidate or large GPU ids exist in nvidia-smi output")

    def best_that_fits(pool: list[GpuInfo], threshold: int) -> GpuInfo | None:
        fits = [gpu for gpu in pool if gpu.memory_free_mib >= threshold]
        if not fits:
            return None
        return max(fits, key=lambda gpu: (gpu.memory_free_mib, -gpu.utilization_gpu_pct, -gpu.index))

    threshold = min_free_mib
    gpu = best_that_fits(normal, threshold)
    if gpu:
        return gpu, requested_batch, "preferred_24gb_candidate_has_enough_free_memory", threshold

    if allow_large:
        gpu = best_that_fits(large, threshold)
        if gpu:
            return gpu, requested_batch, "large_gpu_used_to_preserve_requested_batch_size", threshold

    for batch in downgrade_batches:
        threshold = enough_threshold(min_free_mib, requested_batch, batch)
        gpu = best_that_fits(normal, threshold)
        if gpu:
            return gpu, batch, "batch_downgraded_on_preferred_candidate", threshold
        if allow_large:
            gpu = best_that_fits(large, threshold)
            if gpu:
                return gpu, batch, "batch_downgraded_on_large_gpu", threshold

    all_allowed = normal + (large if allow_large else [])
    gpu = max(all_allowed, key=lambda item: (item.memory_free_mib, -item.utilization_gpu_pct, -item.index))
    batch = downgrade_batches[-1] if downgrade_batches else requested_batch
    threshold = enough_threshold(min_free_mib, requested_batch, batch)
    return gpu, batch, "no_gpu_met_threshold_selected_best_available_with_final_batch", threshold


def shell_export(name: str, value: object) -> str:
    return f"export {name}={shlex.quote(str(value))}"


def main() -> int:
    args = parse_args()
    candidate_ids = parse_int_list(args.candidate_gpus)
    large_ids = parse_int_list(args.large_gpus)
    downgrade_batches = parse_int_list(args.downgrade_batches)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if any(batch <= 0 for batch in downgrade_batches):
        raise ValueError("--downgrade-batches must contain only positive integers")

    raw_smi = args.nvidia_smi_output if args.nvidia_smi_output is not None else run_nvidia_smi()
    gpus = parse_nvidia_smi_csv(raw_smi)
    gpu, batch, reason, threshold = choose_gpu(
        gpus,
        candidate_ids=candidate_ids,
        large_ids=large_ids,
        requested_batch=args.batch_size,
        min_free_mib=args.min_free_mib,
        allow_large=args.allow_large_gpus,
        downgrade_batches=downgrade_batches,
    )

    payload = {
        "generated_at": utc_now(),
        "candidate_gpus": candidate_ids,
        "large_gpus": large_ids,
        "requested_batch_size": args.batch_size,
        "selected_gpu": asdict(gpu),
        "selected_batch_size": batch,
        "selection_reason": reason,
        "required_free_mib_for_selected_batch": threshold,
        "all_gpus": [asdict(item) for item in gpus],
    }

    env_path = Path(args.write_env)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = env_path.with_suffix(env_path.suffix + ".json")
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env_lines = [
        "# Source this file before Day 4 ACT training commands.",
        f"# Generated at {payload['generated_at']}",
        shell_export("HW3_GPU_ID", gpu.index),
        shell_export("HW3_BATCH_SIZE", batch),
        shell_export("HW3_GPU_REASON", reason),
        shell_export("HW3_GPU_REQUIRED_FREE_MIB", threshold),
        shell_export("HW3_GPU_FREE_MIB", gpu.memory_free_mib),
        shell_export("HW3_GPU_SNAPSHOT_JSON", snapshot_path),
    ]
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"ok: wrote {env_path}")
    print(f"ok: wrote {snapshot_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI diagnostics should be concise.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
