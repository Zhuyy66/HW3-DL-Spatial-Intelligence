#!/usr/bin/env bash
# Source this before CUDA/PyTorch commands on the server.
#
# Usage:
#   source scripts/activate_cuda_driver_shim.sh [cuda_visible_devices]
#
# The script filters CUDA compat/stubs paths that can trigger CUDA error 804,
# creates project-local libcuda.so symlinks to the real host driver, and puts
# that shim first in LD_LIBRARY_PATH.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[WARN] Source this script so it can modify the current shell:"
  echo "       source scripts/activate_cuda_driver_shim.sh [cuda_visible_devices]"
  echo "       Continuing in this subprocess for diagnostics only."
fi

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

detect_nvidia_driver_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | head -n 1 \
    | tr -d '[:space:]'
}

find_real_libcuda() {
  local driver_version="${1:-}"
  local roots=(
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/lib
    /lib/x86_64-linux-gnu
    /run/nvidia/driver/usr/lib/x86_64-linux-gnu
    /run/nvidia/driver/usr/lib64
  )
  local candidate resolved root

  if [[ -n "${REAL_CUDA_DRIVER:-}" && -r "${REAL_CUDA_DRIVER}" ]]; then
    readlink -f "${REAL_CUDA_DRIVER}"
    return 0
  fi

  if [[ -n "${driver_version}" ]]; then
    for root in "${roots[@]}"; do
      candidate="${root}/libcuda.so.${driver_version}"
      if [[ -e "${candidate}" ]]; then
        resolved="$(readlink -f "${candidate}")"
        if [[ -f "${resolved}" && "${resolved}" != *"/compat/"* && "${resolved}" != *"/stubs/"* ]]; then
          echo "${resolved}"
          return 0
        fi
      fi
    done
  fi

  while IFS= read -r candidate; do
    resolved="$(readlink -f "${candidate}")"
    if [[ -f "${resolved}" && "${resolved}" != *"/compat/"* && "${resolved}" != *"/stubs/"* ]]; then
      echo "${resolved}"
      return 0
    fi
  done < <(
    find /usr /lib /run/nvidia/driver \
      -path '*/compat/*' -prune -o \
      -path '*/stubs/*' -prune -o \
      -type f -name 'libcuda.so.*' -print 2>/dev/null \
      | sort -Vr
  )

  return 1
}

filter_ld_library_path() {
  CUDA_DRIVER_SHIM="${CUDA_DRIVER_SHIM}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" python - <<'PY'
import os

shim = os.environ["CUDA_DRIVER_SHIM"]
parts = [shim]
for path in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if not path:
        continue
    lower = path.lower()
    if "/compat" in lower or lower.endswith("/compat"):
        continue
    if "/stubs" in lower or lower.endswith("/stubs"):
        continue
    if path == shim:
        continue
    parts.append(path)
print(":".join(parts))
PY
}

driver_version="$(detect_nvidia_driver_version || true)"
real_libcuda="$(find_real_libcuda "${driver_version}" || true)"

if [[ -z "${real_libcuda}" ]]; then
  echo "[ERROR] Could not find a non-compat libcuda.so for driver '${driver_version:-unknown}'." >&2
  echo "        Override with REAL_CUDA_DRIVER=/path/to/libcuda.so.xxx before sourcing this script." >&2
  return 1 2>/dev/null || exit 1
fi

export CUDA_VISIBLE_DEVICES="${1:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_DRIVER_SHIM="${PROJECT_ROOT}/.cuda_driver_shim"
mkdir -p "${CUDA_DRIVER_SHIM}"
ln -sfn "${real_libcuda}" "${CUDA_DRIVER_SHIM}/libcuda.so.1"
ln -sfn "${real_libcuda}" "${CUDA_DRIVER_SHIM}/libcuda.so"

export LD_LIBRARY_PATH="$(filter_ld_library_path)"
unset LD_PRELOAD

echo "[OK] CUDA driver shim active"
echo "     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "     nvidia_driver=${driver_version:-unknown}"
echo "     libcuda.so.1 -> $(readlink -f "${CUDA_DRIVER_SHIM}/libcuda.so.1")"
echo "     LD_LIBRARY_PATH first entries:"
echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | sed -n '1,8p'
