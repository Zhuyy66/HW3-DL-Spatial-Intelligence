#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_colmap_dense_mesh.sh \
    --images /abs/path/images \
    --sparse /abs/path/sparse/0 \
    --workspace /abs/path/dense_workspace \
    [--mode geometric|photometric] \
    [--max-image-size 2000] \
    [--skip-undistort] \
    [--skip-patch-match] \
    [--skip-poisson]

This script reuses an existing COLMAP sparse model to produce:
  - fused_<mode>.ply
  - meshed-poisson_<mode>.ply
  - meshed-delaunay_<mode>.ply

The workspace can be a fresh dense workspace or an existing one.
EOF
}

IMAGES=""
SPARSE=""
WORKSPACE=""
MODE="geometric"
MAX_IMAGE_SIZE="2000"
SKIP_UNDISTORT="false"
SKIP_PATCH_MATCH="false"
SKIP_POISSON="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --images)
      IMAGES="$2"
      shift 2
      ;;
    --sparse)
      SPARSE="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --max-image-size)
      MAX_IMAGE_SIZE="$2"
      shift 2
      ;;
    --skip-undistort)
      SKIP_UNDISTORT="true"
      shift
      ;;
    --skip-patch-match)
      SKIP_PATCH_MATCH="true"
      shift
      ;;
    --skip-poisson)
      SKIP_POISSON="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$IMAGES" || -z "$SPARSE" || -z "$WORKSPACE" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ "$MODE" != "geometric" && "$MODE" != "photometric" ]]; then
  echo "--mode must be 'geometric' or 'photometric'." >&2
  exit 1
fi

COLMAP_BIN="${COLMAP_BIN:-$(command -v colmap || true)}"
if [[ -z "$COLMAP_BIN" ]]; then
  echo "Could not find 'colmap' in PATH. Set COLMAP_BIN=/abs/path/to/colmap." >&2
  exit 1
fi

IMAGES="$(realpath "$IMAGES")"
SPARSE="$(realpath "$SPARSE")"
WORKSPACE="$(realpath -m "$WORKSPACE")"

if [[ ! -d "$IMAGES" ]]; then
  echo "Images directory not found: $IMAGES" >&2
  exit 1
fi

if [[ ! -d "$SPARSE" ]]; then
  echo "Sparse model directory not found: $SPARSE" >&2
  exit 1
fi

mkdir -p "$WORKSPACE"

if [[ "$SKIP_UNDISTORT" != "true" ]]; then
  echo "[1/4] image_undistorter -> $WORKSPACE"
  "$COLMAP_BIN" image_undistorter \
    --image_path "$IMAGES" \
    --input_path "$SPARSE" \
    --output_path "$WORKSPACE" \
    --output_type COLMAP
else
  echo "[1/4] skip image_undistorter"
fi

PATCH_MATCH_ARGS=(
  patch_match_stereo
  --workspace_path "$WORKSPACE"
  --workspace_format COLMAP
  --PatchMatchStereo.max_image_size "$MAX_IMAGE_SIZE"
)

if [[ "$MODE" == "geometric" ]]; then
  PATCH_MATCH_ARGS+=(--PatchMatchStereo.geom_consistency true)
else
  PATCH_MATCH_ARGS+=(--PatchMatchStereo.geom_consistency false)
fi

if [[ "$SKIP_PATCH_MATCH" != "true" ]]; then
  echo "[2/4] patch_match_stereo ($MODE)"
  "$COLMAP_BIN" "${PATCH_MATCH_ARGS[@]}"
else
  echo "[2/4] skip patch_match_stereo"
fi

FUSED_PLY="$WORKSPACE/fused_${MODE}.ply"
POISSON_PLY="$WORKSPACE/meshed-poisson_${MODE}.ply"
DELAUNAY_PLY="$WORKSPACE/meshed-delaunay_${MODE}.ply"

echo "[3/4] stereo_fusion -> $(basename "$FUSED_PLY")"
"$COLMAP_BIN" stereo_fusion \
  --workspace_path "$WORKSPACE" \
  --workspace_format COLMAP \
  --input_type "$MODE" \
  --output_path "$FUSED_PLY"

echo "[4/4] meshing"
if [[ "$SKIP_POISSON" != "true" ]]; then
  "$COLMAP_BIN" poisson_mesher \
    --input_path "$FUSED_PLY" \
    --output_path "$POISSON_PLY"
else
  echo "skip poisson_mesher"
fi

# COLMAP's dense delaunay mesher expects fused.ply(.vis) in the workspace root.
if [[ "$MODE" != "geometric" ]]; then
  echo "Warning: delaunay_mesher expects fused.ply from dense reconstruction; current mode is $MODE." >&2
fi
ln -sfn "$(basename "$FUSED_PLY")" "$WORKSPACE/fused.ply"
if [[ -f "${FUSED_PLY}.vis" ]]; then
  ln -sfn "$(basename "${FUSED_PLY}.vis")" "$WORKSPACE/fused.ply.vis"
fi

"$COLMAP_BIN" delaunay_mesher \
  --input_path "$WORKSPACE" \
  --input_type dense \
  --output_path "$DELAUNAY_PLY"

echo "Done."
echo "  fused:    $FUSED_PLY"
echo "  poisson:  $POISSON_PLY"
echo "  delaunay: $DELAUNAY_PLY"
