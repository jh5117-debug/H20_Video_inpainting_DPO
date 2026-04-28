#!/usr/bin/env bash
set -Eeuo pipefail

# Build GT+mask+four-model comparison videos from H20 smoke outputs.
# If SMOKE_ROOTS is empty, all DPO_Multimodel_Smoke_* roots are scanned.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"
SMOKE_OUTPUTS_DIR="${SMOKE_OUTPUTS_DIR:-${PROJECT_ROOT}/smoke_outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${SMOKE_OUTPUTS_DIR}/five_panel_smoke_comparisons_$(date +%Y%m%d_%H%M%S)}"
METHODS="${METHODS:-propainter,cococo,diffueraser,minimax}"
FPS="${FPS:-8}"
MAX_FRAMES="${MAX_FRAMES:-0}"
PANEL_WIDTH="${PANEL_WIDTH:-320}"
PANEL_HEIGHT="${PANEL_HEIGHT:-320}"

PYTHON_BIN=(python)
if [[ -x "/home/nvme01/miniconda3/bin/conda" && -d "${DIFFUERASER_ENV}" ]]; then
  PYTHON_BIN=(/home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${DIFFUERASER_ENV}" python)
fi

ARGS=(
  "${PROJECT_ROOT}/DPO_finetune/make_smoke_comparison_videos.py"
  --output_dir "${OUTPUT_DIR}"
  --methods "${METHODS}"
  --fps "${FPS}"
  --max_frames "${MAX_FRAMES}"
  --panel_width "${PANEL_WIDTH}"
  --panel_height "${PANEL_HEIGHT}"
)

if [[ -n "${SMOKE_ROOTS:-}" ]]; then
  IFS=',' read -r -a ROOT_ARR <<< "${SMOKE_ROOTS}"
  for root in "${ROOT_ARR[@]}"; do
    root="$(echo "${root}" | xargs)"
    [[ -n "${root}" ]] || continue
    ARGS+=(--smoke_root "${root}")
  done
else
  ARGS+=(--smoke_outputs_dir "${SMOKE_OUTPUTS_DIR}")
fi

echo "[compare] project=${PROJECT_ROOT}"
echo "[compare] output=${OUTPUT_DIR}"
echo "[compare] methods=${METHODS}"
echo "[compare] smoke_roots=${SMOKE_ROOTS:-ALL under ${SMOKE_OUTPUTS_DIR}}"

PYTHONNOUSERSITE=1 "${PYTHON_BIN[@]}" "${ARGS[@]}"

echo
echo "[done] comparison videos:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name "*.mp4" | sort
