#!/usr/bin/env bash
set -Eeuo pipefail

# 20-ish sample smoke for the new DPO data-generation band:
#   mask area 20%-30%, no mask dilation, margin 0.10,
#   negative quality band 0.30-0.65 with target 0.45.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
DATA_NAME="${DATA_NAME:-Video_inpainting_DPO}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"
DATA="${DATA:-${PROJECT_ROOT}/data}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${DATA}/third_party_video_inpainting}"
SMOKE_ROOT="${SMOKE_ROOT:-${PROJECT_ROOT}/smoke_outputs/DPO_NewBand20_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_GPUS="0,1,2,3,4,5,6,7"

pick_first_dir() {
  for p in "$@"; do
    if [[ -d "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

DAVIS_ROOT="${DAVIS_ROOT:-$(pick_first_dir \
  "${DATA}/external/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution" \
  "${DATA}/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution" \
  "${DATA}/external/davis_432_240/JPEGImages_432_240" \
  "${DATA}/davis_432_240/JPEGImages_432_240" \
  || true)}"

YTBV_ROOT="${YTBV_ROOT:-$(pick_first_dir \
  "${DATA}/external/ytbv_2019_full_resolution/train/JPEGImages" \
  "${DATA}/ytbv_2019_full_resolution/train/JPEGImages" \
  "${DATA}/external/youtubevos_432_240/JPEGImages_432_240" \
  "${DATA}/youtubevos_432_240/JPEGImages_432_240" \
  || true)}"

if [[ -z "${DAVIS_ROOT}" || -z "${YTBV_ROOT}" ]]; then
  echo "[smoke][error] DAVIS_ROOT or YTBV_ROOT not found." >&2
  echo "[smoke][error] DAVIS_ROOT=${DAVIS_ROOT}" >&2
  echo "[smoke][error] YTBV_ROOT=${YTBV_ROOT}" >&2
  exit 1
fi

mkdir -p "${SMOKE_ROOT}"

export PROJECT_ROOT
export PROJECT_NAME
export DATA
export DIFFUERASER_ENV
export THIRD_PARTY_ROOT
export OUT_ROOT="${SMOKE_ROOT}"
export DAVIS_ROOT
export YTBV_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_GPUS}}"
export GPUS="${GPUS:-${DEFAULT_GPUS}}"
export METHODS="${METHODS:-propainter,cococo,diffueraser,minimax}"
export NUM_VIDEOS="${NUM_VIDEOS:-20}"
export MAX_FRAMES="${MAX_FRAMES:-32}"
export HEIGHT="${HEIGHT:-512}"
export WIDTH="${WIDTH:-512}"
export TRAIN_NFRAMES="${TRAIN_NFRAMES:-16}"
export SCORE_WINDOWS="${SCORE_WINDOWS:-32,24,16}"

export MASK_DILATION_ITER="${MASK_DILATION_ITER:-0}"
export MASK_AREA_MIN="${MASK_AREA_MIN:-0.20}"
export MASK_AREA_MAX="${MASK_AREA_MAX:-0.30}"
export MASK_MARGIN_RATIO="${MASK_MARGIN_RATIO:-0.10}"

export NEG_QUALITY_MIN="${NEG_QUALITY_MIN:-0.30}"
export NEG_QUALITY_MAX="${NEG_QUALITY_MAX:-0.65}"
export NEG_QUALITY_TARGET="${NEG_QUALITY_TARGET:-0.45}"

export PARALLEL_METHODS="${PARALLEL_METHODS:-4}"
export PARALLEL_VIDEOS="${PARALLEL_VIDEOS:-4}"
export GPU_SLOTS="${GPU_SLOTS:-1}"
export METHOD_GPU_MAP="${METHOD_GPU_MAP:-cococo=0,1;diffueraser=2,3;minimax=4,5;propainter=6,7,0,1,2,3,4,5}"

export ENABLE_LPIPS="${ENABLE_LPIPS:-1}"
export ENABLE_VBENCH="${ENABLE_VBENCH:-1}"
export SAVE_PREVIEWS="${SAVE_PREVIEWS:-1}"
export CANDIDATE_RETENTION="${CANDIDATE_RETENTION:-selected}"
export CLEANUP_FAILED="${CLEANUP_FAILED:-1}"

LOG_PATH="${LOG_PATH:-${SMOKE_ROOT}/smoke_newband20.log}"

echo "[smoke] project=${PROJECT_ROOT}"
echo "[smoke] output=${SMOKE_ROOT}"
echo "[smoke] log=${LOG_PATH}"
echo "[smoke] gpus=${GPUS} slots=${GPU_SLOTS} parallel_videos=${PARALLEL_VIDEOS} parallel_methods=${PARALLEL_METHODS}"
echo "[smoke] davis=${DAVIS_ROOT}"
echo "[smoke] ytbv=${YTBV_ROOT}"

bash "${PROJECT_ROOT}/DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh" 2>&1 | tee "${LOG_PATH}"

echo
echo "================ SUMMARY ================"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
manifest_path = root / "manifest.json"
summary_path = root / "generation_summary.json"
manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}
summary = json.load(open(summary_path)) if summary_path.exists() else {}
print("smoke_root =", root)
print("manifest_entries =", len(manifest))
print("mask_policy =", summary.get("mask_policy"))
print("neg_selection =", summary.get("neg_selection"))
PY

echo
echo "================ PREVIEWS ================"
find "${SMOKE_ROOT}" -type f \( -name "gt_mask_raw_comp.mp4" -o -name "meta.json" -o -name "generation_summary.json" \) | sort
