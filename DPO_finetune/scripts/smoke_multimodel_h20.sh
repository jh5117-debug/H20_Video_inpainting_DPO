#!/usr/bin/env bash
set -Eeuo pipefail

# Run one tiny generation job per adapter and print preview mp4 paths.
# This is for checking whether each model's real output video is sane before
# spending time on the full dataset.

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"
ADAPTER_CONFIG="${ADAPTER_CONFIG:-${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.json}"
SMOKE_ROOT="${SMOKE_ROOT:-/home/nvme03/workspace/world_model_phys/DPO_Multimodel_Smoke_$(date +%Y%m%d_%H%M%S)}"
METHOD_LIST="${METHODS:-propainter,cococo,minimax}"
SMOKE_MODE="${SMOKE_MODE:-combined}"

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
  "${PROJECT_ROOT}/data/external/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution" \
  "${PROJECT_ROOT}/data/external/davis_432_240/JPEGImages_432_240" \
  || true)}"

YTBV_ROOT="${YTBV_ROOT:-$(pick_first_dir \
  "${PROJECT_ROOT}/data/external/ytbv_2019_full_resolution/train/JPEGImages" \
  "${PROJECT_ROOT}/data/external/youtubevos_432_240/JPEGImages_432_240" \
  || true)}"

if [[ -z "${DAVIS_ROOT}" || -z "${YTBV_ROOT}" ]]; then
  echo "[error] DAVIS_ROOT or YTBV_ROOT not found."
  echo "        DAVIS_ROOT=${DAVIS_ROOT}"
  echo "        YTBV_ROOT=${YTBV_ROOT}"
  exit 1
fi

if [[ "${REFRESH_ADAPTER_CONFIG:-0}" == "1" ]]; then
  echo "[setup] refreshing adapter config from example"
  cp "${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.example.json" "${ADAPTER_CONFIG}"
elif [[ ! -f "${ADAPTER_CONFIG}" ]]; then
  echo "[setup] adapter config missing; copying example"
  cp "${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.example.json" "${ADAPTER_CONFIG}"
fi

mkdir -p "${SMOKE_ROOT}"

echo "[smoke] project=${PROJECT_ROOT}"
echo "[smoke] output=${SMOKE_ROOT}"
echo "[smoke] davis=${DAVIS_ROOT}"
echo "[smoke] ytbv=${YTBV_ROOT}"
echo "[smoke] adapters=${ADAPTER_CONFIG}"
echo "[smoke] methods=${METHOD_LIST}"
echo "[smoke] mode=${SMOKE_MODE}"

if [[ "${SMOKE_MODE}" == "combined" ]]; then
  echo
  echo "================ SMOKE: combined multimodel ================"
  OUT_ROOT="${SMOKE_ROOT}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  DIFFUERASER_ENV="${DIFFUERASER_ENV}" \
  ADAPTER_CONFIG="${ADAPTER_CONFIG}" \
  DAVIS_ROOT="${DAVIS_ROOT}" \
  YTBV_ROOT="${YTBV_ROOT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  GPUS="${GPUS:-0,1,2,3}" \
  METHODS="${METHOD_LIST}" \
  NUM_VIDEOS="${NUM_VIDEOS:-2}" \
  MAX_FRAMES="${MAX_FRAMES:-32}" \
  HEIGHT="${HEIGHT:-512}" \
  WIDTH="${WIDTH:-512}" \
  PARALLEL_METHODS="${PARALLEL_METHODS:-4}" \
  ENABLE_LPIPS=0 \
  ENABLE_VBENCH=0 \
  SAVE_PREVIEWS=1 \
  bash "${PROJECT_ROOT}/DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh" || true
else
  IFS=',' read -r -a METHODS_ARR <<< "${METHOD_LIST}"
  for method in "${METHODS_ARR[@]}"; do
    method="$(echo "${method}" | xargs)"
    [[ -n "${method}" ]] || continue
    echo
    echo "================ SMOKE: ${method} ================"
    OUT_ROOT="${SMOKE_ROOT}/${method}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    DIFFUERASER_ENV="${DIFFUERASER_ENV}" \
    ADAPTER_CONFIG="${ADAPTER_CONFIG}" \
    DAVIS_ROOT="${DAVIS_ROOT}" \
    YTBV_ROOT="${YTBV_ROOT}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    GPUS="${GPUS:-0,1,2,3}" \
    METHODS="${method}" \
    NUM_VIDEOS="${NUM_VIDEOS:-2}" \
    MAX_FRAMES="${MAX_FRAMES:-32}" \
    HEIGHT="${HEIGHT:-512}" \
    WIDTH="${WIDTH:-512}" \
    ENABLE_LPIPS=0 \
    ENABLE_VBENCH=0 \
    SAVE_PREVIEWS=1 \
    bash "${PROJECT_ROOT}/DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh" || true
  done
fi

echo
echo "================ PREVIEWS ================"
find "${SMOKE_ROOT}" -type f \( -name "gt_mask_raw_comp.mp4" -o -name "composited.mp4" -o -name "raw_output.mp4" \) | sort || true

echo
echo "================ LOGS / META ================"
find "${SMOKE_ROOT}" -type f \( -name "meta.json" -o -name "inference.log" -o -name "generation_summary.json" \) | sort || true

cat <<EOF

[done] Smoke root:
  ${SMOKE_ROOT}

Best files to inspect:
  */candidates/<method>/previews/gt_mask_raw_comp.mp4

If COCOCO or MiniMax produce no preview, open that method's inference.log and
check the third-party env and weight paths printed there.
EOF
