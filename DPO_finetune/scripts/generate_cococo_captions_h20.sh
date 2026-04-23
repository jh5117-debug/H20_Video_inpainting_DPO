#!/usr/bin/env bash
set -Eeuo pipefail

# Generate a caption JSON for COCOCO prompts with a local Qwen2.5-VL model.

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"
CAPTION_MODEL="${CAPTION_MODEL:-${PROJECT_ROOT}/weights/Qwen2.5-VL-7B-Instruct}"
CAPTION_JSON="${CAPTION_JSON:-${PROJECT_ROOT}/DPO_finetune/captions/cococo_qwen_captions.json}"
CAPTION_GPU="${CAPTION_GPU:-7}"
CAPTION_NUM_VIDEOS="${CAPTION_NUM_VIDEOS:-0}"
CAPTION_MAX_FRAMES="${CAPTION_MAX_FRAMES:-48}"
CAPTION_FRAMES="${CAPTION_FRAMES:-4}"
CAPTION_DTYPE="${CAPTION_DTYPE:-bfloat16}"
CAPTION_DEVICE_MAP="${CAPTION_DEVICE_MAP:-auto}"
CAPTION_INSTALL_DEPS="${CAPTION_INSTALL_DEPS:-0}"
FALLBACK_ONLY="${FALLBACK_ONLY:-0}"

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

PYTHON_RUN=(python)
if [[ -x "/home/nvme01/miniconda3/bin/conda" && -d "${DIFFUERASER_ENV}" ]]; then
  PYTHON_RUN=(/home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${DIFFUERASER_ENV}" python)
fi

if [[ "${CAPTION_INSTALL_DEPS}" == "1" ]]; then
  echo "[caption] installing/upgrading Qwen caption dependencies in ${DIFFUERASER_ENV}"
  "${PYTHON_RUN[@]}" -m pip install -U "transformers>=4.49.0" accelerate qwen-vl-utils
fi

ARGS=(
  "${PROJECT_ROOT}/DPO_finetune/generate_cococo_captions_qwen.py"
  --project_root "${PROJECT_ROOT}"
  --ytbv_root "${YTBV_ROOT}"
  --davis_root "${DAVIS_ROOT}"
  --output_json "${CAPTION_JSON}"
  --model_path "${CAPTION_MODEL}"
  --num_videos "${CAPTION_NUM_VIDEOS}"
  --max_frames "${CAPTION_MAX_FRAMES}"
  --caption_frames "${CAPTION_FRAMES}"
  --dtype "${CAPTION_DTYPE}"
  --device_map "${CAPTION_DEVICE_MAP}"
)

if [[ "${FALLBACK_ONLY}" == "1" ]]; then
  ARGS+=(--fallback_only)
fi
if [[ "${OVERWRITE_CAPTIONS:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[caption] CUDA_VISIBLE_DEVICES=${CAPTION_GPU}"
echo "[caption] model=${CAPTION_MODEL}"
echo "[caption] output=${CAPTION_JSON}"

CUDA_VISIBLE_DEVICES="${CAPTION_GPU}" "${PYTHON_RUN[@]}" "${ARGS[@]}"
