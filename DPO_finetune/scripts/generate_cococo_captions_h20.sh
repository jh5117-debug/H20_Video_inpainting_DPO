#!/usr/bin/env bash
set -Eeuo pipefail

# Generate a caption JSON for COCOCO prompts with a local Qwen2.5-VL model.

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"
CAPTION_ENV="${CAPTION_ENV:-${PROJECT_ROOT}/third_party_video_inpainting/envs/qwen_caption}"
CAPTION_MODEL="${CAPTION_MODEL:-${PROJECT_ROOT}/weights/Qwen2.5-VL-7B-Instruct}"
CAPTION_JSON="${CAPTION_JSON:-${PROJECT_ROOT}/DPO_finetune/captions/cococo_qwen_captions.json}"
CAPTION_GPU="${CAPTION_GPU:-7}"
CAPTION_NUM_VIDEOS="${CAPTION_NUM_VIDEOS:-0}"
CAPTION_MAX_FRAMES="${CAPTION_MAX_FRAMES:-48}"
CAPTION_FRAMES="${CAPTION_FRAMES:-4}"
CAPTION_DTYPE="${CAPTION_DTYPE:-bfloat16}"
CAPTION_DEVICE_MAP="${CAPTION_DEVICE_MAP:-auto}"
CAPTION_CREATE_ENV="${CAPTION_CREATE_ENV:-0}"
CAPTION_INSTALL_DEPS="${CAPTION_INSTALL_DEPS:-0}"
CAPTION_TRANSFORMERS_SPEC="${CAPTION_TRANSFORMERS_SPEC:-transformers>=4.49.0,<4.58}"
CAPTION_HF_HUB_SPEC="${CAPTION_HF_HUB_SPEC:-huggingface_hub<1.0}"
CAPTION_ACCELERATE_SPEC="${CAPTION_ACCELERATE_SPEC:-accelerate>=0.30}"
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

CONDA_BIN="${CONDA_BIN:-/home/nvme01/miniconda3/bin/conda}"

if [[ "${CAPTION_CREATE_ENV}" == "1" ]]; then
  if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "[caption][error] conda not found: ${CONDA_BIN}" >&2
    exit 1
  fi
  if [[ ! -d "${CAPTION_ENV}" ]]; then
    mkdir -p "$(dirname "${CAPTION_ENV}")"
    if [[ -d "${DIFFUERASER_ENV}" ]]; then
      echo "[caption] create isolated caption env by cloning ${DIFFUERASER_ENV}"
      "${CONDA_BIN}" create -y -p "${CAPTION_ENV}" --clone "${DIFFUERASER_ENV}"
    else
      echo "[caption] create isolated caption env ${CAPTION_ENV}"
      "${CONDA_BIN}" create -y -p "${CAPTION_ENV}" python=3.10 pip
    fi
  fi
fi

PYTHON_ENV=""
if [[ -x "${CONDA_BIN}" && -d "${CAPTION_ENV}" ]]; then
  PYTHON_ENV="${CAPTION_ENV}"
elif [[ -x "${CONDA_BIN}" && -d "${DIFFUERASER_ENV}" ]]; then
  PYTHON_ENV="${DIFFUERASER_ENV}"
fi

PYTHON_RUN=(python)
if [[ -n "${PYTHON_ENV}" ]]; then
  PYTHON_RUN=("${CONDA_BIN}" run --no-capture-output -p "${PYTHON_ENV}" python)
fi

if [[ "${CAPTION_INSTALL_DEPS}" == "1" ]]; then
  echo "[caption] installing/upgrading Qwen caption dependencies in ${PYTHON_ENV:-current shell python}"
  "${PYTHON_RUN[@]}" -m pip install -U \
    "${CAPTION_TRANSFORMERS_SPEC}" \
    "${CAPTION_HF_HUB_SPEC}" \
    "${CAPTION_ACCELERATE_SPEC}" \
    "qwen-vl-utils[decord]" \
    pillow
fi

"${PYTHON_RUN[@]}" - <<'PY'
import importlib.metadata as md

for name in ("torch", "transformers", "huggingface_hub", "accelerate"):
    try:
        print(f"[caption] {name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"[caption] {name}=not-installed")
PY

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
echo "[caption] env=${PYTHON_ENV:-current shell python}"
echo "[caption] model=${CAPTION_MODEL}"
echo "[caption] output=${CAPTION_JSON}"

CUDA_VISIBLE_DEVICES="${CAPTION_GPU}" "${PYTHON_RUN[@]}" "${ARGS[@]}"
