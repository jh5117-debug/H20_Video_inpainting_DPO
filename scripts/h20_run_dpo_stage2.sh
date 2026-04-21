#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_HOME:-${DEFAULT_PROJECT_ROOT}}}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
PROJECT_PARENT="$(dirname "${PROJECT_ROOT}")"
DEFAULT_CONDA_ENV_PREFIX="${PROJECT_PARENT}/conda_envs/diffueraser"

if [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
  CONDA_ENV="${CONDA_ENV_PREFIX}"
elif [[ -z "${CONDA_ENV:-}" && -d "${DEFAULT_CONDA_ENV_PREFIX}" ]]; then
  CONDA_ENV="${DEFAULT_CONDA_ENV_PREFIX}"
else
  CONDA_ENV="${CONDA_ENV:-diffueraser}"
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES}" || "${CUDA_VISIBLE_DEVICES}" == "all" ]]; then
    NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  else
    IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS="${#_visible_gpus[@]}"
  fi
fi

if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -x "${PROJECT_PARENT}/miniconda3/bin/conda" ]]; then
  CONDA_BASE="${PROJECT_PARENT}/miniconda3"
else
  echo "conda not found; set CONDA_EXE or install Miniconda." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PROJECT_ROOT}/.hf_cache}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.tmp}"
mkdir -p "${HF_HOME}" "${TMPDIR}" logs

WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_ROOT}/data/external/DPO_Finetune_data}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${PROJECT_ROOT}/data/external/davis_432_240}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"
RUN_NAME="${RUN_NAME:-h20-stage2}"
RUN_VERSION="${RUN_VERSION:-$(date -u +%Y%m%d_%H%M%S)}"
export RUN_VERSION
PRETRAINED_DPO_S1="${PRETRAINED_DPO_S1:-}"
REF_MODEL_PATH="${REF_MODEL_PATH:-}"
BASELINE_UNET_PATH="${BASELINE_UNET_PATH:-${WEIGHTS_DIR}/diffuEraser}"
CHUNK_ALIGNED="${CHUNK_ALIGNED:-1}"
XFORMERS="${XFORMERS:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}"

sanitize_path_component() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-' | sed -e 's/^-*//' -e 's/-*$//'
}

DEFAULT_DPO_LOG_ROOT="/home/nvme03/workspace/world_model_phys/Diffueraser_DPO_Log"
DPO_LOG_ROOT="${DPO_LOG_ROOT:-${DEFAULT_DPO_LOG_ROOT}}"
if ! mkdir -p "${DPO_LOG_ROOT}" 2>/dev/null; then
  DPO_LOG_ROOT="${PROJECT_ROOT}/logs/Diffueraser_DPO_Log"
  mkdir -p "${DPO_LOG_ROOT}"
fi

RUN_NAME_SLUG="$(sanitize_path_component "${RUN_NAME}")"
DPO_RUN_LOG_DIR="${DPO_RUN_LOG_DIR:-${DPO_LOG_ROOT}/${RUN_VERSION}_${RUN_NAME_SLUG}}"
DPO_EXTERNAL_LOG_DIR="${DPO_EXTERNAL_LOG_DIR:-${DPO_RUN_LOG_DIR}/experiment}"
DPO_STDOUT_LOG="${DPO_STDOUT_LOG:-${DPO_RUN_LOG_DIR}/train_stdout.log}"
mkdir -p "${DPO_RUN_LOG_DIR}" "${DPO_EXTERNAL_LOG_DIR}"
export DPO_LOG_ROOT DPO_RUN_LOG_DIR DPO_EXTERNAL_LOG_DIR

export WANDB_DIR="${WANDB_DIR:-${DPO_RUN_LOG_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${DPO_RUN_LOG_DIR}/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${DPO_RUN_LOG_DIR}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${DPO_RUN_LOG_DIR}/wandb/config}"
mkdir -p "${WANDB_DIR}" "${WANDB_CONFIG_DIR}"

if [[ "${DPO_TEE_STDOUT:-1}" != "0" && -z "${DPO_STDOUT_TEE_ACTIVE:-}" ]]; then
  export DPO_STDOUT_TEE_ACTIVE=1
  exec > >(tee -a "${DPO_STDOUT_LOG}") 2>&1
fi

echo "[h20 launcher] External log root: ${DPO_LOG_ROOT}"
echo "[h20 launcher] Run log dir:      ${DPO_RUN_LOG_DIR}"
echo "[h20 launcher] Train stdout log: ${DPO_STDOUT_LOG}"

PRETRAINED_DPO_S1_ARG=()
if [[ -n "${PRETRAINED_DPO_S1}" ]]; then
  PRETRAINED_DPO_S1_ARG=(--pretrained_dpo_stage1 "${PRETRAINED_DPO_S1}")
fi

REF_MODEL_ARG=()
if [[ -n "${REF_MODEL_PATH}" ]]; then
  REF_MODEL_ARG=(--ref_model_path "${REF_MODEL_PATH}")
fi

RUN_VERSION_ARG=()
if [[ -n "${RUN_VERSION}" ]]; then
  RUN_VERSION_ARG=(--run_version "${RUN_VERSION}")
fi

CHUNK_ARG=()
if [[ "${CHUNK_ALIGNED}" == "1" || "${CHUNK_ALIGNED}" == "true" ]]; then
  CHUNK_ARG=(--chunk_aligned)
fi

XFORMERS_ARG=()
case "${XFORMERS,,}" in
  1|true|yes|on)
    XFORMERS_ARG=(--enable_xformers)
    ;;
esac

GRADIENT_CHECKPOINTING_ARG=()
case "${GRADIENT_CHECKPOINTING,,}" in
  0|false|no|off)
    GRADIENT_CHECKPOINTING_ARG=(--disable_gradient_checkpointing)
    ;;
esac

SPLIT_POS_NEG_FORWARD_ARG=()
case "${SPLIT_POS_NEG_FORWARD,,}" in
  1|true|yes|on)
    SPLIT_POS_NEG_FORWARD_ARG=(--split_pos_neg_forward)
    ;;
esac

python training/dpo/scripts/run_stage2.py \
  --num_gpus "${NUM_GPUS}" \
  --weights_dir "${WEIGHTS_DIR}" \
  --dpo_data_root "${DPO_DATA_ROOT}" \
  --val_data_dir "${VAL_DATA_DIR}" \
  --baseline_unet_path "${BASELINE_UNET_PATH}" \
  --experiments_dir "${EXPERIMENTS_DIR}" \
  --run_name "${RUN_NAME}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --learning_rate "${LR:-1e-6}" \
  --lr_scheduler "${LR_SCHEDULER:-constant}" \
  --lr_warmup_steps "${LR_WARMUP:-500}" \
  --max_train_steps "${MAX_STEPS:-30000}" \
  --checkpointing_steps "${CKPT_STEPS:-2000}" \
  --checkpoints_total_limit "${CKPT_LIMIT:-3}" \
  --validation_steps "${VAL_STEPS:-2000}" \
  --nframes "${NFRAMES:-16}" \
  --seed "${SEED:-42}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --wandb_project "${WANDB_PROJECT:-DPO_Diffueraser}" \
  --beta_dpo "${BETA_DPO:-500.0}" \
  --davis_oversample "${DAVIS_OVERSAMPLE:-10}" \
  "${CHUNK_ARG[@]}" \
  "${XFORMERS_ARG[@]}" \
  "${GRADIENT_CHECKPOINTING_ARG[@]}" \
  "${SPLIT_POS_NEG_FORWARD_ARG[@]}" \
  "${PRETRAINED_DPO_S1_ARG[@]}" \
  "${REF_MODEL_ARG[@]}" \
  "${RUN_VERSION_ARG[@]}" \
  "$@"
