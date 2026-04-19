#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${CONDA_ENV:-}" && -d "/home/nvme01/conda_envs/diffueraser" ]]; then
  CONDA_ENV="/home/nvme01/conda_envs/diffueraser"
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

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -x "/home/nvme01/miniconda3/bin/conda" ]]; then
  CONDA_BASE="/home/nvme01/miniconda3"
else
  echo "conda not found; set CONDA_EXE or install Miniconda." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/.wandb_cache}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_ROOT}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${PROJECT_ROOT}/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PROJECT_ROOT}/.wandb_cache/config}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PROJECT_ROOT}/.hf_cache}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.tmp}"
mkdir -p "${WANDB_DIR}" "${WANDB_CONFIG_DIR}" "${HF_HOME}" "${TMPDIR}" logs

WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_ROOT}/data/external/DPO_Finetune_data}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${PROJECT_ROOT}/data/external/davis_432_240}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"
RUN_NAME="${RUN_NAME:-h20-stage1}"
RUN_VERSION="${RUN_VERSION:-}"
REF_MODEL_PATH="${REF_MODEL_PATH:-}"
CHUNK_ALIGNED="${CHUNK_ALIGNED:-1}"

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

python training/dpo/scripts/run_stage1.py \
  --num_gpus "${NUM_GPUS}" \
  --weights_dir "${WEIGHTS_DIR}" \
  --dpo_data_root "${DPO_DATA_ROOT}" \
  --val_data_dir "${VAL_DATA_DIR}" \
  --experiments_dir "${EXPERIMENTS_DIR}" \
  --run_name "${RUN_NAME}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --learning_rate "${LR:-1e-6}" \
  --lr_scheduler "${LR_SCHEDULER:-constant}" \
  --lr_warmup_steps "${LR_WARMUP:-500}" \
  --max_train_steps "${MAX_STEPS:-20000}" \
  --checkpointing_steps "${CKPT_STEPS:-2000}" \
  --checkpoints_total_limit "${CKPT_LIMIT:-3}" \
  --validation_steps "${VAL_STEPS:-2000}" \
  --nframes "${NFRAMES:-16}" \
  --seed "${SEED:-42}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --wandb_project "${WANDB_PROJECT:-DPO_Diffueraser}" \
  --beta_dpo "${BETA_DPO:-500.0}" \
  --davis_oversample "${DAVIS_OVERSAMPLE:-10}" \
  "${CHUNK_ARG[@]}" \
  "${REF_MODEL_ARG[@]}" \
  "${RUN_VERSION_ARG[@]}" \
  "$@"
