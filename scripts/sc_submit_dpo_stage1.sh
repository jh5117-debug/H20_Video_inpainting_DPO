#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME:+${PROJECT_HOME}/dev}}"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DEV:+${PROJECT_DEV}/H20_Video_inpainting_DPO}}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

if [[ -f "${PROJECT_ROOT}/env.sc.sh" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/env.sc.sh"
fi

if [[ -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
  echo "WANDB_API_KEY is not set and ~/.netrc is missing; W&B visibility is required." >&2
  exit 1
fi

export PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
export PROJECT_ROOT
export WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser}"
export WANDB_ENTITY="${WANDB_ENTITY:-jh5117-columbia-university}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_ROOT}/data}"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_DATA}/DPO_Finetune_data}"
export VAL_DATA_DIR="${VAL_DATA_DIR:-${PROJECT_DATA}/davis_432_240}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"
export RUN_NAME="${RUN_NAME:-sc-dpo-stage1}"
export RUN_VERSION="${RUN_VERSION:-$(date -u +%Y%m%d_%H%M%S)}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export LR="${LR:-1e-6}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export CKPT_STEPS="${CKPT_STEPS:-2000}"
export VAL_STEPS="${VAL_STEPS:-2000}"
export CKPT_LIMIT="${CKPT_LIMIT:-3}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export CHUNK_ALIGNED="${CHUNK_ALIGNED:-1}"
export SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}"
export XFORMERS="${XFORMERS:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export BETA_DPO="${BETA_DPO:-500.0}"
export SFT_REG_WEIGHT="${SFT_REG_WEIGHT:-0.0}"
export VAL_NUM_INFERENCE_STEPS="${VAL_NUM_INFERENCE_STEPS:-6}"
export VAL_MASK_DILATION_ITER="${VAL_MASK_DILATION_ITER:-0}"

mkdir -p "${PROJECT_ROOT}/logs"

echo "[sc-submit] project_root=${PROJECT_ROOT}"
echo "[sc-submit] project_home=${PROJECT_HOME}"
echo "[sc-submit] project_dev=${PROJECT_DEV}"
echo "[sc-submit] project_data=${PROJECT_DATA}"
echo "[sc-submit] run_name=${RUN_NAME}"
echo "[sc-submit] run_version=${RUN_VERSION}"
echo "[sc-submit] data=${DPO_DATA_ROOT}"
echo "[sc-submit] val=${VAL_DATA_DIR}"
echo "[sc-submit] weights=${WEIGHTS_DIR}"
echo "[sc-submit] wandb_project=${WANDB_PROJECT}"
echo "[sc-submit] wandb_entity=${WANDB_ENTITY}"
echo "[sc-submit] val_num_inference_steps=${VAL_NUM_INFERENCE_STEPS}"
echo "[sc-submit] val_mask_dilation=${VAL_MASK_DILATION_ITER}"

exec sbatch "${PROJECT_ROOT}/DPO_finetune/scripts/03_dpo_stage1.sbatch"
