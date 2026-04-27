#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_HOME:-${DEFAULT_PROJECT_ROOT}}}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

if [[ -f "${PROJECT_ROOT}/env.sc.sh" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/env.sc.sh"
fi

if [[ -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
  echo "WANDB_API_KEY is not set and ~/.netrc is missing; W&B visibility is required." >&2
  exit 1
fi

export PROJECT_HOME="${PROJECT_ROOT}"
export PROJECT_ROOT
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

mkdir -p "${PROJECT_ROOT}/logs"

echo "[sc-submit] project_root=${PROJECT_ROOT}"
echo "[sc-submit] run_name=${RUN_NAME}"
echo "[sc-submit] run_version=${RUN_VERSION}"
echo "[sc-submit] data=${DPO_DATA_ROOT:-${PROJECT_ROOT}/data/external/DPO_Finetune_data}"
echo "[sc-submit] val=${VAL_DATA_DIR:-${PROJECT_ROOT}/data/external/davis_432_240}"
echo "[sc-submit] weights=${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
echo "[sc-submit] wandb_project=${WANDB_PROJECT:-DPO_Diffueraser}"

exec sbatch "${PROJECT_ROOT}/DPO_finetune/scripts/03_dpo_stage1.sbatch"
