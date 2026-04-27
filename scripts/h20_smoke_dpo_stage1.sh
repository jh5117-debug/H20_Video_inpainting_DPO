#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_HOME:-${DEFAULT_PROJECT_ROOT}}}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
if [[ -z "${DPO_DATA_ROOT:-}" ]]; then
  if [[ -d "${PROJECT_ROOT}/DPO_Finetune_Data_Multimodel_v1" ]]; then
    export DPO_DATA_ROOT="${PROJECT_ROOT}/DPO_Finetune_Data_Multimodel_v1"
  else
    export DPO_DATA_ROOT="${PROJECT_ROOT}/data/external/DPO_Finetune_data"
  fi
fi
export VAL_DATA_DIR="${VAL_DATA_DIR:-${PROJECT_ROOT}/data/external/davis_432_240}"
export RUN_NAME="${RUN_NAME:-h20-dpo-stage1-smoke}"
export RUN_VERSION="${RUN_VERSION:-$(date -u +%Y%m%d_%H%M%S)}"
export MAX_STEPS="${MAX_STEPS:-5}"
export CKPT_STEPS="${CKPT_STEPS:-5}"
export VAL_STEPS="${VAL_STEPS:-5}"
export CKPT_LIMIT="${CKPT_LIMIT:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MIXED_PRECISION="${MIXED_PRECISION:-no}"
export XFORMERS="${XFORMERS:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}"
export VAL_NUM_INFERENCE_STEPS="${VAL_NUM_INFERENCE_STEPS:-6}"
export VAL_MASK_DILATION_ITER="${VAL_MASK_DILATION_ITER:-0}"

echo "[smoke] project=${PROJECT_ROOT}"
echo "[smoke] run_name=${RUN_NAME}"
echo "[smoke] run_version=${RUN_VERSION}"
echo "[smoke] data=${DPO_DATA_ROOT}"
echo "[smoke] val=${VAL_DATA_DIR}"
echo "[smoke] gpus=${CUDA_VISIBLE_DEVICES}"
echo "[smoke] steps=${MAX_STEPS} val_steps=${VAL_STEPS} ckpt_steps=${CKPT_STEPS}"
echo "[smoke] val_num_inference_steps=${VAL_NUM_INFERENCE_STEPS} val_mask_dilation=${VAL_MASK_DILATION_ITER}"

exec bash "${PROJECT_ROOT}/scripts/h20_run_dpo_stage1.sh" "$@"
