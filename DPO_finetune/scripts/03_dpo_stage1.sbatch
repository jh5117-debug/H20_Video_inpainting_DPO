#!/bin/bash
#SBATCH --job-name=DPO_S1
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dpo-stage1-%j.out

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ====================== Editable SC Defaults ======================
# Repo checkout path: ${PROJECT_DEV}/${PROJECT_NAME}
# Data/root assets path: ${PROJECT_DATA}/${DATA_NAME}
PROJECT_NAME="Video_inpainting_DPO"
DATA_NAME="Video_inpainting_DPO"

MIXED_PRECISION="bf16"
VAL_NUM_INFERENCE_STEPS=6
VAL_MASK_DILATION_ITER=0
NUM_GPUS=8
BATCH_SIZE=1
GRAD_ACCUM=1
MAX_STEPS=20000
CKPT_STEPS=2000
VAL_STEPS=2000
CKPT_LIMIT=1

RUN_NAME="sc-dpo-stage1-newdata-bf16-raw6-nodil"
CHUNK_ALIGNED=1
SPLIT_POS_NEG_FORWARD=1
GRADIENT_CHECKPOINTING=1
XFORMERS=0
BETA_DPO=500.0
SFT_REG_WEIGHT=0.0
WANDB_ENTITY="jh5117-columbia-university"
WANDB_PROJECT="DPO_Diffueraser"
# ================================================================

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PROJECT_DEV:-}" ]]; then
    export PROJECT_ROOT="${PROJECT_DEV}/${PROJECT_NAME}"
  elif [[ -n "${PROJECT_HOME:-}" ]]; then
    export PROJECT_ROOT="${PROJECT_HOME}/dev/${PROJECT_NAME}"
  else
    export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
export PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_DEV}/data}"
export DATA="${PROJECT_DATA}/${DATA_NAME}"
export DATA_NAME
export WEIGHTS_DIR="${WEIGHTS_DIR:-${DATA}/weights}"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${DATA}/DPO_Finetune_data}"
export VAL_DATA_DIR="${VAL_DATA_DIR:-${DATA}/davis_432_240}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${DATA}/experiments}"
export WANDB_PROJECT WANDB_ENTITY
export MIXED_PRECISION VAL_NUM_INFERENCE_STEPS VAL_MASK_DILATION_ITER
export NUM_GPUS BATCH_SIZE GRAD_ACCUM MAX_STEPS CKPT_STEPS VAL_STEPS CKPT_LIMIT
export RUN_NAME CHUNK_ALIGNED SPLIT_POS_NEG_FORWARD GRADIENT_CHECKPOINTING XFORMERS
export BETA_DPO SFT_REG_WEIGHT
exec bash "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" "$@"
