#!/bin/bash
#SBATCH --job-name=DPO_GEN_NB
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=47:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/dpo-generate-newband-%j.out
#SBATCH --open-mode=append

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ====================== Editable SC Defaults ======================
# Repo checkout path: ${PROJECT_DEV}/${PROJECT_NAME}
# Data/root assets path: ${PROJECT_DATA}/${DATA_NAME}
PROJECT_NAME="Video_inpainting_DPO"
DATA_NAME="Video_inpainting_DPO"

OUT_DATASET_NAME="DPO_Finetune_Data_Multimodel_mask20_30_q30_65_nodil_v1"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS="0,1,2,3,4,5,6,7"
METHODS="propainter,cococo,diffueraser,minimax"
NUM_VIDEOS=0
MAX_FRAMES=48
HEIGHT=512
WIDTH=512
TRAIN_NFRAMES=16
SCORE_WINDOWS="32,24,16"

MASK_DILATION_ITER=0
MASK_AREA_MIN=0.20
MASK_AREA_MAX=0.30
MASK_MARGIN_RATIO=0.10

NEG_QUALITY_MIN=0.30
NEG_QUALITY_MAX=0.65
NEG_QUALITY_TARGET=0.45

PARALLEL_METHODS=4
PARALLEL_VIDEOS=8
GPU_SLOTS=2
METHOD_GPU_MAP="cococo=0,1,2,3;diffueraser=4,5,6,7;minimax=0,1,2,3,4,5,6,7;propainter=0,1,2,3,4,5,6,7"

ENABLE_LPIPS=1
ENABLE_VBENCH=1
SAVE_PREVIEWS=0
CANDIDATE_RETENTION="selected"
CLEANUP_FAILED=1
AUTO_REQUEUE=1
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
export PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

if [[ -f "${PROJECT_ROOT}/env.sc.sh" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/env.sc.sh"
fi

export PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_DEV}/data}"
export DATA="${PROJECT_DATA}/${DATA_NAME}"
export DATA_NAME
export THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${DATA}/third_party_video_inpainting}"
export DIFFUERASER_ENV="${DIFFUERASER_ENV:-${CONDA_ENV_PREFIX:-${PROJECT_DEV}/conda_envs/diffueraser}}"
export ADAPTER_CONFIG="${ADAPTER_CONFIG:-${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.json}"
export OUT_ROOT="${DATA}/${OUT_DATASET_NAME}"

pick_first_dir() {
  for p in "$@"; do
    if [[ -d "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

export DAVIS_ROOT="${DAVIS_ROOT:-$(pick_first_dir \
  "${DATA}/external/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution" \
  "${DATA}/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution" \
  "${DATA}/external/davis_432_240/JPEGImages_432_240" \
  "${DATA}/davis_432_240/JPEGImages_432_240" \
  || true)}"
export YTBV_ROOT="${YTBV_ROOT:-$(pick_first_dir \
  "${DATA}/external/ytbv_2019_full_resolution/train/JPEGImages" \
  "${DATA}/ytbv_2019_full_resolution/train/JPEGImages" \
  "${DATA}/external/youtubevos_432_240/JPEGImages_432_240" \
  "${DATA}/youtubevos_432_240/JPEGImages_432_240" \
  || true)}"

if [[ -z "${DAVIS_ROOT}" || -z "${YTBV_ROOT}" ]]; then
  echo "[generate][error] DAVIS_ROOT or YTBV_ROOT not found." >&2
  echo "[generate][error] PROJECT_DATA=${PROJECT_DATA}" >&2
  echo "[generate][error] DAVIS_ROOT=${DAVIS_ROOT}" >&2
  echo "[generate][error] YTBV_ROOT=${YTBV_ROOT}" >&2
  exit 1
fi

if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [[ -f "${PROJECT_DEV}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${PROJECT_DEV}/miniconda3/etc/profile.d/conda.sh"
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUT_ROOT}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES GPUS METHODS NUM_VIDEOS MAX_FRAMES HEIGHT WIDTH TRAIN_NFRAMES SCORE_WINDOWS
export MASK_DILATION_ITER MASK_AREA_MIN MASK_AREA_MAX MASK_MARGIN_RATIO
export NEG_QUALITY_MIN NEG_QUALITY_MAX NEG_QUALITY_TARGET
export PARALLEL_METHODS PARALLEL_VIDEOS GPU_SLOTS METHOD_GPU_MAP
export ENABLE_LPIPS ENABLE_VBENCH SAVE_PREVIEWS CANDIDATE_RETENTION CLEANUP_FAILED AUTO_REQUEUE

GEN_PID=""
on_usr1() {
  echo "[generate] received USR1 before Slurm time limit; stopping child so completed samples stay resumable"
  if [[ -n "${GEN_PID}" ]]; then
    kill -TERM "${GEN_PID}" 2>/dev/null || true
    wait "${GEN_PID}" 2>/dev/null || true
  fi
  if [[ "${AUTO_REQUEUE:-1}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[generate] attempting scontrol requeue ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}" || echo "[generate][warn] requeue failed; resubmit this sbatch manually to resume"
  fi
  exit 0
}
trap on_usr1 USR1

echo "[generate] project_root=${PROJECT_ROOT}"
echo "[generate] project_data=${PROJECT_DATA}"
echo "[generate] data=${DATA}"
echo "[generate] output_root=${OUT_ROOT}"
echo "[generate] davis_root=${DAVIS_ROOT}"
echo "[generate] ytbv_root=${YTBV_ROOT}"
echo "[generate] diffueraser_env=${DIFFUERASER_ENV}"
echo "[generate] gpus=${GPUS} gpu_slots=${GPU_SLOTS}"
echo "[generate] parallel_videos=${PARALLEL_VIDEOS} parallel_methods=${PARALLEL_METHODS}"
echo "[generate] method_gpu_map=${METHOD_GPU_MAP}"
echo "[generate] resume=enabled"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true

bash "${PROJECT_ROOT}/DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh" &
GEN_PID=$!
wait "${GEN_PID}"
