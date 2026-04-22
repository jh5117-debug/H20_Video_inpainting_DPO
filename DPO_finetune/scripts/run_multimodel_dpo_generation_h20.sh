#!/usr/bin/env bash
set -Eeuo pipefail

# One-command entry for generating a model-output-based DPO dataset on H20.
# It keeps the training-side schema unchanged:
#   manifest.json + {video}/gt_frames,masks,neg_frames_1,neg_frames_2,meta.json

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${PROJECT_ROOT}/third_party_video_inpainting}"
OUT_ROOT="${OUT_ROOT:-/home/nvme03/workspace/world_model_phys/DPO_Finetune_Data_Multimodel_v1}"
YTBV_ROOT="${YTBV_ROOT:-${PROJECT_ROOT}/data/external/ytbv_2019_full_resolution/train/JPEGImages}"
DAVIS_ROOT="${DAVIS_ROOT:-${PROJECT_ROOT}/data/external/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution}"
ADAPTER_CONFIG="${ADAPTER_CONFIG:-${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.json}"
CAPTION_JSON="${CAPTION_JSON:-}"

mkdir -p "${OUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VBENCH_ROOT="${VBENCH_ROOT:-${THIRD_PARTY_ROOT}/repos/VBench}"

ARGS=(
  "${PROJECT_ROOT}/DPO_finetune/generate_multimodel_dpo_dataset.py"
  --ytbv_root "${YTBV_ROOT}"
  --davis_root "${DAVIS_ROOT}"
  --output_root "${OUT_ROOT}"
  --third_party_root "${THIRD_PARTY_ROOT}"
  --adapter_config "${ADAPTER_CONFIG}"
  --methods "${METHODS:-propainter,cococo,minimax}"
  --gpus "${GPUS:-0,1,2,3}"
  --num_videos "${NUM_VIDEOS:-0}"
  --max_frames "${MAX_FRAMES:-48}"
  --height "${HEIGHT:-512}"
  --width "${WIDTH:-512}"
  --train_nframes "${TRAIN_NFRAMES:-16}"
  --score_windows "${SCORE_WINDOWS:-32,24,16}"
  --mask_seeds_per_video "${MASK_SEEDS_PER_VIDEO:-1}"
  --mask_dilation_iter "${MASK_DILATION_ITER:-8}"
  --parallel_methods "${PARALLEL_METHODS:-4}"
  --resume
)

if [[ -n "${CAPTION_JSON}" ]]; then
  ARGS+=(--caption_json "${CAPTION_JSON}")
fi
if [[ "${ENABLE_LPIPS:-1}" == "1" ]]; then
  ARGS+=(--enable_lpips)
fi
if [[ "${ENABLE_VBENCH:-0}" == "1" ]]; then
  ARGS+=(--enable_vbench)
fi
if [[ "${SAVE_PREVIEWS:-0}" == "1" ]]; then
  ARGS+=(--save_previews)
fi
if [[ "${SKIP_INFERENCE:-0}" == "1" ]]; then
  ARGS+=(--skip_inference)
fi

echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] output_root=${OUT_ROOT}"
echo "[run] adapter_config=${ADAPTER_CONFIG}"

conda run -p "${DIFFUERASER_ENV}" python "${ARGS[@]}"

cat <<EOF

[done] DPO data root:
  ${OUT_ROOT}

Use it for training with:
  DPO_DATA_ROOT=${OUT_ROOT} bash scripts/h20_run_dpo_stage1.sh
EOF
