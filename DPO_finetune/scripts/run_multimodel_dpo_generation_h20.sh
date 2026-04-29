#!/usr/bin/env bash
set -Eeuo pipefail

# One-command entry for generating a model-output-based DPO dataset on H20.
# It keeps the training-side schema unchanged:
#   manifest.json + {video}/gt_frames,masks,neg_frames_1,neg_frames_2,meta.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
DATA_NAME="${DATA_NAME:-Video_inpainting_DPO}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"
DATA="${DATA:-${PROJECT_ROOT}/data}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${DATA}/third_party_video_inpainting}"
PROJECT_THIRD_PARTY_ROOT="${PROJECT_ROOT}/third_party_video_inpainting"
OUT_ROOT="${OUT_ROOT:-${DATA}/DPO_Finetune_Data_Multimodel_v1}"
YTBV_ROOT="${YTBV_ROOT:-${DATA}/external/ytbv_2019_full_resolution/train/JPEGImages}"
DAVIS_ROOT="${DAVIS_ROOT:-${DATA}/external/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution}"
ADAPTER_CONFIG="${ADAPTER_CONFIG:-${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.json}"
CAPTION_JSON="${CAPTION_JSON:-}"

mkdir -p "${OUT_ROOT}"
if [[ ! -f "${ADAPTER_CONFIG}" && -f "${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.example.json" ]]; then
  echo "[run] adapter config missing; copying example to ${ADAPTER_CONFIG}"
  cp "${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.example.json" "${ADAPTER_CONFIG}"
fi

DEFAULT_GPUS="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_GPUS}}"

# Some older H20 setups kept code/envs under ${PROJECT_ROOT}/third_party_video_inpainting
# while the data archive stores weights under ${DATA}/third_party_video_inpainting.
# Bridge them so a single THIRD_PARTY_ROOT works for adapters, scoring, and Slurm.
mkdir -p "${THIRD_PARTY_ROOT}"
if [[ "${THIRD_PARTY_ROOT}" != "${PROJECT_THIRD_PARTY_ROOT}" && -d "${PROJECT_THIRD_PARTY_ROOT}" ]]; then
  for subdir in repos envs weights; do
    if [[ ! -e "${THIRD_PARTY_ROOT}/${subdir}" && -e "${PROJECT_THIRD_PARTY_ROOT}/${subdir}" ]]; then
      ln -s "${PROJECT_THIRD_PARTY_ROOT}/${subdir}" "${THIRD_PARTY_ROOT}/${subdir}"
    fi
  done
fi

export VBENCH_ROOT="${VBENCH_ROOT:-${THIRD_PARTY_ROOT}/repos/VBench}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${THIRD_PARTY_ROOT}/weights/vbench_cache}"
export PYTHONPATH="${PROJECT_ROOT}:${VBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
VBENCH_DIMENSIONS="${VBENCH_DIMENSIONS:-subject_consistency,background_consistency,temporal_flickering,motion_smoothness,aesthetic_quality}"

remove_vbench_dimension() {
  local dims="$1"
  local remove="$2"
  python - "${dims}" "${remove}" <<'PY'
import sys
dims = [d.strip() for d in sys.argv[1].split(",") if d.strip()]
remove = sys.argv[2]
print(",".join(d for d in dims if d != remove))
PY
}

if [[ "${ENABLE_VBENCH:-0}" == "1" ]]; then
  echo "[run] verify VBench scoring deps in ${DIFFUERASER_ENV}"
  if [[ ! -d "${VBENCH_ROOT}/vbench" ]]; then
    echo "[run][error] VBench repo not found at ${VBENCH_ROOT}." >&2
    echo "[run][error] Run DPO_finetune/scripts/setup_multimodel_h20.sh or set VBENCH_ROOT." >&2
    exit 1
  fi
  if ! PYTHONNOUSERSITE=1 conda run --no-capture-output -p "${DIFFUERASER_ENV}" \
    python -c "import decord, omegaconf, vbench" >/dev/null 2>&1; then
    if [[ "${VBENCH_AUTO_INSTALL_DEPS:-1}" != "1" ]]; then
      echo "[run][error] VBench requires decord, omegaconf, and the VBench repo in ${DIFFUERASER_ENV}/PYTHONPATH." >&2
      echo "[run][error] Set VBENCH_AUTO_INSTALL_DEPS=1 or install deps, and check VBENCH_ROOT=${VBENCH_ROOT}." >&2
      exit 1
    fi
    echo "[run] install VBench scoring deps: decord omegaconf"
    PYTHONNOUSERSITE=1 conda run --no-capture-output -p "${DIFFUERASER_ENV}" \
      python -m pip install "decord==0.6.0" "omegaconf>=2.3.0"
    PYTHONNOUSERSITE=1 conda run --no-capture-output -p "${DIFFUERASER_ENV}" \
      python -c "import vbench"
  fi

  mkdir -p "${VBENCH_CACHE_DIR}"
  if [[ ",${VBENCH_DIMENSIONS}," == *",motion_smoothness,"* && ! -s "${VBENCH_CACHE_DIR}/amt_model/amt-s.pth" ]]; then
    echo "[run] predownload VBench AMT weight for motion_smoothness"
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    PYTHONNOUSERSITE=1 conda run --no-capture-output -p "${DIFFUERASER_ENV}" python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

cache_dir = Path(os.environ["VBENCH_CACHE_DIR"])
dst = cache_dir / "amt_model" / "amt-s.pth"
dst.parent.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="lalala125/AMT",
    filename="amt-s.pth",
    repo_type="model",
    local_dir=str(dst.parent),
    local_dir_use_symlinks=False,
)
src = Path(path)
if src.resolve() != dst.resolve():
    shutil.copy2(src, dst)
if not dst.exists() or dst.stat().st_size <= 0:
    raise RuntimeError(f"AMT download failed or empty: {dst}")
print(f"[run] VBench AMT ready: {dst} ({dst.stat().st_size} bytes)")
PY
  fi

  if [[ ",${VBENCH_DIMENSIONS}," == *",aesthetic_quality,"* ]]; then
    mkdir -p "${VBENCH_CACHE_DIR}/aesthetic_model/emb_reader"
    if [[ ! -s "${VBENCH_CACHE_DIR}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" \
      && -s "${PROJECT_ROOT}/weights/metrics/sa_0_4_vit_l_14_linear.pth" ]]; then
      echo "[run] reuse local VBench aesthetic weight"
      cp "${PROJECT_ROOT}/weights/metrics/sa_0_4_vit_l_14_linear.pth" \
        "${VBENCH_CACHE_DIR}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
    fi
  fi

  if [[ ",${VBENCH_DIMENSIONS}," == *",imaging_quality,"* ]]; then
    mkdir -p "${VBENCH_CACHE_DIR}/pyiqa_model"
    if [[ ! -s "${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" \
      && -s "${PROJECT_ROOT}/weights/metrics/musiq_spaq_ckpt-358bb6af.pth" ]]; then
      echo "[run] reuse local VBench MUSIQ/pyiqa weight"
      cp "${PROJECT_ROOT}/weights/metrics/musiq_spaq_ckpt-358bb6af.pth" \
        "${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
    fi
    if [[ ! -s "${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" \
      && -n "${VBENCH_PYIQA_HF_REPO:-}" ]]; then
      echo "[run] download VBench MUSIQ/pyiqa weight from ${VBENCH_PYIQA_HF_REPO}"
      HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
      HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
      PYTHONNOUSERSITE=1 conda run --no-capture-output -p "${DIFFUERASER_ENV}" python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

cache_dir = Path(os.environ["VBENCH_CACHE_DIR"])
dst = cache_dir / "pyiqa_model" / "musiq_spaq_ckpt-358bb6af.pth"
dst.parent.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id=os.environ["VBENCH_PYIQA_HF_REPO"],
    filename=os.environ.get("VBENCH_PYIQA_HF_FILENAME", "musiq_spaq_ckpt-358bb6af.pth"),
    repo_type=os.environ.get("VBENCH_PYIQA_HF_REPO_TYPE", "dataset"),
    local_dir=str(dst.parent),
    local_dir_use_symlinks=False,
)
src = Path(path)
if src.resolve() != dst.resolve():
    shutil.copy2(src, dst)
if not dst.exists() or dst.stat().st_size <= 0:
    raise RuntimeError(f"MUSIQ download failed or empty: {dst}")
print(f"[run] VBench MUSIQ ready: {dst} ({dst.stat().st_size} bytes)")
PY
    fi
    if [[ ! -s "${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" ]]; then
      if [[ "${VBENCH_REQUIRE_IMAGING:-0}" == "1" ]]; then
        echo "[run][error] imaging_quality requires musiq_spaq_ckpt-358bb6af.pth." >&2
        echo "[run][error] Put it at ${PROJECT_ROOT}/weights/metrics/musiq_spaq_ckpt-358bb6af.pth or set VBENCH_PYIQA_HF_REPO." >&2
        exit 1
      fi
      echo "[run][warn] skip VBench imaging_quality: missing musiq_spaq_ckpt-358bb6af.pth and GitHub release is unreliable on H20"
      VBENCH_DIMENSIONS="$(remove_vbench_dimension "${VBENCH_DIMENSIONS}" "imaging_quality")"
    fi
  fi
fi

ARGS=(
  "${PROJECT_ROOT}/DPO_finetune/generate_multimodel_dpo_dataset.py"
  --ytbv_root "${YTBV_ROOT}"
  --davis_root "${DAVIS_ROOT}"
  --output_root "${OUT_ROOT}"
  --third_party_root "${THIRD_PARTY_ROOT}"
  --adapter_config "${ADAPTER_CONFIG}"
  --diffueraser_env "${DIFFUERASER_ENV}"
  --methods "${METHODS:-propainter,cococo,diffueraser,minimax}"
  --gpus "${GPUS:-${DEFAULT_GPUS}}"
  --num_videos "${NUM_VIDEOS:-0}"
  --max_frames "${MAX_FRAMES:-48}"
  --height "${HEIGHT:-512}"
  --width "${WIDTH:-512}"
  --train_nframes "${TRAIN_NFRAMES:-16}"
  --score_windows "${SCORE_WINDOWS:-32,24,16}"
  --mask_seeds_per_video "${MASK_SEEDS_PER_VIDEO:-1}"
  --mask_dilation_iter "${MASK_DILATION_ITER:-0}"
  --mask_area_min "${MASK_AREA_MIN:-0.20}"
  --mask_area_max "${MASK_AREA_MAX:-0.30}"
  --mask_margin_ratio "${MASK_MARGIN_RATIO:-0.10}"
  --mask_static_prob "${MASK_STATIC_PROB:-0.50}"
  --mask_speed_min "${MASK_SPEED_MIN:-0.50}"
  --mask_speed_max "${MASK_SPEED_MAX:-1.50}"
  --mask_center_jitter_ratio "${MASK_CENTER_JITTER_RATIO:-0.04}"
  --mask_motion_box_ratio "${MASK_MOTION_BOX_RATIO:-0.16}"
  --source_selection_weights "${SOURCE_SELECTION_WEIGHTS:-propainter=1.5,cococo=1.0,diffueraser=1.0,minimax=1.0}"
  --source_quality_max_overrides "${SOURCE_QUALITY_MAX_OVERRIDES:-propainter=0.98}"
  --neg_quality_min "${NEG_QUALITY_MIN:-0.30}"
  --neg_quality_max "${NEG_QUALITY_MAX:-0.65}"
  --neg_quality_target "${NEG_QUALITY_TARGET:-0.45}"
  --vbench_dimensions "${VBENCH_DIMENSIONS}"
  --parallel_methods "${PARALLEL_METHODS:-4}"
  --parallel_videos "${PARALLEL_VIDEOS:-1}"
  --gpu_slots "${GPU_SLOTS:-1}"
  --resume
)

if [[ -n "${CAPTION_JSON}" ]]; then
  ARGS+=(--caption_json "${CAPTION_JSON}")
fi
if [[ -n "${METHOD_GPU_MAP:-}" ]]; then
  ARGS+=(--method_gpu_map "${METHOD_GPU_MAP}")
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
ARGS+=(--candidate_retention "${CANDIDATE_RETENTION:-all}")
if [[ "${CLEANUP_FAILED:-0}" == "1" ]]; then
  ARGS+=(--cleanup_failed)
fi
if [[ "${SKIP_INFERENCE:-0}" == "1" ]]; then
  ARGS+=(--skip_inference)
fi

echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] output_root=${OUT_ROOT}"
echo "[run] adapter_config=${ADAPTER_CONFIG}"

PYTHONNOUSERSITE=1 conda run --no-capture-output -p "${DIFFUERASER_ENV}" python "${ARGS[@]}"

cat <<EOF

[done] DPO data root:
  ${OUT_ROOT}

Use it for training with:
  DPO_DATA_ROOT=${OUT_ROOT} bash scripts/h20_run_dpo_stage1.sh
EOF
