#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare third-party video-inpainting repos, environment manifests, and
# weight directories on the H20 machine. This script is intentionally
# idempotent: running it again updates repos and reprints missing pieces.

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${PROJECT_ROOT}/third_party_video_inpainting}"
REPOS_ROOT="${THIRD_PARTY_ROOT}/repos"
WEIGHTS_ROOT="${THIRD_PARTY_ROOT}/weights"
ENVS_ROOT="${THIRD_PARTY_ROOT}/envs"
LOG_ROOT="${THIRD_PARTY_ROOT}/logs"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"
COCOCO_TORCH_INDEX_URL="${COCOCO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
COCOCO_TORCH_PACKAGES="${COCOCO_TORCH_PACKAGES:-torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0}"

mkdir -p "${REPOS_ROOT}" "${WEIGHTS_ROOT}" "${ENVS_ROOT}" "${LOG_ROOT}" "${THIRD_PARTY_ROOT}/manifests"

clone_or_update() {
  local url="$1"
  local name="$2"
  local dst="${REPOS_ROOT}/${name}"
  if [[ -d "${dst}/.git" ]]; then
    echo "[repo] update ${name}"
    git -C "${dst}" pull --ff-only || true
  else
    echo "[repo] clone ${name}"
    git clone "${url}" "${dst}"
  fi
}

clone_or_update "https://github.com/sczhou/ProPainter.git" "ProPainter"
clone_or_update "https://github.com/zibojia/COCOCO.git" "COCOCO"
clone_or_update "https://github.com/zibojia/MiniMax-Remover.git" "MiniMax-Remover"
clone_or_update "https://github.com/lixiaowen-xw/DiffuEraser.git" "DiffuEraser"
clone_or_update "https://github.com/Vchitect/VBench.git" "VBench"

echo "[env] export current DiffuEraser environment manifests"
if [[ -x "/home/nvme01/miniconda3/bin/conda" ]]; then
  /home/nvme01/miniconda3/bin/conda env export -p "${DIFFUERASER_ENV}" > "${THIRD_PARTY_ROOT}/manifests/diffueraser_h20_env.yml" || true
  /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${DIFFUERASER_ENV}" python -m pip freeze > "${THIRD_PARTY_ROOT}/manifests/diffueraser_h20_pip_freeze.txt" || true
fi

if [[ "${CREATE_THIRD_PARTY_ENVS:-0}" == "1" ]]; then
  echo "[env] create third-party envs when requirements.txt exists"
  for pair in "COCOCO:cococo" "MiniMax-Remover:minimax" "ProPainter:propainter" "VBench:vbench"; do
    repo="${pair%%:*}"
    env_name="${pair##*:}"
    repo_dir="${REPOS_ROOT}/${repo}"
    env_dir="${ENVS_ROOT}/${env_name}"
    if [[ ! -d "${env_dir}" && -f "${repo_dir}/requirements.txt" ]]; then
      echo "  create ${env_name}: ${env_dir}"
      /home/nvme01/miniconda3/bin/conda create -p "${env_dir}" python=3.10 -y
      PYTHONNOUSERSITE=1 /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${env_dir}" python -m pip install -U pip wheel setuptools
      PYTHONNOUSERSITE=1 /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${env_dir}" python -m pip install -r "${repo_dir}/requirements.txt" || true
    else
      echo "  skip ${env_name} (exists or no requirements.txt)"
    fi
  done
else
  echo "[env] skip third-party env creation (set CREATE_THIRD_PARTY_ENVS=1 to enable)"
fi

if [[ -x "/home/nvme01/miniconda3/bin/conda" && -d "${ENVS_ROOT}/cococo" ]]; then
  echo "[env] verify COCOCO runtime extras"
  if ! PYTHONNOUSERSITE=1 /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${ENVS_ROOT}/cococo" \
    python -c "import cv2, numpy, diffusers, transformers, huggingface_hub; assert int(numpy.__version__.split('.')[0]) < 2; assert tuple(map(int, huggingface_hub.__version__.split('.')[:2])) < (0, 26); from diffusers import AutoencoderKL, DDIMScheduler; from diffusers.utils import WEIGHTS_NAME; from transformers import CLIPTextModel, CLIPTokenizer" >/dev/null 2>&1; then
    echo "  install cococo extras: numpy<2 opencv-python-headless compatible diffusers stack"
    PYTHONNOUSERSITE=1 /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${ENVS_ROOT}/cococo" \
      python -m pip install \
      "numpy<2" \
      "opencv-python-headless<4.11" \
      "huggingface_hub<0.26" \
      "diffusers==0.11.1" \
      "transformers==4.25.1" \
      "safetensors" \
      "einops" \
      "omegaconf" \
      "imageio==2.34.0" \
      "imageio-ffmpeg==0.4.9" \
      "decord==0.6.0" \
      "wandb"
  else
    echo "  cococo extras ok: cv2 + numpy<2 + compatible diffusers stack"
  fi
  if ! PYTHONNOUSERSITE=1 /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${ENVS_ROOT}/cococo" \
    python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)" >/dev/null 2>&1; then
    echo "  install cococo torch stack: ${COCOCO_TORCH_PACKAGES}"
    # shellcheck disable=SC2086
    PYTHONNOUSERSITE=1 /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${ENVS_ROOT}/cococo" \
      python -m pip install --index-url "${COCOCO_TORCH_INDEX_URL}" ${COCOCO_TORCH_PACKAGES}
  else
    echo "  cococo torch ok"
  fi
fi

if [[ -x "/home/nvme01/miniconda3/bin/conda" && -d "${ENVS_ROOT}/minimax" ]]; then
  echo "[env] verify MiniMax runtime extras"
  if ! PYTHONNOUSERSITE=1 /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${ENVS_ROOT}/minimax" \
    python -c "import typing_extensions" >/dev/null 2>&1; then
    echo "  install minimax extra: typing_extensions"
    PYTHONNOUSERSITE=1 /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${ENVS_ROOT}/minimax" \
      python -m pip install "typing_extensions>=4.10"
  else
    echo "  minimax extras ok: typing_extensions"
  fi
fi

echo "[weights] create weight directories and copy any local known weights"
mkdir -p "${WEIGHTS_ROOT}/propainter" "${WEIGHTS_ROOT}/cococo" "${WEIGHTS_ROOT}/minimax" "${WEIGHTS_ROOT}/diffueraser" "${WEIGHTS_ROOT}/vbench"
if [[ -d "${PROJECT_ROOT}/weights/propainter" ]]; then
  rsync -a "${PROJECT_ROOT}/weights/propainter/" "${WEIGHTS_ROOT}/propainter/" || true
fi
if [[ -d "${PROJECT_ROOT}/weights/diffuEraser" ]]; then
  rsync -a "${PROJECT_ROOT}/weights/diffuEraser/" "${WEIGHTS_ROOT}/diffueraser/" || true
fi
if [[ -d "${PROJECT_ROOT}/weights/metrics" ]]; then
  rsync -a "${PROJECT_ROOT}/weights/metrics/" "${WEIGHTS_ROOT}/metrics/" || true
fi
if [[ -x "/home/nvme01/miniconda3/bin/conda" ]]; then
  echo "[weights] try MiniMax-Remover Hugging Face download"
  /home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${DIFFUERASER_ENV}" python -c \
    "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zibojia/minimax-remover', allow_patterns=['vae/**','transformer/**','scheduler/**'], local_dir='${WEIGHTS_ROOT}/minimax')" || true
fi

cat > "${THIRD_PARTY_ROOT}/WEIGHTS_TODO.md" <<'EOF'
# Third-party weight checklist

The code is cloned under `third_party_video_inpainting/repos`.
Weight downloads are intentionally kept as official-README commands because these projects
move checkpoints between Hugging Face, ModelScope, OneDrive, and Google Drive over time.

Recommended order:

0. Unified script
   - Run:
     bash DPO_finetune/scripts/download_multimodel_weights_h20.sh
   - This downloads the COCOCO Hugging Face bundle, Stable Diffusion inpainting
     fallback weights, MiniMax weights, and links local ProPainter weights into
     `third_party_video_inpainting/weights`.

1. ProPainter
   - Open `repos/ProPainter/README.md`.
   - Put RAFT / recurrent flow completion / ProPainter checkpoints under `weights/propainter`.
   - The current DiffuEraser repo may already contain ProPainter weights under `weights/propainter`.
   - ProPainter can also auto-download `ProPainter.pth`, `recurrent_flow_completion.pth`,
     and `raft-things.pth` during its first inference.

2. COCOCO
   - Open `repos/COCOCO/README.md` if the unified script fails.
   - Download its text-guided video inpainting checkpoints under
     `weights/COCOCO_weight/cococo`.
   - COCOCO requires a prompt; the generation script supports `--caption_json`.
   - It needs both Stable Diffusion inpainting weights and CoCoCo weights.
   - The CoCoCo README currently says the CoCoCo folder should contain model_0.pth
     through model_3.pth.

3. MiniMax-Remover
   - Open `repos/MiniMax-Remover/README.md`.
   - Download its remover/checkpoints under `weights/minimax`.
   - Official command:
     huggingface-cli download zibojia/minimax-remover --include vae transformer scheduler --local-dir .

4. DiffuEraser
   - The new DPO data generation should not use the DiffuEraser SFT inference result as a
     negative by default. The repo and weights are still archived here for reference and ablation.

5. VBench
   - `repos/VBench` is cloned. If VBench asks for extra weights, follow its official README.
EOF

echo "[data] search for full-resolution DAVIS and YouTube-VOS on H20 disks"
for root in /home/nvme01 /home/nvme03 /home/hj; do
  [[ -d "${root}" ]] || continue
  find "${root}" -maxdepth 8 -type d \( \
    -path "*ytbv_2019_full_resolution*JPEGImages*" -o \
    -path "*youtube*vos*JPEGImages*" -o \
    -path "*davis_2017_full_resolution*JPEGImages*" -o \
    -path "*DAVIS*JPEGImages*Full-Resolution*" \
  \) 2>/dev/null | head -40
done | sort -u | tee "${THIRD_PARTY_ROOT}/manifests/dataset_roots_found.txt"

cat <<EOF

[done] Third-party workspace:
  ${THIRD_PARTY_ROOT}

Next:
  1. Run:
     bash ${PROJECT_ROOT}/DPO_finetune/scripts/download_multimodel_weights_h20.sh
  2. Run:
     cp ${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.example.json \\
        ${PROJECT_ROOT}/DPO_finetune/configs/multimodel_adapters_h20.json
  3. Run:
     CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=0,1,2,3 \\
       bash ${PROJECT_ROOT}/DPO_finetune/scripts/smoke_multimodel_h20.sh
EOF
