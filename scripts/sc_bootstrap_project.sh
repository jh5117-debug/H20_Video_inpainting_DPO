#!/usr/bin/env bash
set -euo pipefail

PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
DEV_ROOT="${DEV_ROOT:-${PROJECT_HOME}/dev}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEV_ROOT}/Reg_DPO_Inpainting}"
REPO_URL="${REPO_URL:-git@github.com:jh5117-debug/H20_Video_inpainting_DPO.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"

ARCHIVES_DIR="${ARCHIVES_DIR:-${PROJECT_ROOT}/archives}"
DATA_EXTERNAL_DIR="${DATA_EXTERNAL_DIR:-${PROJECT_ROOT}/data/external}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"

DATASET_REPO_ID="${DATASET_REPO_ID:-JiaHuang01/New_DPO_data}"
DATASET_REPO_TYPE="${DATASET_REPO_TYPE:-dataset}"
DATASET_FILENAME="${DATASET_FILENAME:-DPO_Finetune_data.tar.gz}"

ASSETS_REPO_ID="${ASSETS_REPO_ID:-JiaHuang01/DPO_Finetune_Data}"
ASSETS_REPO_TYPE="${ASSETS_REPO_TYPE:-dataset}"
WEIGHTS_ARCHIVE="${WEIGHTS_ARCHIVE:-DiffuEraser_runtime_weights_20260418.tar.zst}"
BASE_DATA_ARCHIVE="${BASE_DATA_ARCHIVE:-DiffuEraser_DAVIS_YouTubeVOS_datasets_20260418.tar.zst}"

mkdir -p "${DEV_ROOT}"

if [[ -d "${PROJECT_ROOT}/.git" ]]; then
  echo "[sc-bootstrap] updating existing checkout: ${PROJECT_ROOT}"
  git -C "${PROJECT_ROOT}" fetch origin "${REPO_BRANCH}"
  git -C "${PROJECT_ROOT}" checkout "${REPO_BRANCH}"
  git -C "${PROJECT_ROOT}" pull --ff-only origin "${REPO_BRANCH}"
else
  echo "[sc-bootstrap] cloning repo to ${PROJECT_ROOT}"
  git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${PROJECT_ROOT}"
fi

mkdir -p "${ARCHIVES_DIR}" "${DATA_EXTERNAL_DIR}" "${WEIGHTS_DIR}" "${EXPERIMENTS_DIR}"

python - <<'PY' "${ARCHIVES_DIR}" "${DATASET_REPO_ID}" "${DATASET_REPO_TYPE}" "${DATASET_FILENAME}" "${ASSETS_REPO_ID}" "${ASSETS_REPO_TYPE}" "${WEIGHTS_ARCHIVE}" "${BASE_DATA_ARCHIVE}"
import subprocess
import sys
from pathlib import Path

archives_dir = Path(sys.argv[1])
dataset_repo = sys.argv[2]
dataset_repo_type = sys.argv[3]
dataset_filename = sys.argv[4]
assets_repo = sys.argv[5]
assets_repo_type = sys.argv[6]
weights_archive = sys.argv[7]
base_data_archive = sys.argv[8]

try:
    from huggingface_hub import hf_hub_download
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

downloads = [
    (dataset_repo, dataset_repo_type, dataset_filename),
    (assets_repo, assets_repo_type, weights_archive),
    (assets_repo, assets_repo_type, base_data_archive),
]
for repo_id, repo_type, filename in downloads:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=str(archives_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[sc-bootstrap] downloaded {repo_id}/{filename} -> {path}")
PY

cd "${PROJECT_ROOT}"
echo "[sc-bootstrap] extracting dataset archive"
tar -xzf "${ARCHIVES_DIR}/${DATASET_FILENAME}" -C "${DATA_EXTERNAL_DIR}"

echo "[sc-bootstrap] extracting base data archive"
tar --zstd -xf "${ARCHIVES_DIR}/${BASE_DATA_ARCHIVE}" -C "${PROJECT_ROOT}"

echo "[sc-bootstrap] extracting weights archive"
tar --zstd -xf "${ARCHIVES_DIR}/${WEIGHTS_ARCHIVE}" -C "${PROJECT_ROOT}"

ENV_FILE="${PROJECT_ROOT}/env.sc.sh"
cat > "${ENV_FILE}" <<EOF
#!/usr/bin/env bash
export PROJECT_HOME="${PROJECT_ROOT}"
export PROJECT_ROOT="\${PROJECT_HOME}"
export WEIGHTS_DIR="\${PROJECT_HOME}/weights"
export DPO_DATA_ROOT="\${PROJECT_HOME}/data/external/DPO_Finetune_data"
export VAL_DATA_DIR="\${PROJECT_HOME}/data/external/davis_432_240"
export EXPERIMENTS_DIR="\${PROJECT_HOME}/experiments"
export WANDB_PROJECT="\${WANDB_PROJECT:-DPO_Diffueraser}"
export WANDB_ENTITY="\${WANDB_ENTITY:-}"
export HF_HOME="\${PROJECT_HOME}/.hf_cache"
export TRANSFORMERS_CACHE="\${PROJECT_HOME}/.hf_cache"
export WANDB_DIR="\${PROJECT_HOME}/.wandb_cache"
export WANDB_CACHE_DIR="\${PROJECT_HOME}/.wandb_cache"
export WANDB_DATA_DIR="\${PROJECT_HOME}/.wandb_cache"
export WANDB_CONFIG_DIR="\${PROJECT_HOME}/.wandb_cache/config"
EOF
chmod +x "${ENV_FILE}"

echo "[sc-bootstrap] wrote ${ENV_FILE}"
echo "[sc-bootstrap] project_root=${PROJECT_ROOT}"
echo "[sc-bootstrap] dataset_root=${DATA_EXTERNAL_DIR}/DPO_Finetune_data"
echo "[sc-bootstrap] weights_dir=${WEIGHTS_DIR}"
