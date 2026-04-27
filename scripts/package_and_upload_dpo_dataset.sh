#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_HOME:-${DEFAULT_PROJECT_ROOT}}}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/DPO_Finetune_Data_Multimodel_v1}"
EXPORT_ROOT_NAME="${EXPORT_ROOT_NAME:-DPO_Finetune_data}"
ARCHIVES_DIR="${ARCHIVES_DIR:-${PROJECT_ROOT}/archives}"
ARCHIVE_NAME="${ARCHIVE_NAME:-DPO_Finetune_data.tar.gz}"
ARCHIVE_PATH="${ARCHIVES_DIR}/${ARCHIVE_NAME}"
HF_REPO_ID="${HF_REPO_ID:-JiaHuang01/New_DPO_data}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-${ARCHIVE_NAME}}"
UPLOAD="${UPLOAD:-1}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "dataset root not found: ${DATASET_ROOT}" >&2
  exit 1
fi

mkdir -p "${ARCHIVES_DIR}"

echo "[dataset] project_root=${PROJECT_ROOT}"
echo "[dataset] dataset_root=${DATASET_ROOT}"
echo "[dataset] export_root_name=${EXPORT_ROOT_NAME}"
echo "[dataset] archives_dir=${ARCHIVES_DIR}"
echo "[dataset] archive_path=${ARCHIVE_PATH}"
echo "[dataset] hf_repo=${HF_REPO_ID} (${HF_REPO_TYPE}) -> ${HF_PATH_IN_REPO}"
echo
echo "[dataset] uncompressed size"
du -sh "${DATASET_ROOT}"

compress_cmd=(gzip -1)
archive_ext=".gz"
if command -v pigz >/dev/null 2>&1; then
  compress_cmd=(pigz -1)
fi

tmp_archive="${ARCHIVE_PATH}.tmp"
rm -f "${tmp_archive}"

base_name="$(basename "${DATASET_ROOT}")"
parent_dir="$(dirname "${DATASET_ROOT}")"

echo "[dataset] creating archive..."
tar \
  --warning=no-file-changed \
  -C "${parent_dir}" \
  --transform "s,^${base_name},${EXPORT_ROOT_NAME}," \
  -cf - "${base_name}" \
  | "${compress_cmd[@]}" > "${tmp_archive}"

mv "${tmp_archive}" "${ARCHIVE_PATH}"

echo "[dataset] archive size"
du -sh "${ARCHIVE_PATH}"
echo "[dataset] sha256"
sha256sum "${ARCHIVE_PATH}"

if [[ "${UPLOAD}" == "0" ]]; then
  echo "[dataset] UPLOAD=0, skipping Hugging Face upload."
  exit 0
fi

python - <<'PY' "${ARCHIVE_PATH}" "${HF_REPO_ID}" "${HF_REPO_TYPE}" "${HF_PATH_IN_REPO}"
import os
import subprocess
import sys
from pathlib import Path

archive_path = Path(sys.argv[1])
repo_id = sys.argv[2]
repo_type = sys.argv[3]
path_in_repo = sys.argv[4]

try:
    from huggingface_hub import HfApi
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
    from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=str(archive_path),
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"[dataset] uploaded {archive_path} -> {repo_type}:{repo_id}/{path_in_repo}")
PY
