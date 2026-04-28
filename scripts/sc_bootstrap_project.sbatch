#!/bin/bash
#SBATCH --job-name=DPO_BOOT
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sc-bootstrap-%j.out
#SBATCH --open-mode=append

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
DATA_NAME="${DATA_NAME:-Video_inpainting_DPO}"
export PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_DEV}/data}"
export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DEV}/${PROJECT_NAME}}"
export DATA="${DATA:-${PROJECT_DATA}/${DATA_NAME}}"
export DATA_NAME

mkdir -p "${PROJECT_ROOT}/logs"
exec bash "${SCRIPT_DIR}/sc_bootstrap_project.sh"
