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
export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"
exec bash "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" "$@"
