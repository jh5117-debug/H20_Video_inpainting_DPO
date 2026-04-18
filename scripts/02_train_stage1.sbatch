#!/bin/bash
#SBATCH --job-name=DE_Stage1
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train-stage1-%j.out

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec bash "${PROJECT_ROOT}/training/sft/scripts/02_train_stage1.sbatch" "$@"

