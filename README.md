# Video Inpainting DPO

DiffuEraser video inpainting finetuning project with two isolated training tracks:

1. SFT finetunes DiffuEraser in two stages.
2. DPO adapts VideoDPO-style preference learning to DiffuEraser.

## Directory Layout

```text
Video_inpainting_DPO/
├── training/
│   ├── sft/                    # SFT Stage 1/2 training implementation
│   ├── dpo/                    # DPO Stage 1/2 training + DPO dataset
│   └── common/                 # shared paths, manifests, validation helpers
├── scripts/                    # compatibility launchers for SFT
├── DPO_finetune/               # compatibility launchers + DPO docs
├── dataset/                    # SFT dataset code only
├── diffueraser/                # DiffuEraser pipelines and inference wrappers
├── libs/                       # model definitions
├── propainter/                 # local ProPainter support code for evaluation
├── inference/                  # visualization, metrics, report, and sweep tools
├── tools/                      # repository tools
├── docs/                       # engineering notes
├── PRD/                        # reports and project notes
├── data/                       # external input datasets, ignored
├── data_val/                   # external validation inputs, ignored
├── weights/                    # external model weights, ignored
└── experiments/                # generated run outputs, ignored
```

The old entrypoints remain as thin wrappers:

```text
train_DiffuEraser_stage1.py      -> training/sft/train_stage1.py
train_DiffuEraser_stage2.py      -> training/sft/train_stage2.py
DPO_finetune/train_dpo_stage1.py -> training/dpo/train_stage1.py
DPO_finetune/train_dpo_stage2.py -> training/dpo/train_stage2.py
```

## Output Isolation

New launcher defaults write generated outputs to:

```text
experiments/
├── sft/
│   ├── stage1/<version>_<run_name>/
│   └── stage2/<version>_<run_name>/
└── dpo/
    ├── stage1/<version>_<run_name>/
    └── stage2/<version>_<run_name>/
```

Each run directory gets a `run_manifest.json` with command, inputs, key params, git branch/commit/status, and output path. Each stage also maintains a best-effort `latest` symlink plus a `LATEST` text file.

## Quick Start

SFT:

```bash
python scripts/run_train_stage1.py
python scripts/run_train_stage2.py
```

DPO:

```bash
python DPO_finetune/scripts/run_dpo_stage1.py --chunk_aligned
python DPO_finetune/scripts/run_dpo_stage2.py --chunk_aligned
```

Single-node server, no SLURM:

```bash
export PROJECT_HOME=/home/nvme01/Video_inpainting_DPO
cd "${PROJECT_HOME}"
CUDA_VISIBLE_DEVICES=4,5,6,7 python DPO_finetune/scripts/run_dpo_stage1.py --chunk_aligned
CUDA_VISIBLE_DEVICES=4,5,6,7 python DPO_finetune/scripts/run_dpo_stage2.py --chunk_aligned
```

For single-node stability, use `MIXED_PRECISION=bf16` and disable xFormers unless
the runtime stack has already been verified. You can override settings per run:

```bash
MIXED_PRECISION=fp16 XFORMERS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  python DPO_finetune/scripts/run_dpo_stage1.py --chunk_aligned
```

The launchers infer `PROJECT_ROOT` from their location by default. For another
server, set `PROJECT_HOME` to that checkout and optionally override asset/env
locations:

```bash
export PROJECT_HOME=/path/to/Video_inpainting_DPO
export CONDA_BASE=/path/to/miniconda3
export CONDA_ENV_PREFIX=/path/to/conda_envs/diffueraser
export WEIGHTS_DIR="${PROJECT_HOME}/weights"
export DPO_DATA_ROOT="${PROJECT_HOME}/data/external/DPO_Finetune_data"
export VAL_DATA_DIR="${PROJECT_HOME}/data/external/davis_432_240"
export EXPERIMENTS_DIR="${PROJECT_HOME}/experiments"
```

The launchers keep caches and generated files inside the project by default.
Runtime paths resolve to:

```text
project root:   ${PROJECT_HOME}
DPO data:       data/external/DPO_Finetune_data
validation:     data/external/davis_432_240
weights:        weights/
outputs:        experiments/dpo/
```

Override `PROJECT_HOME`, `CONDA_BASE`, `CONDA_ENV_PREFIX`, `CUDA_VISIBLE_DEVICES`,
`NUM_GPUS`, `DPO_DATA_ROOT`, `WEIGHTS_DIR`, `VAL_DATA_DIR`, `EXPERIMENTS_DIR`,
`RUN_NAME`, or `RUN_VERSION` from the shell when needed.

Evaluation / visualization after a new weight is exported:

```bash
bash inference/run_weight_sweep.sh
bash inference/start_weight_sweep.sh
```

The historical entrypoints are kept as wrappers:

```bash
bash run_24exp.sh
bash start_24exp.sh
```

The sweep runner reads DAVIS inputs from `data/external/`, model assets from `weights/`,
and writes generated videos, metrics, logs, and reports to
`experiments/evaluation/weight_sweep/`. BR runs compute pixel metrics by default;
OR runs generate videos by default and can enable VBench with `OR_ENABLE_VBENCH=1`.

SLURM compatibility entrypoints are still available:

```bash
sbatch scripts/02_train_all.sbatch
sbatch DPO_finetune/scripts/03_dpo_stage1.sbatch
sbatch DPO_finetune/scripts/03_dpo_stage2.sbatch
```

Useful environment overrides:

```bash
EXPERIMENTS_DIR=/path/to/experiments
RUN_NAME=my_ablation
RUN_VERSION=20260418_a
DATA_DIR=/path/to/data
WEIGHTS_DIR=/path/to/weights
```

Weights and datasets are intentionally not managed by this repo.

## External Assets

Large datasets, pretrained weights, and compressed backups are kept out of Git.
Use the project-local ignored directories for runtime:

```text
data/external/     unpacked datasets
weights/           model weights
archives/          temporary compressed archives before/after HF transfer
```

The canonical compressed archives live on Hugging Face:

```text
https://huggingface.co/datasets/JiaHuang01/DPO_Finetune_Data/blob/main/DPO_Finetune_data.tar.gz
https://huggingface.co/datasets/JiaHuang01/DPO_Finetune_Data/blob/main/DiffuEraser_DAVIS_YouTubeVOS_datasets_20260418.tar.zst
https://huggingface.co/datasets/JiaHuang01/DPO_Finetune_Data/blob/main/DiffuEraser_runtime_weights_20260418.tar.zst
```

Do not commit unpacked data, weights, generated experiments, or local archives to
GitHub. The repository should contain only code, launchers, lightweight docs, and
small configuration files.
