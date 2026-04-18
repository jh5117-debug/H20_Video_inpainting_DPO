# H20 Video Inpainting DPO

DiffuEraser video inpainting finetuning project with two isolated training tracks:

1. SFT finetunes DiffuEraser in two stages.
2. DPO adapts VideoDPO-style preference learning to DiffuEraser.

## Directory Layout

```text
H20_Video_inpainting_DPO/
├── training/
│   ├── sft/                    # SFT Stage 1/2 training implementation
│   ├── dpo/                    # DPO Stage 1/2 training + DPO dataset
│   └── common/                 # shared paths, manifests, validation helpers
├── scripts/                    # compatibility launchers for SFT
├── DPO_finetune/               # compatibility launchers + DPO docs
├── dataset/                    # SFT dataset code only
├── diffueraser/                # DiffuEraser pipelines and inference wrappers
├── libs/                       # model definitions
├── inference/                  # evaluation and inference tools
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

