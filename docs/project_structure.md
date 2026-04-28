# Project Structure

This repository separates code, external inputs, external weights, and generated outputs.

## Code

```text
training/sft/        SFT Stage 1/2 trainers and launchers
training/dpo/        DPO Stage 1/2 trainers, launchers, and DPO dataset
training/common/     shared experiment/output helpers
dataset/             SFT dataset code
diffueraser/         pipelines and inference wrappers
libs/                model definitions
propainter/          local ProPainter modules used by post-weight evaluation
inference/           visualization, metric, report, and sweep scripts
tools/               utility scripts
```

Compatibility wrappers are kept in the historical locations so existing commands still work.

## Inputs

```text
data/                training inputs, ignored by Git
data_val/            validation inputs, ignored by Git
weights/             pretrained/external weights, ignored by Git
archives/            compressed data archives, ignored by Git
```

These directories should be treated as read-only inputs during training.
Compressed dataset and weight archives are stored on Hugging Face instead of
GitHub:

```text
JiaHuang01/DPO_Finetune_Data/DPO_Finetune_data.tar.gz
JiaHuang01/DPO_Finetune_Data/DiffuEraser_DAVIS_YouTubeVOS_datasets_20260418.tar.zst
JiaHuang01/DPO_Finetune_Data/DiffuEraser_runtime_weights_20260418.tar.zst
```

Local `archives/` is only a temporary staging area for compression or restore.
It may be empty after a successful HF upload.

## Outputs

```text
experiments/sft/stage1/<version>_<run_name>/
experiments/sft/stage2/<version>_<run_name>/
experiments/dpo/stage1/<version>_<run_name>/
experiments/dpo/stage2/<version>_<run_name>/
experiments/evaluation/weight_sweep/
```

Every launcher-created run directory contains:

```text
run_manifest.json    command, inputs, params, git metadata
checkpoint-*         accelerator checkpoints
converted_weights/   exported SFT weights when produced
best_weights/        best DPO weights when produced
last_weights/        last DPO weights when produced
console_logs/        captured process logs when enabled
```

Each stage directory keeps:

```text
latest               symlink to latest run when the filesystem allows it
LATEST               plain-text fallback with latest run path
```

This is weak versioning: enough to keep runs separated and reproducible at the command/input level without introducing DVC or weight/data management.

## Evaluation

After exporting a new DiffuEraser checkpoint into `weights/`, run:

```bash
bash inference/run_weight_sweep.sh
```

The runner compares available weights named `Orign`, `FT_S2_8K`, `FT_S2_26K`,
`FT_S2_34K`, and `FT_S2_48K` across the OR/BR configurations from the previous
test workspace. Missing weight directories are skipped, outputs stay under
`experiments/evaluation/weight_sweep/`, and `inference/generate_report.py`
aggregates completed `summary.json` files into an experiment report.

## H20 Runtime

The H20 server uses the full project under:

```text
/home/nvme01/Video_inpainting_DPO
```

It does not require SLURM. DPO training should use the plain shell launchers:

```bash
cd /home/nvme01/Video_inpainting_DPO
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/h20_run_dpo_stage1.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/h20_run_dpo_stage2.sh
```

The launcher defaults are path-only adaptations:

```text
DPO data root:  data/external/DPO_Finetune_data
val data root:  data/external/davis_432_240
weights root:   weights/
outputs root:   experiments/dpo/
```

For H20 stability, the shell launchers default to `MIXED_PRECISION=bf16`,
`XFORMERS=0`, and `GRADIENT_CHECKPOINTING=1`. These are command-line/runtime
switches only; set the environment variables before a run to override them.

Training logic remains in `training/dpo/train_stage1.py` and
`training/dpo/train_stage2.py`; the H20 scripts only choose paths, GPUs, cache
locations, and command-line arguments.
