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
inference/           evaluation and inference scripts
tools/               utility scripts
```

Compatibility wrappers are kept in the historical locations so existing commands still work.

## Inputs

```text
data/                training inputs, ignored by Git
data_val/            validation inputs, ignored by Git
weights/             pretrained/external weights, ignored by Git
```

These directories should be treated as read-only inputs during training.

## Outputs

```text
experiments/sft/stage1/<version>_<run_name>/
experiments/sft/stage2/<version>_<run_name>/
experiments/dpo/stage1/<version>_<run_name>/
experiments/dpo/stage2/<version>_<run_name>/
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

