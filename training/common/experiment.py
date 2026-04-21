#!/usr/bin/env python
# coding=utf-8
"""Small experiment-output versioning helpers.

The project keeps code, inputs, weights, and generated experiment outputs in
separate top-level areas:

  code:        training/, dataset/, diffueraser/, libs/, inference/
  inputs:      data/, data_val/
  weights:     weights/
  outputs:     experiments/<family>/<stage>/<version>_<run_name>/

This is intentionally lightweight. It gives each run a stable directory,
writes a manifest, and updates a best-effort ``latest`` pointer.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_version() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def slugify(value: str | None, default: str) -> str:
    value = value or default
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = value.strip("-._")
    return value or default


def resolve_output_dir(
    root: str | Path,
    family: str,
    stage: str,
    explicit_output_dir: str | None = None,
    experiments_dir: str | None = None,
    run_name: str | None = None,
    run_version: str | None = None,
) -> Path:
    root = Path(root)
    if explicit_output_dir:
        return Path(explicit_output_dir).expanduser().resolve()

    base = Path(experiments_dir or os.environ.get("EXPERIMENTS_DIR") or root / "experiments")
    version = slugify(run_version or os.environ.get("RUN_VERSION") or utc_version(), "dev")
    name = slugify(run_name, f"{family}-{stage}")
    return (base / family / stage / f"{version}_{name}").resolve()


def latest_dir(root: str | Path, family: str, stage: str, experiments_dir: str | None = None) -> Path:
    root = Path(root)
    base = Path(experiments_dir or os.environ.get("EXPERIMENTS_DIR") or root / "experiments")
    return (base / family / stage / "latest").resolve()


def _git_value(root: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _update_latest(output_dir: Path) -> None:
    latest = output_dir.parent / "latest"
    latest_file = output_dir.parent / "LATEST"
    latest_file.write_text(str(output_dir) + "\n", encoding="utf-8")

    if latest.exists() and not latest.is_symlink():
        return
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(output_dir, target_is_directory=True)
    except OSError:
        pass


def prepare_experiment_dir(
    output_dir: str | Path,
    *,
    root: str | Path,
    family: str,
    stage: str,
    command: list[str] | None = None,
    inputs: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir).resolve()
    root = Path(root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _update_latest(output_dir)

    manifest = {
        "family": family,
        "stage": stage,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "project_root": str(root),
        "output_dir": str(output_dir),
        "inputs": inputs or {},
        "params": params or {},
        "command": command or [],
        "git": {
            "commit": _git_value(root, ["rev-parse", "HEAD"]),
            "branch": _git_value(root, ["branch", "--show-current"]),
            "status_short": _git_value(root, ["status", "--short"]),
        },
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (output_dir / "run_manifest.json").write_text(manifest_text, encoding="utf-8")

    external_log_dir = os.environ.get("DPO_EXTERNAL_LOG_DIR")
    if external_log_dir:
        external_dir = Path(external_log_dir).expanduser().resolve()
        external_dir.mkdir(parents=True, exist_ok=True)
        (external_dir / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
    return output_dir


def first_existing(*paths: str | Path | None) -> str | None:
    for path in paths:
        if path and Path(path).exists():
            return str(Path(path).resolve())
    return str(Path(paths[-1]).resolve()) if paths and paths[-1] else None
