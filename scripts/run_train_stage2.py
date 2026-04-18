#!/usr/bin/env python
# coding=utf-8
"""Compatibility wrapper for SFT Stage 2 launcher."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.sft.scripts.run_stage2 import run_stage2


if __name__ == "__main__":
    sys.exit(run_stage2())
