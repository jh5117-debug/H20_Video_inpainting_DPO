#!/usr/bin/env python
# coding=utf-8
"""Deprecated compatibility import for the active DPO dataset.

The maintained DPO dataset lives at ``training.dpo.dataset.dpo_dataset``.
This module remains only so older scripts fail less surprisingly.
"""

import warnings

warnings.warn(
    "dataset.dpo_dataset is deprecated; import training.dpo.dataset.dpo_dataset instead.",
    DeprecationWarning,
    stacklevel=2,
)

from training.dpo.dataset.dpo_dataset import *  # noqa: F401,F403

