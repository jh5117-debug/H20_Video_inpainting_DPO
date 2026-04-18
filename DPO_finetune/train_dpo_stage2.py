#!/usr/bin/env python
# coding=utf-8
"""Compatibility wrapper for DPO Stage 2 trainer."""

import runpy


if __name__ == "__main__":
    runpy.run_module("training.dpo.train_stage2", run_name="__main__")
