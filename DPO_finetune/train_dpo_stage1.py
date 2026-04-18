#!/usr/bin/env python
# coding=utf-8
"""Compatibility wrapper for DPO Stage 1 trainer."""

import runpy


if __name__ == "__main__":
    runpy.run_module("training.dpo.train_stage1", run_name="__main__")
