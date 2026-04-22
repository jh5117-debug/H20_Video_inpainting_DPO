#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run local ProPainter and save lossless output frames for DPO candidate generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from propainter.inference import Propainter  # noqa: E402
from propainter.model.misc import get_device  # noqa: E402


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ProPainter and save output frames.")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", default=str(REPO_ROOT / "weights" / "propainter"))
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--width", type=int, default=-1)
    parser.add_argument("--height", type=int, default=-1)
    parser.add_argument("--ref_stride", type=int, default=10)
    parser.add_argument("--neighbor_length", type=int, default=20)
    parser.add_argument("--subvideo_length", type=int, default=80)
    parser.add_argument("--mask_dilation", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = Propainter(args.model_dir, device=device)
    frames = model.forward(
        video=args.video_dir,
        mask=args.mask_dir,
        output_path=None,
        resize_ratio=1.0,
        video_length=args.num_frames,
        height=args.height,
        width=args.width,
        mask_dilation=args.mask_dilation,
        ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length,
        raft_iter=20,
        save_fps=24,
        save_frames=False,
        fp16=True,
        return_frames=True,
    )

    for idx, frame in enumerate(frames):
        save_rgb(out_dir / f"{idx:05d}.png", frame)


if __name__ == "__main__":
    main()
