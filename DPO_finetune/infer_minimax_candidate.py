#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run MiniMax-Remover from DPO frame directories and save output frames."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def image_files(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return arr


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))


def center_crop_resize(img: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    h, w = img.shape[:2]
    target_ratio = width / height
    ratio = w / h
    if ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        x0 = max(0, (w - new_w) // 2)
        img = img[:, x0:x0 + new_w]
    elif ratio < target_ratio:
        new_h = int(round(w / target_ratio))
        y0 = max(0, (h - new_h) // 2)
        img = img[y0:y0 + new_h, :]
    return cv2.resize(img, (width, height), interpolation=interpolation)


def compatible_temporal_length(n: int) -> int:
    """Wan-style VAEs are happiest at 4k+1 frames; pad and crop back."""
    if n <= 1:
        return n
    remainder = (n - 1) % 4
    return n if remainder == 0 else n + (4 - remainder)


def prepare_inputs(
    video_dir: Path,
    mask_dir: Path,
    width: int,
    height: int,
    num_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    frame_files = image_files(video_dir)
    mask_files = image_files(mask_dir)
    n = min(len(frame_files), len(mask_files))
    if num_frames > 0:
        n = min(n, num_frames)
    if n <= 0:
        raise RuntimeError("no input frames or masks found for MiniMax")

    frames: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for frame_path, mask_path in zip(frame_files[:n], mask_files[:n]):
        frame = center_crop_resize(read_rgb(frame_path), width, height, cv2.INTER_LINEAR)
        mask = center_crop_resize(read_gray(mask_path), width, height, cv2.INTER_NEAREST)
        frames.append(frame.astype(np.float32) / 127.5 - 1.0)
        masks.append((mask > 20).astype(np.float32)[:, :, None])

    model_n = compatible_temporal_length(n)
    while len(frames) < model_n:
        frames.append(frames[-1].copy())
        masks.append(masks[-1].copy())

    images = torch.from_numpy(np.stack(frames, axis=0))
    mask_tensor = torch.from_numpy(np.stack(masks, axis=0))
    return images, mask_tensor, n, model_n


def frame_to_uint8(frame: object) -> np.ndarray:
    if hasattr(frame, "__array__"):
        arr = np.array(frame)
    else:
        raise TypeError(f"unsupported frame type: {type(frame).__name__}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected output frame shape: {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMax-Remover on one DPO video candidate.")
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", required=True, help="Folder containing vae/, transformer/, scheduler/")
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    for child in ("vae", "transformer", "scheduler"):
        if not (model_dir / child).exists():
            raise FileNotFoundError(f"missing MiniMax weight folder: {model_dir / child}")

    sys.path.insert(0, str(repo_dir))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    from diffusers.models import AutoencoderKLWan  # noqa: WPS433,E402
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433,E402
    from pipeline_minimax_remover import Minimax_Remover_Pipeline  # noqa: WPS433,E402
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433,E402

    images, masks, original_n, model_n = prepare_inputs(
        Path(args.video_dir),
        Path(args.mask_dir),
        args.width,
        args.height,
        args.num_frames,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")

    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed) if device.type == "cuda" else None
    with torch.inference_mode():
        result = pipe(
            images=images,
            masks=masks,
            num_frames=model_n,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            iterations=args.iterations,
        ).frames[0]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(result[:original_n]):
        save_rgb(output_dir / f"{idx:05d}.png", frame_to_uint8(frame))
    print(f"[minimax] saved {original_n} frames to {output_dir}")


if __name__ == "__main__":
    main()
