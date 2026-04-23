#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run local DiffuEraser OR inference and save output frames for DPO candidates."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def image_files(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def first_sequence_name(video_root: Path, mask_root: Path) -> str:
    for child in sorted(video_root.iterdir()):
        if child.is_dir() and (mask_root / child.name).is_dir() and image_files(child):
            return child.name
    raise RuntimeError(f"no matching frame/mask sequence found under {video_root} and {mask_root}")


def mp4_to_frames(mp4_path: Path, output_dir: Path, limit: int) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open DiffuEraser mp4: {mp4_path}")

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if limit > 0 and count >= limit:
            break
        cv2.imwrite(str(output_dir / f"{count:05d}.png"), frame)
        count += 1
    cap.release()
    if count == 0:
        raise RuntimeError(f"no frames decoded from DiffuEraser mp4: {mp4_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffuEraser and save DPO candidate frames.")
    parser.add_argument("--video_root", required=True, help="Batch video root containing one sequence dir.")
    parser.add_argument("--mask_root", required=True, help="Batch mask root containing one sequence dir.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--diffueraser_path", required=True)
    parser.add_argument("--propainter_model_dir", required=True)
    parser.add_argument("--pcm_weights_path", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="worst quality. bad quality.")
    parser.add_argument("--text_guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--mask_dilation_iter", type=int, default=8)
    parser.add_argument("--offload_models", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    video_root = Path(args.video_root).resolve()
    mask_root = Path(args.mask_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    sequence_name = first_sequence_name(video_root, mask_root)
    run_dir = work_dir / "run_or"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(project_root / "inference" / "run_OR.py"),
        "--dataset",
        "custom",
        "--video_root",
        str(video_root),
        "--mask_root",
        str(mask_root),
        "--save_path",
        str(run_dir),
        "--video_length",
        str(args.num_frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--mask_dilation_iter",
        str(args.mask_dilation_iter),
        "--base_model_path",
        str(Path(args.base_model_path).resolve()),
        "--vae_path",
        str(Path(args.vae_path).resolve()),
        "--diffueraser_path",
        str(Path(args.diffueraser_path).resolve()),
        "--propainter_model_dir",
        str(Path(args.propainter_model_dir).resolve()),
        "--pcm_weights_path",
        str(Path(args.pcm_weights_path).resolve()),
        "--summary_out",
        "summary.json",
    ]
    if args.prompt.strip():
        cmd.extend([
            "--use_text",
            "--prompt",
            args.prompt.strip(),
            "--n_prompt",
            args.negative_prompt,
            "--text_guidance_scale",
            str(args.text_guidance_scale),
        ])
    if args.offload_models:
        cmd.append("--offload_models")

    print("[diffueraser] run:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)

    pred_mp4 = run_dir / sequence_name / "diffueraser.mp4"
    saved = mp4_to_frames(pred_mp4, output_dir, args.num_frames)
    print(f"[diffueraser] saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    main()
