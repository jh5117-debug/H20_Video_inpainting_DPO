#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run official COCOCO inference from frame directories and save one sample.

COCOCO's release script expects:
  video_path/
    images.npy  # F,H,W,3 uint8 RGB
    masks.npy   # F,H,W,1 uint8, 255 means inpaint

This wrapper only adapts our DPO frame-dir layout to that official interface.
It writes a temporary patched copy of the official script so smoke tests do not
spend time generating all 10 samples from the release default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

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


def prepare_npys(
    video_dir: Path,
    mask_dir: Path,
    work_dir: Path,
    width: int,
    height: int,
    num_frames: int,
) -> Tuple[Path, int]:
    frame_files = image_files(video_dir)
    mask_files = image_files(mask_dir)
    n = min(len(frame_files), len(mask_files))
    if num_frames > 0:
        n = min(n, num_frames)
    if n <= 0:
        raise RuntimeError("no input frames or masks found for COCOCO")

    input_dir = work_dir / "cococo_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for frame_path, mask_path in zip(frame_files[:n], mask_files[:n]):
        image = center_crop_resize(read_rgb(frame_path), width, height, cv2.INTER_LINEAR)
        mask = center_crop_resize(read_gray(mask_path), width, height, cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        images.append(image.astype(np.uint8))
        masks.append(mask[:, :, None])

    np.save(input_dir / "images.npy", np.stack(images, axis=0))
    np.save(input_dir / "masks.npy", np.stack(masks, axis=0))
    return input_dir, n


def write_config(repo_dir: Path, work_dir: Path, output_base: Path) -> Path:
    template = repo_dir / "configs" / "code_release.yaml"
    if not template.exists():
        raise FileNotFoundError(f"COCOCO config template not found: {template}")
    config = work_dir / "code_release_dpo.yaml"
    text = template.read_text(encoding="utf-8")
    if re.search(r"(?m)^output_dir\s*:", text):
        text = re.sub(r"(?m)^output_dir\s*:.*$", f'output_dir: "{output_base}"', text)
    else:
        text = f'output_dir: "{output_base}"\n' + text
    config.write_text(text, encoding="utf-8")
    return config


def write_patched_runner(repo_dir: Path, work_dir: Path, num_samples: int) -> Path:
    src = repo_dir / "valid_code_release.py"
    if not src.exists():
        raise FileNotFoundError(f"COCOCO release script not found: {src}")
    text = src.read_text(encoding="utf-8")
    if not re.search(r"(?m)^import os\b", text):
        text = "import os\n" + text
    old = "for step in range(10):"
    new = f"for step in range({max(1, num_samples)}):"
    if old not in text:
        raise RuntimeError("could not patch COCOCO sample loop; release script changed")
    text = text.replace(old, new)

    # Some COCOCO releases build `pretrained_model_path` as
    # `<sd_root>/<sub_folder>` and then load VAE/tokenizer/scheduler from that
    # path. That makes AutoencoderKL read the UNet config as VAE config.
    root_loaders = [
        'AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")',
        "AutoencoderKL.from_pretrained(pretrained_model_path, subfolder='vae')",
        'DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")',
        "DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')",
        'DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")',
        "DDPMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')",
        'CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")',
        "CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder='tokenizer')",
        'CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")',
        "CLIPTextModel.from_pretrained(pretrained_model_path, subfolder='text_encoder')",
    ]
    for expr in root_loaders:
        text = text.replace(expr, expr.replace("pretrained_model_path", "args.pretrain_model_path"))
    for cls_name, subfolder in [
        ("AutoencoderKL", "vae"),
        ("DDIMScheduler", "scheduler"),
        ("DDPMScheduler", "scheduler"),
        ("CLIPTokenizer", "tokenizer"),
        ("CLIPTextModel", "text_encoder"),
    ]:
        text = re.sub(
            rf"{cls_name}\.from_pretrained\(\s*pretrained_model_path\s*,\s*subfolder=([\"']){subfolder}\1",
            rf"{cls_name}.from_pretrained(args.pretrain_model_path, subfolder=\1{subfolder}\1",
            text,
        )
        text = re.sub(
            rf"{cls_name}\.from_pretrained\(\s*args\.pretrain_model_path\s*,\s*subfolder=([\"']){subfolder}\1\s*\)",
            rf"{cls_name}.from_pretrained(os.path.join(args.pretrain_model_path, \1{subfolder}\1))",
            text,
        )
    patched = work_dir / "valid_code_release_dpo_patched.py"
    patched.write_text(text, encoding="utf-8")
    return patched


def collect_outputs(output_base: Path, output_dir: Path, num_frames: int, width: int, height: int) -> None:
    files = sorted(output_base.glob("**/*.png"))
    by_step: Dict[int, List[Tuple[int, Path]]] = {}
    pattern = re.compile(r"_guidance_scale_(\d+)_image_(\d+)\.png$")
    for path in files:
        match = pattern.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        frame = int(match.group(2))
        by_step.setdefault(step, []).append((frame, path))
    if not by_step:
        raise RuntimeError(f"COCOCO produced no frame pngs under {output_base}")

    best_step = max(by_step, key=lambda step: (len(by_step[step]), step))
    chosen = sorted(by_step[best_step], key=lambda x: x[0])
    if len(chosen) < num_frames:
        chosen = chosen + [chosen[-1]] * (num_frames - len(chosen))

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (_, src) in enumerate(chosen[:num_frames]):
        arr = center_crop_resize(read_rgb(src), width, height, cv2.INTER_LINEAR)
        save_rgb(output_dir / f"{idx:05d}.png", arr)


def validate_sd_inpainting_root(path: Path) -> None:
    required = [
        path / "model_index.json",
        path / "vae" / "config.json",
        path / "unet" / "config.json",
        path / "tokenizer",
        path / "text_encoder" / "config.json",
        path / "scheduler" / "scheduler_config.json",
    ]
    missing = [str(p.relative_to(path)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"SD inpainting folder is incomplete: {path}; missing={missing}")

    with (path / "vae" / "config.json").open("r", encoding="utf-8") as f:
        vae_config = json.load(f)
    vae_down_blocks = vae_config.get("down_block_types", [])
    vae_up_blocks = vae_config.get("up_block_types", [])
    print(
        f"[cococo] sd_root={path}\n"
        f"[cococo] vae down_block_types={vae_down_blocks} up_block_types={vae_up_blocks}",
        flush=True,
    )
    if any("CrossAttn" in str(x) for x in vae_down_blocks + vae_up_blocks):
        raise RuntimeError(
            "SD inpainting vae/config.json looks like a conditional UNet config, not AutoencoderKL. "
            "Rerun DPO_finetune/scripts/download_multimodel_weights_h20.sh with "
            "SD_INPAINT_HF_REPO=JiaHuang01/COCOCO_SD_INPAINT and SD_INPAINT_HF_FILENAME=stable-diffusion-inpainting.zip."
        )

    with (path / "unet" / "config.json").open("r", encoding="utf-8") as f:
        unet_config = json.load(f)
    print(
        f"[cococo] unet in_channels={unet_config.get('in_channels')} "
        f"cross_attention_dim={unet_config.get('cross_attention_dim')} "
        f"down_block_types={unet_config.get('down_block_types')}",
        flush=True,
    )
    if int(unet_config.get("in_channels", -1)) != 9:
        raise RuntimeError(f"SD inpainting UNet should have in_channels=9, got {unet_config.get('in_channels')}")
    if "cross_attention_dim" not in unet_config:
        raise RuntimeError("SD inpainting unet/config.json missing cross_attention_dim")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COCOCO on one DPO video candidate.")
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--model_path", required=True, help="Folder containing model_0.pth ... model_3.pth")
    parser.add_argument("--pretrain_model_path", required=True, help="Stable Diffusion inpainting diffusers folder")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="worst quality. bad quality.")
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model_path).resolve()
    pretrain_model_path = Path(args.pretrain_model_path).resolve()

    for idx in range(4):
        ckpt = model_path / f"model_{idx}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing COCOCO checkpoint: {ckpt}")
    validate_sd_inpainting_root(pretrain_model_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_dir, num_frames = prepare_npys(
        Path(args.video_dir),
        Path(args.mask_dir),
        work_dir,
        args.width,
        args.height,
        args.num_frames,
    )
    output_base = work_dir / "cococo_outputs"
    config_path = write_config(repo_dir, work_dir, output_base)
    runner = write_patched_runner(repo_dir, work_dir, args.num_samples)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd: Sequence[str] = [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
        "--prompt",
        args.prompt,
        "--negative_prompt",
        args.negative_prompt,
        "--guidance_scale",
        str(args.guidance_scale),
        "--video_path",
        str(input_dir),
        "--model_path",
        str(model_path),
        "--pretrain_model_path",
        str(pretrain_model_path),
    ]
    subprocess.run(cmd, cwd=str(repo_dir), env=env, check=True)
    collect_outputs(output_base, output_dir, num_frames, args.width, args.height)
    print(f"[cococo] saved {num_frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
