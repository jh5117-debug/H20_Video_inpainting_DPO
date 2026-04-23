#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate COCOCO prompt captions with Qwen2.5-VL for multimodel DPO data."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class VideoItem:
    source: str
    name: str
    frame_dir: Path
    frame_files: List[Path]


def image_files(path: Path) -> List[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def scan_videos(root: Path, source: str, min_frames: int) -> List[VideoItem]:
    if not root.exists():
        return []
    items: List[VideoItem] = []
    candidates: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if image_files(p):
            candidates.append(p)
        else:
            for q in sorted(p.iterdir()):
                if q.is_dir() and image_files(q):
                    candidates.append(q)

    seen = set()
    for frame_dir in candidates:
        files = image_files(frame_dir)
        if len(files) < min_frames:
            continue
        key = str(frame_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        clean_name = frame_dir.name.replace(" ", "_")
        prefix = source if not clean_name.startswith(source) else ""
        name = f"{prefix}_{clean_name}" if prefix else clean_name
        items.append(VideoItem(source=source, name=name, frame_dir=frame_dir, frame_files=files))
    return items


def pick_first_dir(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.is_dir():
            return path
    return paths[0]


def sample_frame_paths(files: Sequence[Path], max_frames: int, caption_frames: int) -> List[Path]:
    if max_frames > 0:
        files = list(files[:max_frames])
    if len(files) <= caption_frames:
        return list(files)
    if caption_frames <= 1:
        return [files[len(files) // 2]]
    idxs = [round(i * (len(files) - 1) / (caption_frames - 1)) for i in range(caption_frames)]
    return [files[i] for i in idxs]


def make_frame_grid(paths: Sequence[Path], tile_size: int = 336) -> Image.Image:
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
        x = (tile_size - img.width) // 2
        y = (tile_size - img.height) // 2
        canvas.paste(img, (x, y))
        images.append(canvas)
    if not images:
        raise RuntimeError("no frames available for caption grid")

    cols = 2 if len(images) > 1 else 1
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile_size, rows * tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for idx, img in enumerate(images):
        x = (idx % cols) * tile_size
        y = (idx // cols) * tile_size
        grid.paste(img, (x, y))
        draw.rectangle([x, y, x + tile_size - 1, y + tile_size - 1], outline=(180, 180, 180), width=2)
    return grid


def clean_caption(text: str, fallback: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.strip(" '\"")
    prefixes = ["caption:", "scene:", "description:"]
    lower = text.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if not text or len(text) < 8:
        return fallback
    text = text.split("\n", 1)[0].strip()
    if not text.endswith("."):
        text += "."
    return text


def fallback_caption(video: VideoItem) -> str:
    words = video.frame_dir.name.replace("_", " ").replace("-", " ")
    return f"A realistic video scene of {words} with a clean, coherent background."


def load_qwen(model_path: Path, device_map: str, dtype_name: str, attn_implementation: str):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
    except Exception:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    kwargs = {"torch_dtype": dtype, "device_map": device_map}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = model_cls.from_pretrained(model_path, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = model_cls.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def tensor_device(model):
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def caption_with_qwen(model, processor, image: Image.Image, max_new_tokens: int) -> str:
    prompt = (
        "Describe the scene across these video frames in one concise English sentence for a video "
        "inpainting prompt. Focus on background, setting, camera view, visible objects, and visual "
        "style. Do not mention frame numbers, masks, missing areas, or that this is a collage."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    device = tensor_device(model)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate caption JSON for COCOCO prompts with Qwen2.5-VL.")
    parser.add_argument("--project_root", default=str(REPO_ROOT))
    parser.add_argument("--ytbv_root", default="")
    parser.add_argument("--davis_root", default="")
    parser.add_argument("--output_json", default=str(REPO_ROOT / "DPO_finetune" / "captions" / "cococo_qwen_captions.json"))
    parser.add_argument("--model_path", default=str(REPO_ROOT / "weights" / "Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--num_videos", type=int, default=0, help="0 means all scanned videos.")
    parser.add_argument("--max_frames", type=int, default=48)
    parser.add_argument("--caption_frames", type=int, default=4)
    parser.add_argument("--min_frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--tile_size", type=int, default=336)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attn_implementation", default="")
    parser.add_argument("--fallback_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    ytbv_root = Path(args.ytbv_root) if args.ytbv_root else pick_first_dir([
        project_root / "data/external/ytbv_2019_full_resolution/train/JPEGImages",
        project_root / "data/external/youtubevos_432_240/JPEGImages_432_240",
    ])
    davis_root = Path(args.davis_root) if args.davis_root else pick_first_dir([
        project_root / "data/external/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution",
        project_root / "data/external/davis_432_240/JPEGImages_432_240",
    ])

    videos: List[VideoItem] = []
    videos.extend(scan_videos(davis_root, "davis", args.min_frames))
    videos.extend(scan_videos(ytbv_root, "ytbv", args.min_frames))
    videos = sorted(videos, key=lambda item: (item.source, item.name))
    rng = random.Random(args.seed)
    rng.shuffle(videos)
    if args.num_videos > 0:
        videos = videos[:args.num_videos]

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    captions: Dict[str, str] = {}
    if output_json.exists() and not args.overwrite:
        captions = json.loads(output_json.read_text(encoding="utf-8"))

    model = processor = None
    if not args.fallback_only:
        model, processor = load_qwen(
            Path(args.model_path),
            device_map=args.device_map,
            dtype_name=args.dtype,
            attn_implementation=args.attn_implementation,
        )

    print(f"[caption] videos={len(videos)} davis={davis_root} ytbv={ytbv_root}")
    print(f"[caption] output={output_json}")
    for idx, video in enumerate(videos, 1):
        if not args.overwrite and video.name in captions:
            print(f"[caption] skip existing {video.name}")
            continue
        fallback = fallback_caption(video)
        if args.fallback_only:
            caption = fallback
        else:
            paths = sample_frame_paths(video.frame_files, args.max_frames, args.caption_frames)
            grid = make_frame_grid(paths, tile_size=args.tile_size)
            raw = caption_with_qwen(model, processor, grid, args.max_new_tokens)
            caption = clean_caption(raw, fallback)
        captions[video.name] = caption
        captions[video.frame_dir.name] = caption
        tmp = output_json.with_suffix(output_json.suffix + ".tmp")
        tmp.write_text(json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(output_json)
        print(f"[caption] {idx}/{len(videos)} {video.name}: {caption}")


if __name__ == "__main__":
    main()
