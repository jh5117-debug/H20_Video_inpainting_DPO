#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a model-output-based DPO dataset without changing the training schema.

Output schema:
  output_root/
    manifest.json
    {video_id}/
      gt_frames/
      masks/
      candidates/{method}/raw_output/
      candidates/{method}/normalized_raw/
      candidates/{method}/composited/
      neg_frames_1/
      neg_frames_2/
      meta.json

The key design choice is conservative: every candidate is a complete output from
one inpainting model, composited outside the mask with GT, then scored and
selected with source balancing. This avoids the old hand-made negative problem
where the loser was too far from the winner and DPO learned the wrong shortcut.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import glob
import json
import math
import os
import random
import shutil
import shlex
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.metrics import compute_psnr, compute_ssim  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_VBENCH_DIMS = "subject_consistency,background_consistency,temporal_flickering,motion_smoothness,aesthetic_quality"
QUALITY_METRIC_SPECS: List[Tuple[str, bool, float]] = [
    ("psnr", True, 0.16),
    ("ssim", True, 0.14),
    ("lpips", False, 0.20),
    ("vbench_background_consistency", True, 0.14),
    ("vbench_temporal_flickering", True, 0.14),
    ("vbench_motion_smoothness", True, 0.08),
    ("vbench_imaging_quality", True, 0.10),
    ("vbench_aesthetic_quality", True, 0.06),
    ("vbench_subject_consistency", True, 0.06),
    ("vbench_inpainting_score", True, 0.08),
]
_LPIPS_LOCK = threading.Lock()
_VBENCH_LOCK = threading.Lock()
_VBENCH_SCORERS: Dict[Tuple[str, str], Any] = {}


@dataclass
class VideoItem:
    source: str
    name: str
    frame_dir: Path
    frame_files: List[Path]


@dataclass
class CandidateResult:
    method: str
    ok: bool
    raw_dir: str = ""
    comp_dir: str = ""
    score: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    prompt: str = ""
    quality_score: float = 0.0


def parse_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_method_gpu_map(value: str, methods: Sequence[str], fallback_gpus: Sequence[str]) -> Dict[str, List[str]]:
    if not fallback_gpus:
        raise ValueError("At least one GPU must be provided.")

    mapping: Dict[str, List[str]] = {}
    raw_items = value.replace("\n", ";").split(";") if value.strip() else []
    for raw_item in raw_items:
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid method_gpu_map item: {item!r}")
        method, gpu_text = item.split("=", 1)
        method = method.strip()
        gpus = parse_csv(gpu_text)
        if not method or not gpus:
            raise ValueError(f"Invalid method_gpu_map item: {item!r}")
        mapping[method] = gpus

    unknown_methods = sorted(set(mapping) - set(methods))
    if unknown_methods:
        raise ValueError(f"Unknown methods in method_gpu_map: {unknown_methods}")

    unknown_gpus = sorted({gpu for gpus in mapping.values() for gpu in gpus if gpu not in fallback_gpus})
    if unknown_gpus:
        raise ValueError(f"GPUs in method_gpu_map must be part of --gpus; unknown={unknown_gpus}")

    resolved: Dict[str, List[str]] = {}
    for idx, method in enumerate(methods):
        resolved[method] = list(mapping.get(method, [fallback_gpus[idx % len(fallback_gpus)]]))
    return resolved


class GpuDispatcher:
    def __init__(self, method_to_gpus: Dict[str, Sequence[str]], gpu_slots: Optional[Dict[str, int]] = None):
        unique_gpus = sorted({gpu for gpus in method_to_gpus.values() for gpu in gpus})
        if not unique_gpus:
            raise ValueError("GpuDispatcher requires at least one GPU.")
        self.method_to_gpus = {method: list(gpus) for method, gpus in method_to_gpus.items()}
        self._capacity = {gpu: max(1, int((gpu_slots or {}).get(gpu, 1))) for gpu in unique_gpus}
        self._busy = {gpu: 0 for gpu in unique_gpus}
        self._cond = threading.Condition()

    @contextlib.contextmanager
    def lease(self, method: str) -> Iterable[str]:
        gpu = self.acquire(method)
        try:
            yield gpu
        finally:
            self.release(gpu)

    def acquire(self, method: str) -> str:
        allowed = self.method_to_gpus.get(method)
        if not allowed:
            raise ValueError(f"No GPU assignment available for method {method!r}")

        with self._cond:
            while True:
                free = [gpu for gpu in allowed if self._busy[gpu] < self._capacity.get(gpu, 1)]
                if free:
                    chosen = sorted(free, key=lambda g: (self._busy[g], g))[0]
                    self._busy[chosen] += 1
                    return chosen
                self._cond.wait(timeout=0.5)

    def release(self, gpu: str) -> None:
        with self._cond:
            if gpu not in self._busy:
                return
            self._busy[gpu] = max(0, self._busy[gpu] - 1)
            self._cond.notify_all()


def parse_source_weights(value: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in parse_csv(value):
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        try:
            weights[key.strip()] = max(0.01, float(raw))
        except ValueError:
            continue
    return weights


def parse_gpu_slots(value: str, gpus: Sequence[str]) -> Dict[str, int]:
    if not gpus:
        raise ValueError("parse_gpu_slots requires at least one GPU.")
    text = str(value).strip()
    if not text:
        return {gpu: 1 for gpu in gpus}
    if text.isdigit():
        slots = max(1, int(text))
        return {gpu: slots for gpu in gpus}

    slots_by_gpu = {gpu: 1 for gpu in gpus}
    raw_items = text.replace("\n", ";").replace(",", ";").split(";")
    for raw_item in raw_items:
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid gpu_slots item: {item!r}; use '2' or '0=2;1=2'.")
        gpu, raw_slots = item.split("=", 1)
        gpu = gpu.strip()
        if gpu not in slots_by_gpu:
            raise ValueError(f"GPU in gpu_slots must be part of --gpus; unknown={gpu!r}")
        slots_by_gpu[gpu] = max(1, int(raw_slots.strip()))
    return slots_by_gpu


def image_files(path: Path) -> List[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def scan_videos(root: Path, source: str, min_frames: int) -> List[VideoItem]:
    """Find sequence directories up to two levels deep."""
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


def sample_contiguous(files: Sequence[Path], max_frames: int) -> List[Path]:
    if max_frames <= 0 or len(files) <= max_frames:
        return list(files)
    start = max(0, (len(files) - max_frames) // 2)
    return list(files[start:start + max_frames])


def center_crop_resize(img: np.ndarray, width: int, height: int, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
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


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))


def save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr.astype(np.uint8))


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def cleanup_candidate_storage(video_root_dir: Path, candidates: Sequence[CandidateResult], selected_methods: Sequence[str], retention: str) -> None:
    shutil.rmtree(video_root_dir / "_adapter_inputs", ignore_errors=True)
    candidates_root = video_root_dir / "candidates"
    if retention == "all" or not candidates_root.exists():
        return
    if retention == "none":
        shutil.rmtree(candidates_root, ignore_errors=True)
        return
    selected = set(selected_methods)
    for cand in candidates:
        method_dir = candidates_root / cand.method
        if cand.method not in selected and method_dir.exists():
            shutil.rmtree(method_dir, ignore_errors=True)


def prepare_gt_frames(video: VideoItem, out_dir: Path, width: int, height: int, max_frames: int) -> List[Path]:
    files = sample_contiguous(video.frame_files, max_frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for i, frame_path in enumerate(files):
        frame = center_crop_resize(read_rgb(frame_path), width, height)
        dst = out_dir / f"{i:05d}.png"
        save_rgb(dst, frame)
        saved.append(dst)
    return saved


def random_polygon(width: int, height: int, rng: random.Random, vertices_min: int = 5, vertices_max: int = 8) -> np.ndarray:
    n = rng.randint(vertices_min, vertices_max)
    step = 2 * math.pi / n
    phase = rng.random() * step
    jitter = step * 0.22
    angles = sorted(phase + i * step + rng.uniform(-jitter, jitter) for i in range(n))
    pts = []
    for a in angles:
        # Push points toward a square-like boundary rather than a circle.
        # This keeps 40%-50% masks large and central without forcing the bbox
        # to touch image borders, even with only 5-8 vertices.
        c = math.cos(a)
        s = math.sin(a)
        boundary = 1.0 / max(abs(c), abs(s), 1e-6)
        r = rng.uniform(0.86, 1.0)
        x = c * width * 0.5 * boundary * r
        y = s * height * 0.5 * boundary * r
        pts.append((x, y))
    return np.array(pts, dtype=np.float32)


def rasterize_polygon(points: np.ndarray, width: int, height: int, dilation_iter: int) -> np.ndarray:
    pil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(pil)
    draw.polygon([tuple(p) for p in points.tolist()], fill=255)
    mask = np.array(pil, dtype=np.uint8)
    if dilation_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilation_iter)
    return mask


def fit_polygon_to_area(
    base: np.ndarray,
    width: int,
    height: int,
    target_area_ratio: float,
    dilation_iter: int,
) -> Tuple[np.ndarray, float]:
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    lo, hi = 0.2, min(width, height)
    best = base.copy()
    best_ratio = 0.0
    for _ in range(26):
        scale = (lo + hi) / 2.0
        pts = base * scale + center
        mask = rasterize_polygon(pts, width, height, dilation_iter)
        ratio = float(np.count_nonzero(mask)) / float(width * height)
        best = base * scale
        best_ratio = ratio
        if ratio < target_area_ratio:
            lo = scale
        else:
            hi = scale
    return best.astype(np.float32), best_ratio


def recenter_polygon_by_bbox(points: np.ndarray) -> np.ndarray:
    center = np.array(
        [
            (float(points[:, 0].min()) + float(points[:, 0].max())) / 2.0,
            (float(points[:, 1].min()) + float(points[:, 1].max())) / 2.0,
        ],
        dtype=np.float32,
    )
    return (points - center).astype(np.float32)


def polygon_center_bounds(points: np.ndarray, width: int, height: int, pad: int = 0) -> Tuple[float, float, float, float]:
    min_x = float(points[:, 0].min())
    max_x = float(points[:, 0].max())
    min_y = float(points[:, 1].min())
    max_y = float(points[:, 1].max())
    cx_min = pad - min_x
    cx_max = (width - 1 - pad) - max_x
    cy_min = pad - min_y
    cy_max = (height - 1 - pad) - max_y
    if cx_min > cx_max:
        cx_min, cx_max = width / 2.0, width / 2.0
    if cy_min > cy_max:
        cy_min, cy_max = height / 2.0, height / 2.0
    return cx_min, cy_min, cx_max, cy_max


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    y, x = np.where(mask > 0)
    if not len(x):
        return None
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def mask_bbox_ratio(mask: np.ndarray) -> List[float]:
    bbox = mask_bbox(mask)
    if bbox is None:
        return [0.0, 0.0]
    x0, y0, x1, y1 = bbox
    return [
        float((x1 - x0) / mask.shape[1]),
        float((y1 - y0) / mask.shape[0]),
    ]


def mask_bbox_center_ratio(mask: np.ndarray) -> List[float]:
    bbox = mask_bbox(mask)
    if bbox is None:
        return [0.5, 0.5]
    x0, y0, x1, y1 = bbox
    return [
        float(((x0 + x1) / 2.0) / mask.shape[1]),
        float(((y0 + y1) / 2.0) / mask.shape[0]),
    ]


def mask_bbox_margin_ratio(mask: np.ndarray) -> List[float]:
    bbox = mask_bbox(mask)
    if bbox is None:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = bbox
    h, w = mask.shape[:2]
    return [
        float(x0 / w),
        float(y0 / h),
        float((w - x1) / w),
        float((h - y1) / h),
    ]


def generate_hard_masks(
    out_dir: Path,
    num_frames: int,
    width: int,
    height: int,
    seed: int,
    dilation_iter: int,
    margin_ratio: float = 0.15,
    area_min: float = 0.35,
    area_max: float = 0.45,
    static_prob: float = 0.50,
    speed_min: float = 0.50,
    speed_max: float = 1.50,
    center_jitter_ratio: float = 0.04,
    motion_box_ratio: float = 0.16,
) -> Dict[str, Any]:
    """Generate and save large, central, reproducible hard-negative masks."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_area_ratio = rng.uniform(area_min, area_max)
    unit_base = random_polygon(2, 2, rng)
    base, fitted_area_ratio = fit_polygon_to_area(unit_base, width, height, target_area_ratio, dilation_iter)
    base = recenter_polygon_by_bbox(base)

    pad = max(0, dilation_iter + 2)
    cx_min, cy_min, cx_max, cy_max = polygon_center_bounds(base, width, height, pad=pad)
    motion_half_x = max(1.0, width * motion_box_ratio / 2.0)
    motion_half_y = max(1.0, height * motion_box_ratio / 2.0)
    motion_cx_min = max(cx_min, width / 2.0 - motion_half_x)
    motion_cx_max = min(cx_max, width / 2.0 + motion_half_x)
    motion_cy_min = max(cy_min, height / 2.0 - motion_half_y)
    motion_cy_max = min(cy_max, height / 2.0 + motion_half_y)
    if motion_cx_min > motion_cx_max:
        center_x = min(max(width / 2.0, cx_min), cx_max)
        motion_cx_min = motion_cx_max = center_x
    if motion_cy_min > motion_cy_max:
        center_y = min(max(height / 2.0, cy_min), cy_max)
        motion_cy_min = motion_cy_max = center_y

    center_jitter_x = center_jitter_ratio * width
    center_jitter_y = center_jitter_ratio * height
    cx = min(max(width / 2.0 + rng.uniform(-center_jitter_x, center_jitter_x), motion_cx_min), motion_cx_max)
    cy = min(max(height / 2.0 + rng.uniform(-center_jitter_y, center_jitter_y), motion_cy_min), motion_cy_max)

    if rng.random() < static_prob:
        vx = vy = 0.0
        motion_type = "static"
    else:
        speed = rng.uniform(speed_min, speed_max)
        theta = rng.uniform(0, 2 * math.pi)
        vx = math.cos(theta) * speed
        vy = math.sin(theta) * speed
        motion_type = "slow"

    frame_meta = []
    area_ratios = []
    bbox_ratios = []
    bbox_center_ratios = []
    bbox_margin_ratios = []
    initial_velocity = [vx, vy]
    for t in range(num_frames):
        pts = base + np.array([cx, cy], dtype=np.float32)
        mask = rasterize_polygon(pts, width, height, dilation_iter)
        save_gray(out_dir / f"{t:05d}.png", mask)

        area_ratio = float(np.count_nonzero(mask)) / float(width * height)
        bbox_ratio = mask_bbox_ratio(mask)
        bbox_center = mask_bbox_center_ratio(mask)
        bbox_margin = mask_bbox_margin_ratio(mask)
        area_ratios.append(area_ratio)
        bbox_ratios.append(bbox_ratio)
        bbox_center_ratios.append(bbox_center)
        bbox_margin_ratios.append(bbox_margin)
        frame_meta.append(
            {
                "frame": t,
                "center": [round(cx, 3), round(cy, 3)],
                "area_ratio": round(area_ratio, 6),
                "bbox_ratio": [round(float(bbox_ratio[0]), 6), round(float(bbox_ratio[1]), 6)],
                "bbox_center_ratio": [round(float(bbox_center[0]), 6), round(float(bbox_center[1]), 6)],
                "bbox_margin_ratio": [round(float(x), 6) for x in bbox_margin],
            }
        )

        nx = cx + vx
        ny = cy + vy
        if nx < motion_cx_min or nx > motion_cx_max:
            vx = -vx
            nx = cx + vx
        if ny < motion_cy_min or ny > motion_cy_max:
            vy = -vy
            ny = cy + vy
        cx = min(max(nx, motion_cx_min), motion_cx_max)
        cy = min(max(ny, motion_cy_min), motion_cy_max)

    mean_bbox_ratio = [
        float(np.mean([b[0] for b in bbox_ratios])) if bbox_ratios else 0.0,
        float(np.mean([b[1] for b in bbox_ratios])) if bbox_ratios else 0.0,
    ]
    mean_bbox_center_ratio = [
        float(np.mean([b[0] for b in bbox_center_ratios])) if bbox_center_ratios else 0.5,
        float(np.mean([b[1] for b in bbox_center_ratios])) if bbox_center_ratios else 0.5,
    ]
    min_bbox_margin_ratio = [
        float(np.min([b[i] for b in bbox_margin_ratios])) if bbox_margin_ratios else 0.0
        for i in range(4)
    ]

    return {
        "seed": seed,
        "motion_type": motion_type,
        "motion": motion_type,
        "velocity": [round(initial_velocity[0], 6), round(initial_velocity[1], 6)],
        "target_area_ratio": target_area_ratio,
        "area_ratio": float(np.mean(area_ratios)) if area_ratios else fitted_area_ratio,
        "area_ratio_min": float(np.min(area_ratios)) if area_ratios else fitted_area_ratio,
        "area_ratio_max": float(np.max(area_ratios)) if area_ratios else fitted_area_ratio,
        "area_ratio_range": [area_min, area_max],
        "bbox_ratio": mean_bbox_ratio,
        "bbox_center_ratio": mean_bbox_center_ratio,
        "bbox_margin_ratio_min": min_bbox_margin_ratio,
        "vertices": int(len(base)),
        "margin_ratio": margin_ratio,
        "center_bounds": [cx_min / width, cy_min / height, cx_max / width, cy_max / height],
        "motion_center_bounds": [motion_cx_min / width, motion_cy_min / height, motion_cx_max / width, motion_cy_max / height],
        "center_jitter_ratio": center_jitter_ratio,
        "motion_box_ratio": motion_box_ratio,
        "dilation_iter": dilation_iter,
        "frames": frame_meta,
    }


def load_captions(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    raise ValueError(f"caption_json must be a dict, got {type(data).__name__}")


def prompt_for(video: VideoItem, video_id: str, captions: Dict[str, str]) -> str:
    for key in (video_id, video.name, video.frame_dir.name):
        if key in captions and captions[key].strip():
            return captions[key].strip()
    words = video.frame_dir.name.replace("_", " ").replace("-", " ")
    return f"clean realistic background restoration in a video of {words}"


def load_adapter_config(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Adapter config not found: {path}. Copy multimodel_adapters_h20.example.json "
            "to multimodel_adapters_h20.json and fill third-party commands."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): dict(v) for k, v in data.items()}


def expand_brace_glob(pattern: str) -> List[str]:
    if "{" not in pattern or "}" not in pattern:
        return [pattern]
    start = pattern.index("{")
    end = pattern.index("}", start)
    opts = pattern[start + 1:end].split(",")
    return [pattern[:start] + opt + pattern[end + 1:] for opt in opts]


def collect_frames_by_glob(pattern: str) -> List[Path]:
    files: List[Path] = []
    for p in expand_brace_glob(pattern):
        files.extend(Path(x) for x in glob.glob(p, recursive=True))
    files = [p for p in files if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(set(files))


def format_template(template: str, values: Dict[str, Any]) -> str:
    return template.format(**{k: str(v) for k, v in values.items()})


def build_template_values(
    args: argparse.Namespace,
    video_id: str,
    method: str,
    video_root: Path,
    mask_root: Path,
    input_dir: Path,
    mask_dir: Path,
    raw_output_dir: Path,
    method_work_dir: Path,
    prompt: str,
    gpu: str,
    num_frames: int,
) -> Dict[str, Any]:
    return {
        "project_root": Path(args.project_root),
        "third_party_root": Path(args.third_party_root),
        "diffueraser_env": args.diffueraser_env,
        "video_id": video_id,
        "method": method,
        "batch_video_root": video_root,
        "batch_mask_root": mask_root,
        "input_dir": input_dir,
        "mask_dir": mask_dir,
        "raw_output_dir": raw_output_dir,
        "method_work_dir": method_work_dir,
        "prompt": prompt.replace('"', '\\"'),
        "gpu": gpu,
        "num_frames": num_frames,
        "width": args.width,
        "height": args.height,
    }


def run_command(cmd: str, cwd: Optional[Path], env_path: Optional[str], gpu: str, log_path: Path) -> None:
    shell_cmd = cmd
    if env_path:
        conda_bin = os.environ.get("CONDA_EXE") or "/home/nvme01/miniconda3/bin/conda"
        if conda_bin != "conda" and not Path(conda_bin).exists():
            conda_bin = "conda"
        shell_cmd = f"{shlex.quote(conda_bin)} run --no-capture-output -p {shlex.quote(env_path)} {cmd}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if env_path:
        env["PYTHONNOUSERSITE"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shell_cmd}\n")
        log.write(f"cwd={cwd or os.getcwd()}\nCUDA_VISIBLE_DEVICES={gpu}\n\n")
        log.flush()
        proc = subprocess.run(
            shell_cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}; see {log_path}")


def normalize_candidate_frames(frame_files: List[Path], dst_dir: Path, num_frames: int, width: int, height: int) -> List[Path]:
    if not frame_files:
        raise RuntimeError("no output frames found")
    dst_dir.mkdir(parents=True, exist_ok=True)
    chosen = frame_files[:num_frames]
    if len(chosen) < num_frames:
        chosen = chosen + [chosen[-1]] * (num_frames - len(chosen))
    saved = []
    for i, src in enumerate(chosen):
        arr = center_crop_resize(read_rgb(src), width, height)
        dst = dst_dir / f"{i:05d}.png"
        save_rgb(dst, arr)
        saved.append(dst)
    return saved


def composite_candidate(raw_dir: Path, gt_dir: Path, mask_dir: Path, comp_dir: Path) -> List[Path]:
    raw_files = image_files(raw_dir)
    gt_files = image_files(gt_dir)
    mask_files = image_files(mask_dir)
    n = min(len(raw_files), len(gt_files), len(mask_files))
    if n == 0:
        raise RuntimeError("cannot composite empty candidate")
    comp_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in range(n):
        pred = read_rgb(raw_files[i])
        gt = read_rgb(gt_files[i])
        mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {mask_files[i]}")
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != gt.shape[:2]:
            mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (mask > 0).astype(np.float32)[:, :, None]
        comp = pred.astype(np.float32) * m + gt.astype(np.float32) * (1.0 - m)
        dst = comp_dir / f"{i:05d}.png"
        save_rgb(dst, np.clip(comp, 0, 255).astype(np.uint8))
        saved.append(dst)
    return saved


def select_score_indices(n: int, score_windows: List[int]) -> Tuple[List[int], int]:
    for win in score_windows:
        if n >= win:
            start = max(0, (n - win) // 2)
            return list(range(start, start + win)), win
    return list(range(n)), n


def union_mask_bbox(mask_files: Sequence[Path], pad: int = 8) -> Optional[Tuple[int, int, int, int]]:
    xs: List[int] = []
    ys: List[int] = []
    h = w = None
    for path in mask_files:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape[:2]
        y, x = np.where(mask > 0)
        if len(x):
            xs.extend([int(x.min()), int(x.max())])
            ys.extend([int(y.min()), int(y.max())])
    if not xs or h is None or w is None:
        return None
    x0 = max(0, min(xs) - pad)
    x1 = min(w, max(xs) + pad + 1)
    y0 = max(0, min(ys) - pad)
    y1 = min(h, max(ys) + pad + 1)
    return x0, y0, x1, y1


def frame_metrics(
    gt_files: Sequence[Path],
    comp_files: Sequence[Path],
    mask_files: Sequence[Path],
    score_windows: List[int],
    enable_lpips: bool,
    lpips_device: str,
) -> Dict[str, Any]:
    n = min(len(gt_files), len(comp_files), len(mask_files))
    indices, win = select_score_indices(n, score_windows)
    bbox = union_mask_bbox([mask_files[i] for i in indices])

    psnrs: List[float] = []
    ssims: List[float] = []
    lpips_vals: List[float] = []

    lpips_metric = None
    if enable_lpips:
        try:
            from inference.metrics import LPIPSMetric
            lpips_metric = LPIPSMetric
        except Exception as exc:
            print(f"[warn] LPIPS disabled: {exc}")
            lpips_metric = None

    for i in indices:
        gt = read_rgb(gt_files[i])
        comp = read_rgb(comp_files[i])
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            gt_eval = gt[y0:y1, x0:x1]
            comp_eval = comp[y0:y1, x0:x1]
        else:
            gt_eval = gt
            comp_eval = comp
        psnrs.append(compute_psnr(gt_eval, comp_eval))
        ssims.append(compute_ssim(gt_eval, comp_eval))
        if lpips_metric is not None:
            try:
                with _LPIPS_LOCK:
                    lpips_vals.append(float(lpips_metric.compute(gt_eval, comp_eval, device=lpips_device)))
            except Exception as exc:
                print(f"[warn] LPIPS frame {i} failed: {exc}")
                lpips_metric = None

    result: Dict[str, Any] = {
        "score_window": win,
        "score_frame_count": len(indices),
        "mask_bbox": list(bbox) if bbox is not None else None,
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim": float(np.mean(ssims)) if ssims else 0.0,
    }
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    return result


def encode_mp4(frame_dir: Path, out_path: Path, fps: int = 8) -> Path:
    files = image_files(frame_dir)
    if not files:
        raise RuntimeError(f"no frames to encode: {frame_dir}")
    first = read_rgb(files[0])
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fp in files:
        arr = read_rgb(fp)
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    writer.release()
    return out_path


def make_side_by_side_preview(
    gt_dir: Path,
    mask_dir: Path,
    raw_dir: Path,
    comp_dir: Path,
    out_path: Path,
    fps: int = 8,
) -> Path:
    gt_files = image_files(gt_dir)
    mask_files = image_files(mask_dir)
    raw_files = image_files(raw_dir)
    comp_files = image_files(comp_dir)
    n = min(len(gt_files), len(mask_files), len(raw_files), len(comp_files))
    if n == 0:
        raise RuntimeError("no frames to preview")

    first = read_rgb(gt_files[0])
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 4, h))

    for i in range(n):
        gt = read_rgb(gt_files[i])
        raw = read_rgb(raw_files[i])
        comp = read_rgb(comp_files[i])
        mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {mask_files[i]}")
        if raw.shape[:2] != (h, w):
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        if comp.shape[:2] != (h, w):
            comp = cv2.resize(comp, (w, h), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_overlay = gt.copy()
        red = np.zeros_like(gt)
        red[:, :, 0] = 255
        m = (mask > 0)[:, :, None]
        mask_overlay = np.where(m, (0.55 * gt + 0.45 * red).astype(np.uint8), mask_overlay)

        panels = [gt, mask_overlay, raw, comp]
        labels = ["GT", "MASK", "RAW_OUT", "COMPOSITED"]
        labeled = []
        for panel, label in zip(panels, labels):
            bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            labeled.append(bgr)
        writer.write(np.concatenate(labeled, axis=1))
    writer.release()
    return out_path


def mean_existing(values: Sequence[Optional[float]]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return 0.5
    return float(np.mean(vals))


def maybe_vbench_score(
    frame_dir: Path,
    name: str,
    device: str,
    work_dir: Path,
    dimensions: str,
) -> Dict[str, Any]:
    try:
        from tools.score_inpainting_quality import InpaintingScorer
        video_path = encode_mp4(frame_dir, work_dir / f"{name}.mp4")
        dims = parse_csv(dimensions) or parse_csv(DEFAULT_VBENCH_DIMS)
        key = (device, ",".join(dims))
        with _VBENCH_LOCK:
            scorer = _VBENCH_SCORERS.get(key)
            if scorer is None:
                scorer = InpaintingScorer(device=device, dimensions=dims)
                _VBENCH_SCORERS[key] = scorer
            return scorer.score_video(str(video_path), name=name)
    except Exception as exc:
        return {"error": str(exc), "inpainting_score": None}


def resolve_local_cuda_device(gpu: str, enabled: bool = True, cpu_fallback: bool = False) -> str:
    if not enabled:
        return "cpu" if cpu_fallback else "cuda"
    gpu_text = str(gpu).strip()
    if not gpu_text.isdigit():
        return "cuda"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        visible_list = [item.strip() for item in visible.split(",") if item.strip()]
        if gpu_text in visible_list:
            return f"cuda:{visible_list.index(gpu_text)}"
        if int(gpu_text) < len(visible_list):
            return f"cuda:{gpu_text}"
    return f"cuda:{gpu_text}"


def score_candidate(
    method: str,
    gt_dir: Path,
    comp_dir: Path,
    mask_dir: Path,
    args: argparse.Namespace,
    gpu: str,
) -> Dict[str, Any]:
    gt_files = image_files(gt_dir)
    comp_files = image_files(comp_dir)
    mask_files = image_files(mask_dir)
    score = frame_metrics(
        gt_files,
        comp_files,
        mask_files,
        [int(x) for x in parse_csv(args.score_windows)],
        args.enable_lpips,
        lpips_device=resolve_local_cuda_device(gpu, enabled=args.enable_lpips, cpu_fallback=True),
    )
    if args.enable_vbench:
        vbench_device = resolve_local_cuda_device(gpu, enabled=True, cpu_fallback=False)
        vb = maybe_vbench_score(comp_dir, method, vbench_device, comp_dir.parent / "_videos", args.vbench_dimensions)
        score["vbench"] = vb
        if vb.get("error"):
            raise RuntimeError(f"VBench failed for {method}: {vb.get('error')}")
        if vb.get("inpainting_score") is not None:
            score["vbench_inpainting_score"] = float(vb["inpainting_score"])
        for dim, value in (vb.get("per_dim") or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0:
                score[f"vbench_{dim}"] = float(value)
    return score


def assign_relative_quality(candidates: List[CandidateResult]) -> None:
    ok = [c for c in candidates if c.ok]
    if not ok:
        return

    available: List[Tuple[str, bool, float, float, float]] = []
    for key, higher, weight in QUALITY_METRIC_SPECS:
        vals = [c.score.get(key) for c in ok]
        vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if vals:
            available.append((key, higher, weight, float(min(vals)), float(max(vals))))

    if not available:
        for c in ok:
            c.quality_score = 0.5
            c.score["relative_quality_score"] = 0.5
            c.score["defect_bucket"] = "unscored"
        return

    total_w = sum(w for _, _, w, _, _ in available)
    for c in ok:
        q = 0.0
        metric_norms: Dict[str, float] = {}
        for key, higher, weight, lo, hi in available:
            val = float(c.score.get(key, lo if higher else hi))
            if hi == lo:
                norm = 0.5
            else:
                norm = (val - lo) / (hi - lo)
            if not higher:
                norm = 1.0 - norm
            norm = float(max(0.0, min(1.0, norm)))
            metric_norms[key] = norm
            q += (weight / total_w) * norm
        c.quality_score = float(q)
        c.score["relative_quality_score"] = c.quality_score
        c.score["metric_norms"] = metric_norms

        pixel_quality = mean_existing([
            metric_norms.get("psnr"),
            metric_norms.get("ssim"),
            metric_norms.get("lpips"),
        ])
        temporal_quality = mean_existing([
            metric_norms.get("vbench_background_consistency"),
            metric_norms.get("vbench_temporal_flickering"),
            metric_norms.get("vbench_motion_smoothness"),
        ])
        perceptual_quality = mean_existing([
            metric_norms.get("vbench_imaging_quality"),
            metric_norms.get("vbench_aesthetic_quality"),
            metric_norms.get("vbench_subject_consistency"),
        ])
        c.score["pixel_quality_norm"] = pixel_quality
        c.score["temporal_quality_norm"] = temporal_quality
        c.score["perceptual_quality_norm"] = perceptual_quality

        if c.quality_score >= 0.80:
            bucket = "too_good"
        elif c.quality_score <= 0.20:
            bucket = "too_bad"
        elif pixel_quality >= 0.65 and perceptual_quality <= 0.45:
            bucket = "blur_or_texture_loss_high_pixel"
        elif temporal_quality <= 0.35 and pixel_quality >= 0.40:
            bucket = "temporal_inconsistent"
        elif pixel_quality <= 0.35 and perceptual_quality >= 0.55:
            bucket = "semantic_or_content_shift"
        else:
            bucket = "balanced_hard"
        c.score["defect_bucket"] = bucket


def select_negatives(
    candidates: List[CandidateResult],
    source_counts: Dict[str, int],
    source_weights: Dict[str, float],
    source_quality_max_overrides: Dict[str, float],
    quality_min: float,
    quality_max: float,
    quality_target: float,
) -> Tuple[CandidateResult, CandidateResult, Dict[str, Any]]:
    ok = [c for c in candidates if c.ok]
    if not ok:
        raise RuntimeError("need at least one successful model candidate")

    ok_sorted = sorted(ok, key=lambda c: c.quality_score)
    if len(ok_sorted) == 1:
        only = ok_sorted[0]
        source_counts[only.method] = source_counts.get(only.method, 0) + 2
        return only, only, {
            "policy": "single_successful_source_duplicated",
            "note": "Only one model candidate succeeded; duplicated it into neg_frames_1 and neg_frames_2 for smoke-test compatibility.",
            "source_counts_after_selection": dict(source_counts),
            "source_selection_weights": dict(source_weights),
            "candidate_quality_order": [
                {"method": only.method, "quality": only.quality_score, "score": only.score}
            ],
            "trimmed_extremes": False,
        }

    def source_quality_max(cand: CandidateResult) -> float:
        return source_quality_max_overrides.get(cand.method, quality_max)

    eligible = [
        c for c in ok_sorted
        if quality_min <= c.quality_score <= source_quality_max(c)
    ]
    if len(eligible) < 2:
        eligible_ids = {id(c) for c in eligible}
        outside = [c for c in ok_sorted if id(c) not in eligible_ids]

        def band_distance(cand: CandidateResult) -> float:
            if cand.quality_score < quality_min:
                return quality_min - cand.quality_score
            max_allowed = source_quality_max(cand)
            if cand.quality_score > max_allowed:
                return cand.quality_score - max_allowed
            return 0.0

        outside = sorted(outside, key=lambda c: (band_distance(c), abs(c.quality_score - quality_target)))
        eligible = eligible + outside[: max(0, 2 - len(eligible))]
    if len(eligible) < 2:
        eligible = ok_sorted

    def source_pressure(cand: CandidateResult) -> float:
        return source_counts.get(cand.method, 0) / max(0.01, source_weights.get(cand.method, 1.0))

    # Prefer under-represented source models first, then hard-but-not-catastrophic candidates.
    ranked = sorted(
        eligible,
        key=lambda c: (
            source_pressure(c),
            abs(c.quality_score - quality_target),
            c.quality_score,
        ),
    )
    first = ranked[0]
    first_bucket = first.score.get("defect_bucket")
    second = sorted(
        ranked[1:],
        key=lambda c: (
            c.method == first.method,
            c.score.get("defect_bucket") == first_bucket,
            source_pressure(c),
            abs(c.quality_score - quality_target),
            c.quality_score,
        ),
    )[0]

    source_counts[first.method] = source_counts.get(first.method, 0) + 1
    source_counts[second.method] = source_counts.get(second.method, 0) + 1
    eligible_ids = {id(c) for c in eligible}

    policy = {
        "policy": "balanced_source_hard_plausible",
        "note": "GT is always the DPO winner; selected negatives are complete outputs from source models.",
        "quality_band": [quality_min, quality_max],
        "quality_target": quality_target,
        "source_counts_after_selection": dict(source_counts),
        "source_selection_weights": dict(source_weights),
        "source_quality_max_overrides": dict(source_quality_max_overrides),
        "candidate_quality_order": [
            {"method": c.method, "quality": c.quality_score, "score": c.score}
            for c in ok_sorted
        ],
        "eligible_candidate_order": [
            {"method": c.method, "quality": c.quality_score, "bucket": c.score.get("defect_bucket")}
            for c in ranked
        ],
        "filtered_out_candidates": [
            {"method": c.method, "quality": c.quality_score, "bucket": c.score.get("defect_bucket")}
            for c in ok_sorted
            if id(c) not in eligible_ids
        ],
        "trimmed_extremes": len(eligible) < len(ok_sorted),
    }
    return first, second, policy


def run_method(
    method: str,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    video_id: str,
    video_root: Path,
    mask_root: Path,
    gt_dir: Path,
    mask_dir: Path,
    method_root: Path,
    prompt: str,
    gpu: str,
    num_frames: int,
) -> CandidateResult:
    raw_output_dir = method_root / "raw_output"
    method_work_dir = method_root / "work"
    normalized_raw_dir = method_root / "normalized_raw"
    comp_dir = method_root / "composited"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    method_work_dir.mkdir(parents=True, exist_ok=True)

    result = CandidateResult(method=method, ok=False, prompt=prompt)
    try:
        if not cfg.get("enabled", False):
            raise RuntimeError("adapter disabled in config")
        cmd_template = str(cfg.get("cmd", ""))
        if not cmd_template or "TODO_" in cmd_template:
            raise RuntimeError("adapter command is still TODO")

        values = build_template_values(
            args=args,
            video_id=video_id,
            method=method,
            video_root=video_root,
            mask_root=mask_root,
            input_dir=gt_dir,
            mask_dir=mask_dir,
            raw_output_dir=raw_output_dir,
            method_work_dir=method_work_dir,
            prompt=prompt,
            gpu=gpu,
            num_frames=num_frames,
        )

        if not args.skip_inference:
            cmd = format_template(cmd_template, values)
            cwd = Path(format_template(str(cfg["cwd"]), values)) if cfg.get("cwd") else None
            env_path = format_template(str(cfg["env"]), values) if cfg.get("env") else None
            run_command(cmd, cwd=cwd, env_path=env_path, gpu=gpu, log_path=method_root / "inference.log")

        output_glob = cfg.get("output_glob")
        if output_glob:
            frame_files = collect_frames_by_glob(format_template(str(output_glob), values))
        else:
            frame_files = collect_frames_by_glob(str(raw_output_dir / "**" / "*.{png,jpg,jpeg}"))

        normalize_candidate_frames(frame_files, normalized_raw_dir, num_frames, args.width, args.height)
        composite_candidate(normalized_raw_dir, gt_dir, mask_dir, comp_dir)

        score = score_candidate(method, gt_dir, comp_dir, mask_dir, args, gpu)
        if args.save_previews:
            preview_dir = method_root / "previews"
            previews = {}
            try:
                previews["raw_mp4"] = str(encode_mp4(normalized_raw_dir, preview_dir / "raw_output.mp4"))
                previews["composited_mp4"] = str(encode_mp4(comp_dir, preview_dir / "composited.mp4"))
                previews["side_by_side_mp4"] = str(
                    make_side_by_side_preview(
                        gt_dir,
                        mask_dir,
                        normalized_raw_dir,
                        comp_dir,
                        preview_dir / "gt_mask_raw_comp.mp4",
                    )
                )
                score["previews"] = previews
            except Exception as exc:
                score["preview_error"] = str(exc)
        result.ok = True
        result.raw_dir = str(normalized_raw_dir)
        result.comp_dir = str(comp_dir)
        result.score = score
        return result
    except Exception as exc:
        result.error = str(exc)
        return result


def write_manifest(output_root: Path, manifest: Dict[str, Any]) -> None:
    tmp = output_root / "manifest.json.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp.replace(output_root / "manifest.json")


def seed_source_counts_from_existing(output_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for meta_path in sorted(output_root.glob("*/meta.json")):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        for key in ("selected_neg_frames_1", "selected_neg_frames_2"):
            method = meta.get(key)
            if method:
                counts[method] = counts.get(method, 0) + 1
    return counts


def process_one_video(
    video: VideoItem,
    mask_seed: int,
    args: argparse.Namespace,
    adapter_config: Dict[str, Dict[str, Any]],
    captions: Dict[str, str],
    source_counts: Dict[str, int],
    gpu_dispatcher: GpuDispatcher,
    selection_lock: threading.Lock,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    video_id = f"{video.name}_mask{mask_seed}"
    video_root_dir = Path(args.output_root) / video_id
    complete_on_disk = (
        (video_root_dir / "meta.json").exists()
        and (video_root_dir / "gt_frames").is_dir()
        and (video_root_dir / "masks").is_dir()
        and (video_root_dir / "neg_frames_1").is_dir()
        and (video_root_dir / "neg_frames_2").is_dir()
    )
    if args.resume and complete_on_disk:
        print(f"[skip] {video_id} already complete")
        with (video_root_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return video_id, {
            "gt_frames": f"{video_id}/gt_frames",
            "masks": f"{video_id}/masks",
            "mask_meta": f"{video_id}/mask_meta.json",
            "neg_frames_1": f"{video_id}/neg_frames_1",
            "neg_frames_2": f"{video_id}/neg_frames_2",
            "num_frames": int(meta.get("num_frames", 0)),
            "source": video.source,
            "source_frame_dir": meta.get("source_frame_dir", str(video.frame_dir)),
            "mask_seed": (meta.get("mask") or {}).get("seed"),
            "neg_1_source": meta.get("selected_neg_frames_1"),
            "neg_2_source": meta.get("selected_neg_frames_2"),
        }

    if args.resume and video_root_dir.exists() and not complete_on_disk:
        print(f"[resume] {video_id} is incomplete; removing partial directory and regenerating")
        shutil.rmtree(video_root_dir, ignore_errors=True)

    if video_root_dir.exists() and not args.resume:
        shutil.rmtree(video_root_dir)
    video_root_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = video_root_dir / "gt_frames"
    mask_dir = video_root_dir / "masks"
    gt_files = prepare_gt_frames(video, gt_dir, args.width, args.height, args.max_frames)
    num_frames = len(gt_files)
    if num_frames < args.train_nframes:
        print(f"[skip] {video_id}: too few frames after sampling")
        if args.cleanup_failed:
            shutil.rmtree(video_root_dir, ignore_errors=True)
        return None

    mask_meta = generate_hard_masks(
        mask_dir,
        num_frames=num_frames,
        width=args.width,
        height=args.height,
        seed=mask_seed,
        dilation_iter=args.mask_dilation_iter,
        margin_ratio=args.mask_margin_ratio,
        area_min=args.mask_area_min,
        area_max=args.mask_area_max,
        static_prob=args.mask_static_prob,
        speed_min=args.mask_speed_min,
        speed_max=args.mask_speed_max,
        center_jitter_ratio=args.mask_center_jitter_ratio,
        motion_box_ratio=args.mask_motion_box_ratio,
    )
    with (video_root_dir / "mask_meta.json").open("w", encoding="utf-8") as f:
        json.dump(mask_meta, f, indent=2, ensure_ascii=False)

    # Some third-party CLIs expect a root containing one sequence directory.
    batch_video_root = video_root_dir / "_adapter_inputs" / "videos"
    batch_mask_root = video_root_dir / "_adapter_inputs" / "masks"
    batch_video_dir = batch_video_root / video_id
    batch_mask_dir = batch_mask_root / video_id
    copytree_clean(gt_dir, batch_video_dir)
    copytree_clean(mask_dir, batch_mask_dir)

    prompt = prompt_for(video, video_id, captions)
    methods = parse_csv(args.methods)
    candidates_root = video_root_dir / "candidates"
    candidate_results: List[CandidateResult] = []

    def _job(idx_method: Tuple[int, str]) -> CandidateResult:
        _, method = idx_method
        cfg = copy.deepcopy(adapter_config.get(method, {}))
        with gpu_dispatcher.lease(method) as gpu:
            print(f"[infer] {video_id} {method} on GPU {gpu}")
            return run_method(
                method=method,
                cfg=cfg,
                args=args,
                video_id=video_id,
                video_root=batch_video_root,
                mask_root=batch_mask_root,
                gt_dir=gt_dir,
                mask_dir=mask_dir,
                method_root=candidates_root / method,
                prompt=prompt,
                gpu=gpu,
                num_frames=num_frames,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel_methods)) as pool:
        for result in pool.map(_job, enumerate(methods)):
            candidate_results.append(result)
            status = "ok" if result.ok else f"failed: {result.error}"
            print(f"[candidate] {video_id} {result.method}: {status}")

    assign_relative_quality(candidate_results)
    try:
        with selection_lock:
            neg1, neg2, selection_policy = select_negatives(
                candidate_results,
                source_counts,
                parse_source_weights(args.source_selection_weights),
                parse_source_weights(args.source_quality_max_overrides),
                args.neg_quality_min,
                args.neg_quality_max,
                args.neg_quality_target,
            )
    except Exception as exc:
        meta = {
            "video_id": video_id,
            "source": video.source,
            "source_frame_dir": str(video.frame_dir),
            "num_frames": num_frames,
            "prompt": prompt,
            "mask": mask_meta,
            "candidates": [c.__dict__ for c in candidate_results],
            "error": str(exc),
        }
        with (video_root_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[skip] {video_id}: {exc}")
        if args.cleanup_failed:
            shutil.rmtree(video_root_dir, ignore_errors=True)
        return None

    copytree_clean(Path(neg1.comp_dir), video_root_dir / "neg_frames_1")
    copytree_clean(Path(neg2.comp_dir), video_root_dir / "neg_frames_2")

    chunks = []
    score_windows = [int(x) for x in parse_csv(args.score_windows)]
    for start in range(0, max(1, num_frames - args.train_nframes + 1), args.train_nframes):
        end = min(num_frames, start + max(score_windows + [args.train_nframes]))
        if end - start >= args.train_nframes:
            chunks.append({"start": start, "end": end})

    meta = {
        "video_id": video_id,
        "source": video.source,
        "source_frame_dir": str(video.frame_dir),
        "num_frames": num_frames,
        "frame_size": [args.width, args.height],
        "prompt": prompt,
        "mask": mask_meta,
        "score_windows": score_windows,
        "train_nframes": args.train_nframes,
        "selected_neg_frames_1": neg1.method,
        "selected_neg_frames_2": neg2.method,
        "selection_policy": selection_policy,
        "candidates": [c.__dict__ for c in candidate_results],
        "chunks": chunks,
    }
    with (video_root_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    cleanup_candidate_storage(
        video_root_dir,
        candidate_results,
        [neg1.method, neg2.method],
        args.candidate_retention,
    )

    manifest_entry = {
        "gt_frames": f"{video_id}/gt_frames",
        "masks": f"{video_id}/masks",
        "mask_meta": f"{video_id}/mask_meta.json",
        "neg_frames_1": f"{video_id}/neg_frames_1",
        "neg_frames_2": f"{video_id}/neg_frames_2",
        "num_frames": num_frames,
        "source": video.source,
        "source_frame_dir": str(video.frame_dir),
        "mask_seed": mask_meta.get("seed"),
        "neg_1_source": neg1.method,
        "neg_2_source": neg2.method,
        "neg_1_quality": neg1.quality_score,
        "neg_2_quality": neg2.quality_score,
    }
    print(f"[done] {video_id}: neg1={neg1.method}, neg2={neg2.method}")
    return video_id, manifest_entry


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multimodel DPO negatives for DiffuEraser.")
    parser.add_argument("--project_root", default=str(REPO_ROOT))
    parser.add_argument("--ytbv_root", required=True)
    parser.add_argument("--davis_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--third_party_root", required=True)
    parser.add_argument("--adapter_config", required=True)
    parser.add_argument("--diffueraser_env", default=os.environ.get("DIFFUERASER_ENV", "/home/nvme01/conda_envs/diffueraser"))
    parser.add_argument("--methods", default="propainter,cococo,diffueraser,minimax")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--caption_json", default=None)
    parser.add_argument("--num_videos", type=int, default=0, help="0 means all scanned videos.")
    parser.add_argument("--max_frames", type=int, default=48)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--train_nframes", type=int, default=16)
    parser.add_argument("--score_windows", default="32,24,16")
    parser.add_argument("--mask_seeds_per_video", type=int, default=1)
    parser.add_argument("--mask_dilation_iter", type=int, default=0)
    parser.add_argument("--mask_area_min", type=float, default=0.20)
    parser.add_argument("--mask_area_max", type=float, default=0.30)
    parser.add_argument("--mask_margin_ratio", type=float, default=0.10)
    parser.add_argument("--mask_static_prob", type=float, default=0.50)
    parser.add_argument("--mask_speed_min", type=float, default=0.50)
    parser.add_argument("--mask_speed_max", type=float, default=1.50)
    parser.add_argument("--mask_center_jitter_ratio", type=float, default=0.04)
    parser.add_argument("--mask_motion_box_ratio", type=float, default=0.16)
    parser.add_argument("--source_selection_weights", default="propainter=1.5,cococo=1.0,diffueraser=1.0,minimax=1.0")
    parser.add_argument("--source_quality_max_overrides", default="propainter=0.98")
    parser.add_argument("--neg_quality_min", type=float, default=0.30)
    parser.add_argument("--neg_quality_max", type=float, default=0.65)
    parser.add_argument("--neg_quality_target", type=float, default=0.45)
    parser.add_argument("--parallel_methods", type=int, default=3)
    parser.add_argument("--parallel_videos", type=int, default=1)
    parser.add_argument("--method_gpu_map", default="")
    parser.add_argument("--gpu_slots", default="1", help="Per-GPU concurrent leases. Use '2' or '0=2;1=1'.")
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--enable_lpips", action="store_true")
    parser.add_argument("--enable_vbench", action="store_true")
    parser.add_argument("--vbench_dimensions", default=DEFAULT_VBENCH_DIMS)
    parser.add_argument("--save_previews", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--candidate_retention", choices=["all", "selected", "none"], default="all")
    parser.add_argument("--cleanup_failed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_root = str(Path(args.output_root))
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    adapter_config = load_adapter_config(Path(args.adapter_config))
    captions = load_captions(Path(args.caption_json) if args.caption_json else None)

    min_frames = max(args.train_nframes, min(int(x) for x in parse_csv(args.score_windows)))
    videos = []
    videos.extend(scan_videos(Path(args.davis_root), "davis", min_frames=min_frames))
    videos.extend(scan_videos(Path(args.ytbv_root), "ytbv", min_frames=args.train_nframes))
    videos = sorted(videos, key=lambda v: (v.source, v.name))

    rng = random.Random(args.seed)
    rng.shuffle(videos)
    if args.num_videos > 0:
        videos = videos[:args.num_videos]

    print(f"[scan] videos={len(videos)} davis={sum(v.source == 'davis' for v in videos)} ytbv={sum(v.source == 'ytbv' for v in videos)}")
    print(f"[scan] output_root={args.output_root}")

    manifest_path = Path(args.output_root) / "manifest.json"
    if args.resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    methods = parse_csv(args.methods)
    gpus = parse_csv(args.gpus)
    method_gpu_map = parse_method_gpu_map(args.method_gpu_map, methods, gpus)
    gpu_slots = parse_gpu_slots(args.gpu_slots, gpus)
    print(f"[scheduler] parallel_videos={args.parallel_videos} parallel_methods={args.parallel_methods}")
    print(f"[scheduler] method_gpu_map={json.dumps(method_gpu_map, ensure_ascii=False)}")
    print(f"[scheduler] gpu_slots={json.dumps(gpu_slots, ensure_ascii=False)}")

    source_counts: Dict[str, int] = seed_source_counts_from_existing(Path(args.output_root)) if args.resume else {}
    if source_counts:
        print(f"[resume] seeded source_counts from existing completed items: {json.dumps(source_counts, ensure_ascii=False)}")
    manifest_lock = threading.Lock()
    selection_lock = threading.Lock()
    gpu_dispatcher = GpuDispatcher(method_gpu_map, gpu_slots=gpu_slots)

    tasks: List[Tuple[VideoItem, int]] = []
    for idx, video in enumerate(videos, 1):
        for local_seed_idx in range(args.mask_seeds_per_video):
            mask_seed = args.seed + idx * 1000 + local_seed_idx
            tasks.append((video, mask_seed))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel_videos)) as video_pool:
        futures = [
            video_pool.submit(
                process_one_video,
                video,
                mask_seed,
                args,
                adapter_config,
                captions,
                source_counts,
                gpu_dispatcher,
                selection_lock,
            )
            for video, mask_seed in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            processed = future.result()
            if processed is None:
                continue
            video_id, entry = processed
            with manifest_lock:
                manifest[video_id] = entry
                write_manifest(Path(args.output_root), manifest)

    summary = {
        "output_root": args.output_root,
        "num_manifest_entries": len(manifest),
        "parallel_videos": args.parallel_videos,
        "parallel_methods": args.parallel_methods,
        "method_gpu_map": method_gpu_map,
        "gpu_slots": gpu_slots,
        "source_selection_counts": source_counts,
        "source_selection_weights": parse_source_weights(args.source_selection_weights),
        "neg_selection": {
            "quality_band": [args.neg_quality_min, args.neg_quality_max],
            "quality_target": args.neg_quality_target,
            "source_quality_max_overrides": parse_source_weights(args.source_quality_max_overrides),
            "vbench_dimensions": parse_csv(args.vbench_dimensions),
            "lpips_enabled": bool(args.enable_lpips),
            "vbench_enabled": bool(args.enable_vbench),
            "quality_score_metrics": [
                {"key": key, "higher_better": higher, "weight": weight}
                for key, higher, weight in QUALITY_METRIC_SPECS
            ],
        },
        "mask_policy": {
            "area_ratio_range": [args.mask_area_min, args.mask_area_max],
            "margin_ratio": args.mask_margin_ratio,
            "static_prob": args.mask_static_prob,
            "speed_range_px_per_frame": [args.mask_speed_min, args.mask_speed_max],
            "center_jitter_ratio": args.mask_center_jitter_ratio,
            "motion_box_ratio": args.mask_motion_box_ratio,
            "dilation_iter": args.mask_dilation_iter,
        },
        "methods": methods,
        "candidate_retention": args.candidate_retention,
        "cleanup_failed": bool(args.cleanup_failed),
        "schema": "gt_frames,masks,mask_meta.json,neg_frames_1,neg_frames_2,meta.json",
    }
    with (Path(args.output_root) / "generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
