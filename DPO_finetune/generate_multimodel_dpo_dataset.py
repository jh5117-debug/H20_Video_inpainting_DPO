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
import copy
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
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
    angles = sorted(rng.random() * 2 * math.pi for _ in range(n))
    pts = []
    for a in angles:
        r = rng.uniform(0.6, 0.8)
        x = math.cos(a) * width * 0.5 * r
        y = math.sin(a) * height * 0.5 * r
        pts.append((x, y))
    return np.array(pts, dtype=np.float32)


def generate_hard_masks(
    out_dir: Path,
    num_frames: int,
    width: int,
    height: int,
    seed: int,
    dilation_iter: int,
    margin_ratio: float = 0.15,
) -> Dict[str, Any]:
    """Generate and save large central slow/static polygon masks.

    The requested 75%-85% polygon size can be larger than the 70% central
    safe zone implied by a 15% border. We therefore keep the center inside the
    safe zone and clamp only image-boundary crossings, preserving the intended
    "large, central, hard" behavior.
    """
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    poly_w = rng.uniform(0.75, 0.85) * width
    poly_h = rng.uniform(0.75, 0.85) * height
    base = random_polygon(int(poly_w), int(poly_h), rng)

    margin_x = margin_ratio * width
    margin_y = margin_ratio * height
    cx = rng.uniform(margin_x, width - margin_x)
    cy = rng.uniform(margin_y, height - margin_y)

    if rng.random() < 0.5:
        vx = vy = 0.0
        motion = "static"
    else:
        speed = rng.uniform(0.5, 1.5)
        theta = rng.uniform(0, 2 * math.pi)
        vx = math.cos(theta) * speed
        vy = math.sin(theta) * speed
        motion = "slow_drift"

    kernel = np.ones((3, 3), dtype=np.uint8)
    frame_meta = []
    for t in range(num_frames):
        pts = base + np.array([cx, cy], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

        pil = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(pil)
        draw.polygon([tuple(p) for p in pts.tolist()], fill=255)
        mask = np.array(pil, dtype=np.uint8)
        if dilation_iter > 0:
            mask = cv2.dilate(mask, kernel, iterations=dilation_iter)
        save_gray(out_dir / f"{t:05d}.png", mask)

        frame_meta.append({"frame": t, "center": [round(cx, 3), round(cy, 3)]})

        nx = cx + vx
        ny = cy + vy
        if nx < margin_x or nx > width - margin_x:
            vx = -vx
            nx = cx + vx
        if ny < margin_y or ny > height - margin_y:
            vy = -vy
            ny = cy + vy
        cx, cy = nx, ny

    return {
        "seed": seed,
        "motion": motion,
        "polygon_size_ratio": [poly_w / width, poly_h / height],
        "margin_ratio": margin_ratio,
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
        shell_cmd = f"conda run -p {env_path} {cmd}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

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


def maybe_vbench_score(frame_dir: Path, name: str, device: str, work_dir: Path) -> Dict[str, Any]:
    try:
        from tools.score_inpainting_quality import InpaintingScorer
        video_path = encode_mp4(frame_dir, work_dir / f"{name}.mp4")
        scorer = InpaintingScorer(device=device)
        return scorer.score_video(str(video_path), name=name)
    except Exception as exc:
        return {"error": str(exc), "inpainting_score": None}


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
        lpips_device="cuda" if args.enable_lpips else "cpu",
    )
    if args.enable_vbench:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        vb = maybe_vbench_score(comp_dir, method, "cuda", comp_dir.parent / "_videos")
        score["vbench"] = vb
        if vb.get("inpainting_score") is not None:
            score["vbench_inpainting_score"] = float(vb["inpainting_score"])
    return score


def assign_relative_quality(candidates: List[CandidateResult]) -> None:
    ok = [c for c in candidates if c.ok]
    if not ok:
        return

    metric_specs = [
        ("psnr", True, 0.20),
        ("ssim", True, 0.20),
        ("lpips", False, 0.25),
        ("vbench_inpainting_score", True, 0.35),
    ]
    available = []
    for key, higher, weight in metric_specs:
        vals = [c.score.get(key) for c in ok]
        vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if vals:
            available.append((key, higher, weight))

    if not available:
        for c in ok:
            c.quality_score = 0.5
        return

    total_w = sum(w for _, _, w in available)
    for c in ok:
        q = 0.0
        for key, higher, weight in available:
            values = [float(x.score[key]) for x in ok if key in x.score and isinstance(x.score[key], (int, float))]
            lo, hi = min(values), max(values)
            val = float(c.score.get(key, lo if higher else hi))
            if hi == lo:
                norm = 0.5
            else:
                norm = (val - lo) / (hi - lo)
            if not higher:
                norm = 1.0 - norm
            q += (weight / total_w) * norm
        c.quality_score = float(q)
        c.score["relative_quality_score"] = c.quality_score


def select_negatives(candidates: List[CandidateResult], source_counts: Dict[str, int]) -> Tuple[CandidateResult, CandidateResult, Dict[str, Any]]:
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
            "candidate_quality_order": [
                {"method": only.method, "quality": only.quality_score, "score": only.score}
            ],
            "trimmed_extremes": False,
        }

    if len(ok_sorted) >= 5:
        eligible = ok_sorted[1:-1]
    else:
        eligible = ok_sorted

    # Prefer under-represented source models first, then hard-but-not-catastrophic candidates.
    ranked = sorted(
        eligible,
        key=lambda c: (
            source_counts.get(c.method, 0),
            abs(c.quality_score - 0.35),
            c.quality_score,
        ),
    )
    first = ranked[0]
    second = None
    for cand in ranked[1:]:
        if cand.method != first.method:
            second = cand
            break
    if second is None:
        second = ranked[1]

    source_counts[first.method] = source_counts.get(first.method, 0) + 1
    source_counts[second.method] = source_counts.get(second.method, 0) + 1

    policy = {
        "policy": "balanced_source_hard_plausible",
        "note": "GT is always the DPO winner; selected negatives are complete outputs from source models.",
        "source_counts_after_selection": dict(source_counts),
        "candidate_quality_order": [
            {"method": c.method, "quality": c.quality_score, "score": c.score}
            for c in ok_sorted
        ],
        "trimmed_extremes": len(ok_sorted) >= 5,
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


def process_one_video(
    video: VideoItem,
    mask_seed: int,
    args: argparse.Namespace,
    adapter_config: Dict[str, Dict[str, Any]],
    captions: Dict[str, str],
    source_counts: Dict[str, int],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    video_id = f"{video.name}_mask{mask_seed}"
    video_root_dir = Path(args.output_root) / video_id
    if args.resume and (video_root_dir / "meta.json").exists() and (video_root_dir / "neg_frames_1").is_dir() and (video_root_dir / "neg_frames_2").is_dir():
        print(f"[skip] {video_id} already complete")
        with (video_root_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return video_id, {
            "gt_frames": f"{video_id}/gt_frames",
            "masks": f"{video_id}/masks",
            "neg_frames_1": f"{video_id}/neg_frames_1",
            "neg_frames_2": f"{video_id}/neg_frames_2",
            "num_frames": int(meta.get("num_frames", 0)),
            "source": video.source,
        }

    if video_root_dir.exists() and not args.resume:
        shutil.rmtree(video_root_dir)
    video_root_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = video_root_dir / "gt_frames"
    mask_dir = video_root_dir / "masks"
    gt_files = prepare_gt_frames(video, gt_dir, args.width, args.height, args.max_frames)
    num_frames = len(gt_files)
    if num_frames < args.train_nframes:
        print(f"[skip] {video_id}: too few frames after sampling")
        return None

    mask_meta = generate_hard_masks(
        mask_dir,
        num_frames=num_frames,
        width=args.width,
        height=args.height,
        seed=mask_seed,
        dilation_iter=args.mask_dilation_iter,
    )

    # Some third-party CLIs expect a root containing one sequence directory.
    batch_video_root = video_root_dir / "_adapter_inputs" / "videos"
    batch_mask_root = video_root_dir / "_adapter_inputs" / "masks"
    batch_video_dir = batch_video_root / video_id
    batch_mask_dir = batch_mask_root / video_id
    copytree_clean(gt_dir, batch_video_dir)
    copytree_clean(mask_dir, batch_mask_dir)

    prompt = prompt_for(video, video_id, captions)
    methods = parse_csv(args.methods)
    gpus = parse_csv(args.gpus)
    candidates_root = video_root_dir / "candidates"
    candidate_results: List[CandidateResult] = []

    def _job(idx_method: Tuple[int, str]) -> CandidateResult:
        idx, method = idx_method
        cfg = copy.deepcopy(adapter_config.get(method, {}))
        gpu = gpus[idx % len(gpus)]
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
        neg1, neg2, selection_policy = select_negatives(candidate_results, source_counts)
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

    manifest_entry = {
        "gt_frames": f"{video_id}/gt_frames",
        "masks": f"{video_id}/masks",
        "neg_frames_1": f"{video_id}/neg_frames_1",
        "neg_frames_2": f"{video_id}/neg_frames_2",
        "num_frames": num_frames,
        "source": video.source,
        "neg_1_source": neg1.method,
        "neg_2_source": neg2.method,
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
    parser.add_argument("--methods", default="propainter,cococo,minimax")
    parser.add_argument("--gpus", default="1,2,3")
    parser.add_argument("--caption_json", default=None)
    parser.add_argument("--num_videos", type=int, default=0, help="0 means all scanned videos.")
    parser.add_argument("--max_frames", type=int, default=48)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--train_nframes", type=int, default=16)
    parser.add_argument("--score_windows", default="32,24,16")
    parser.add_argument("--mask_seeds_per_video", type=int, default=1)
    parser.add_argument("--mask_dilation_iter", type=int, default=8)
    parser.add_argument("--parallel_methods", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--enable_lpips", action="store_true")
    parser.add_argument("--enable_vbench", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
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

    source_counts: Dict[str, int] = {}
    for idx, video in enumerate(videos, 1):
        for local_seed_idx in range(args.mask_seeds_per_video):
            mask_seed = args.seed + idx * 1000 + local_seed_idx
            processed = process_one_video(
                video=video,
                mask_seed=mask_seed,
                args=args,
                adapter_config=adapter_config,
                captions=captions,
                source_counts=source_counts,
            )
            if processed is None:
                continue
            video_id, entry = processed
            manifest[video_id] = entry
            write_manifest(Path(args.output_root), manifest)

    summary = {
        "output_root": args.output_root,
        "num_manifest_entries": len(manifest),
        "source_selection_counts": source_counts,
        "methods": parse_csv(args.methods),
        "schema": "gt_frames,masks,neg_frames_1,neg_frames_2,meta.json",
    }
    with (Path(args.output_root) / "generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
