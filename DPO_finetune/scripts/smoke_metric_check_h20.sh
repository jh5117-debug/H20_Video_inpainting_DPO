#!/usr/bin/env bash
set -Eeuo pipefail

# H20 end-to-end smoke for multimodel DPO generation with LPIPS and VBench on.
# It fails fast if any model candidate is missing metrics or if the dataset
# schema is not directly trainable.

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/smoke_outputs/DPO_Multimodel_Metric_Check_${TIMESTAMP}}"
METHODS="${METHODS:-propainter,cococo,diffueraser,minimax}"
VBENCH_DIMENSIONS="${VBENCH_DIMENSIONS:-subject_consistency,background_consistency,temporal_flickering,motion_smoothness,aesthetic_quality}"

cd "${PROJECT_ROOT}"

echo "[metric-smoke] project=${PROJECT_ROOT}"
echo "[metric-smoke] output=${OUT_ROOT}"
echo "[metric-smoke] methods=${METHODS}"
echo "[metric-smoke] vbench=${VBENCH_DIMENSIONS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
OUT_ROOT="${OUT_ROOT}" \
METHODS="${METHODS}" \
CAPTION_JSON="${CAPTION_JSON:-${PROJECT_ROOT}/DPO_finetune/captions/cococo_qwen_smoke_captions.json}" \
NUM_VIDEOS="${NUM_VIDEOS:-1}" \
MASK_SEEDS_PER_VIDEO="${MASK_SEEDS_PER_VIDEO:-1}" \
MAX_FRAMES="${MAX_FRAMES:-32}" \
HEIGHT="${HEIGHT:-512}" \
WIDTH="${WIDTH:-512}" \
SCORE_WINDOWS="${SCORE_WINDOWS:-32,24,16}" \
TRAIN_NFRAMES="${TRAIN_NFRAMES:-16}" \
MASK_AREA_MIN="${MASK_AREA_MIN:-0.35}" \
MASK_AREA_MAX="${MASK_AREA_MAX:-0.45}" \
MASK_CENTER_JITTER_RATIO="${MASK_CENTER_JITTER_RATIO:-0.04}" \
MASK_MOTION_BOX_RATIO="${MASK_MOTION_BOX_RATIO:-0.16}" \
MASK_STATIC_PROB="${MASK_STATIC_PROB:-0.50}" \
MASK_SPEED_MIN="${MASK_SPEED_MIN:-0.50}" \
MASK_SPEED_MAX="${MASK_SPEED_MAX:-1.50}" \
MASK_DILATION_ITER="${MASK_DILATION_ITER:-8}" \
SOURCE_SELECTION_WEIGHTS="${SOURCE_SELECTION_WEIGHTS:-propainter=1.5,cococo=1.0,diffueraser=1.0,minimax=1.0}" \
NEG_QUALITY_MIN="${NEG_QUALITY_MIN:-0.20}" \
NEG_QUALITY_MAX="${NEG_QUALITY_MAX:-0.80}" \
NEG_QUALITY_TARGET="${NEG_QUALITY_TARGET:-0.40}" \
VBENCH_DIMENSIONS="${VBENCH_DIMENSIONS}" \
PARALLEL_METHODS="${PARALLEL_METHODS:-4}" \
ENABLE_LPIPS=1 \
ENABLE_VBENCH=1 \
SAVE_PREVIEWS="${SAVE_PREVIEWS:-1}" \
bash DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh

export OUT_ROOT METHODS VBENCH_DIMENSIONS
python - <<'PY'
import json
import math
import os
import sys
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
methods = [m.strip() for m in os.environ["METHODS"].split(",") if m.strip()]
vbench_dims = [d.strip() for d in os.environ["VBENCH_DIMENSIONS"].split(",") if d.strip()]

errors = []

def fail(msg):
    errors.append(msg)

def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        fail(f"cannot read {path}: {exc}")
        return None

manifest = load_json(root / "manifest.json")
summary = load_json(root / "generation_summary.json")
if not isinstance(manifest, dict) or not manifest:
    fail(f"manifest is empty or missing: {root / 'manifest.json'}")
if not isinstance(summary, dict):
    fail(f"summary is missing: {root / 'generation_summary.json'}")

required_score_keys = [
    "psnr",
    "ssim",
    "lpips",
    "vbench_inpainting_score",
    "relative_quality_score",
    "metric_norms",
    "pixel_quality_norm",
    "temporal_quality_norm",
    "perceptual_quality_norm",
    "defect_bucket",
]

if isinstance(manifest, dict):
    for video_id, entry in manifest.items():
        video_root = root / video_id
        meta = load_json(video_root / "meta.json")
        mask_meta = load_json(video_root / "mask_meta.json")
        if not isinstance(meta, dict):
            continue
        if not isinstance(mask_meta, dict):
            continue

        for subdir in ["gt_frames", "masks", "neg_frames_1", "neg_frames_2"]:
            files = sorted((video_root / subdir).glob("*.png"))
            if not files:
                fail(f"{video_id}: missing frames in {subdir}")

        area = float(mask_meta.get("area_ratio", -1))
        if not (0.30 <= area <= 0.50):
            fail(f"{video_id}: mask area looks wrong: {area}")
        centers = [frame.get("bbox_center_ratio") for frame in mask_meta.get("frames", [])]
        centers = [c for c in centers if isinstance(c, list) and len(c) == 2]
        if not centers:
            fail(f"{video_id}: mask_meta has no frame centers")
        else:
            xs = [float(c[0]) for c in centers]
            ys = [float(c[1]) for c in centers]
            if min(xs) < 0.30 or max(xs) > 0.70 or min(ys) < 0.30 or max(ys) > 0.70:
                fail(f"{video_id}: mask center left central band: x={min(xs):.3f}-{max(xs):.3f} y={min(ys):.3f}-{max(ys):.3f}")

        candidates = meta.get("candidates", [])
        ok_by_method = {c.get("method"): c for c in candidates if c.get("ok") is True}
        missing_methods = sorted(set(methods) - set(ok_by_method))
        if missing_methods:
            fail(f"{video_id}: missing successful methods: {missing_methods}")

        for method in methods:
            cand = ok_by_method.get(method)
            if not cand:
                continue
            score = cand.get("score", {})
            for key in required_score_keys:
                if key not in score:
                    fail(f"{video_id}/{method}: missing score key {key}")
            for key in ["psnr", "ssim", "lpips", "vbench_inpainting_score", "relative_quality_score"]:
                val = score.get(key)
                if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                    fail(f"{video_id}/{method}: invalid numeric score {key}={val}")
            vb = score.get("vbench", {})
            if vb.get("error"):
                fail(f"{video_id}/{method}: vbench error: {vb.get('error')}")
            per_dim = vb.get("per_dim", {})
            for dim in vbench_dims:
                flat_key = f"vbench_{dim}"
                raw = per_dim.get(dim)
                if flat_key not in score:
                    fail(f"{video_id}/{method}: missing flattened {flat_key}")
                if not isinstance(raw, (int, float)) or float(raw) < 0:
                    fail(f"{video_id}/{method}: invalid VBench dim {dim}={raw}")

        policy = meta.get("selection_policy", {})
        for key in ["candidate_quality_order", "eligible_candidate_order", "quality_band", "quality_target"]:
            if key not in policy:
                fail(f"{video_id}: selection_policy missing {key}")

if errors:
    print("[metric-smoke][FAIL]")
    for err in errors:
        print(f"  - {err}")
    print(f"[metric-smoke] output={root}")
    sys.exit(1)

print("[metric-smoke][OK]")
print(f"[metric-smoke] output={root}")
if isinstance(manifest, dict):
    print(f"[metric-smoke] manifest_entries={len(manifest)}")
    for video_id in manifest:
        meta = load_json(root / video_id / "meta.json") or {}
        order = meta.get("selection_policy", {}).get("candidate_quality_order", [])
        print(f"[metric-smoke] {video_id}")
        for item in order:
            score = item.get("score", {})
            print(
                "  "
                f"{item.get('method')}: quality={item.get('quality'):.4f} "
                f"bucket={score.get('defect_bucket')} "
                f"psnr={score.get('psnr'):.3f} "
                f"ssim={score.get('ssim'):.4f} "
                f"lpips={score.get('lpips'):.4f} "
                f"vbench={score.get('vbench_inpainting_score'):.4f}"
            )
PY

echo "[metric-smoke] checking logs"
if grep -R "Traceback\|RuntimeError\|ModuleNotFoundError\|ImportError\|FileNotFoundError\|CUDA out of memory\|ValueError\|No such file\|does not exist" \
  "${OUT_ROOT}"/*/candidates/*/inference.log; then
  echo "[metric-smoke][FAIL] error pattern found in inference logs"
  exit 1
fi

echo "[metric-smoke][DONE] ${OUT_ROOT}"
