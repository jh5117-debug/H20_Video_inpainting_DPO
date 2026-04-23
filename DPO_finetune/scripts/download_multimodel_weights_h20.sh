#!/usr/bin/env bash
set -Eeuo pipefail

# Download and normalize third-party weights used by multimodel DPO generation.
# The layout after this script is:
#   third_party_video_inpainting/weights/
#     COCOCO_weight/
#       cococo/model_0.pth ... model_3.pth
#       stable-diffusion-v1-5-inpainting/model_index.json ...
#     minimax/{vae,transformer,scheduler}
#     propainter/{ProPainter.pth,raft-things.pth,recurrent_flow_completion.pth}

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${PROJECT_ROOT}/third_party_video_inpainting}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${THIRD_PARTY_ROOT}/weights}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${THIRD_PARTY_ROOT}/downloads}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-/home/nvme01/conda_envs/diffueraser}"

COCOCO_HF_REPO="${COCOCO_HF_REPO:-JiaHuang01/COCOCO}"
COCOCO_HF_REPO_TYPE="${COCOCO_HF_REPO_TYPE:-dataset}"
COCOCO_HF_FILENAME="${COCOCO_HF_FILENAME:-OneDrive_1_2026-4-23.zip}"
SD_INPAINT_REPO="${SD_INPAINT_REPO:-runwayml/stable-diffusion-inpainting}"
SD_INPAINT_REPOS="${SD_INPAINT_REPOS:-${SD_INPAINT_REPO},stable-diffusion-v1-5/stable-diffusion-inpainting,genai-archive/stable-diffusion-v1-5-inpainting,benjamin-paine/stable-diffusion-v1-5-inpainting}"
SD_INPAINT_VARIANT="${SD_INPAINT_VARIANT:-fp16}"
SD_INPAINT_SEARCH_DIRS="${SD_INPAINT_SEARCH_DIRS:-}"
MINIMAX_HF_REPO="${MINIMAX_HF_REPO:-zibojia/minimax-remover}"
COCOCO_LOCAL_ZIP="${COCOCO_LOCAL_ZIP:-}"
SD_INPAINT_LOCAL_DIR="${SD_INPAINT_LOCAL_DIR:-}"
MINIMAX_LOCAL_DIR="${MINIMAX_LOCAL_DIR:-}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
HF_SNAPSHOT_MAX_WORKERS="${HF_SNAPSHOT_MAX_WORKERS:-1}"
USE_H20_PROXY="${USE_H20_PROXY:-0}"
START_H20_CLASH="${START_H20_CLASH:-0}"
CLASH_ROOT="${CLASH_ROOT:-/home/nvme01/clash-for-linux}"

if [[ "${START_H20_CLASH}" == "1" && -x "${CLASH_ROOT}/start.sh" ]]; then
  echo "[net] start clash: ${CLASH_ROOT}/start.sh"
  bash "${CLASH_ROOT}/start.sh" || true
fi

if [[ "${USE_H20_PROXY}" == "1" ]]; then
  if [[ -f "${CLASH_ROOT}/clash.sh" ]]; then
    echo "[net] enable H20 proxy from ${CLASH_ROOT}/clash.sh"
    set +u
    # shellcheck disable=SC1090
    source "${CLASH_ROOT}/clash.sh"
    proxy_on || true
    set -u
  else
    echo "[net][warn] clash.sh not found: ${CLASH_ROOT}/clash.sh"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "/home/nvme01/miniconda3/bin/conda" && -d "${DIFFUERASER_ENV}" ]]; then
    PYTHON_RUN=(/home/nvme01/miniconda3/bin/conda run --no-capture-output -p "${DIFFUERASER_ENV}" python)
  else
    PYTHON_RUN=(python)
  fi
else
  PYTHON_RUN=("${PYTHON_BIN}")
fi

mkdir -p "${WEIGHTS_ROOT}" "${DOWNLOAD_ROOT}"
export PROJECT_ROOT THIRD_PARTY_ROOT WEIGHTS_ROOT DOWNLOAD_ROOT
export COCOCO_HF_REPO COCOCO_HF_REPO_TYPE COCOCO_HF_FILENAME SD_INPAINT_REPO SD_INPAINT_REPOS SD_INPAINT_VARIANT MINIMAX_HF_REPO
export COCOCO_LOCAL_ZIP SD_INPAINT_LOCAL_DIR SD_INPAINT_SEARCH_DIRS MINIMAX_LOCAL_DIR HF_LOCAL_FILES_ONLY
export HF_ENDPOINT HF_HUB_DISABLE_XET HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT HF_SNAPSHOT_MAX_WORKERS

echo "[weights] project=${PROJECT_ROOT}"
echo "[weights] root=${WEIGHTS_ROOT}"
echo "[weights] python=${PYTHON_RUN[*]}"
echo "[weights] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[weights] HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "[weights] HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "[weights] HF_SNAPSHOT_MAX_WORKERS=${HF_SNAPSHOT_MAX_WORKERS}"
echo "[weights] SD_INPAINT_VARIANT=${SD_INPAINT_VARIANT}"
if [[ -n "${SD_INPAINT_SEARCH_DIRS}" ]]; then
  echo "[weights] SD_INPAINT_SEARCH_DIRS=${SD_INPAINT_SEARCH_DIRS}"
fi

PY_SCRIPT="$(mktemp "${DOWNLOAD_ROOT}/download_multimodel_weights.XXXXXX.py")"
cleanup() {
  rm -f "${PY_SCRIPT}"
}
trap cleanup EXIT

cat > "${PY_SCRIPT}" <<'PY'
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def ensure_hf_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, snapshot_download
    return hf_hub_download, snapshot_download


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    try:
        os.symlink(src, dst)
        return
    except Exception:
        pass
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def find_cococo_ckpts(root: Path):
    result = {}
    for p in root.rglob("model_*.pth"):
        if p.name in {f"model_{i}.pth" for i in range(4)}:
            result[p.name] = p
    return result


def find_sd_root(root: Path):
    candidates = []
    for p in root.rglob("model_index.json"):
        parent = p.parent
        names = {x.name for x in parent.iterdir()}
        score = sum(x in names for x in ["unet", "vae", "text_encoder", "tokenizer", "scheduler"])
        candidates.append((score, parent))
    candidates.sort(reverse=True, key=lambda x: x[0])
    if candidates and candidates[0][0] >= 3:
        return candidates[0][1]
    return None


def sd_patterns(variant: str):
    if variant == "fp16":
        return (
            [
                "model_index.json",
                "scheduler/**",
                "tokenizer/**",
                "text_encoder/config.json",
                "text_encoder/*fp16.bin",
                "text_encoder/*fp16.safetensors",
                "vae/config.json",
                "vae/*fp16.bin",
                "vae/*fp16.safetensors",
                "unet/config.json",
                "unet/*fp16.bin",
                "unet/*fp16.safetensors",
            ],
            ["*.onnx", "*.msgpack"],
        )
    return (
        [
            "model_index.json",
            "scheduler/**",
            "tokenizer/**",
            "text_encoder/**",
            "vae/**",
            "unet/**",
        ],
        ["*.fp16.*", "*.onnx", "*.msgpack"],
    )


def normalize_sd_fp16_names(root: Path) -> None:
    for fp16_path in root.rglob("*.fp16.bin"):
        base_path = fp16_path.with_name(fp16_path.name.replace(".fp16.bin", ".bin"))
        if base_path != fp16_path:
            link_or_copy(fp16_path, base_path)
    for fp16_path in root.rglob("*.fp16.safetensors"):
        base_path = fp16_path.with_name(fp16_path.name.replace(".fp16.safetensors", ".safetensors"))
        if base_path != fp16_path:
            link_or_copy(fp16_path, base_path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sd_health_errors(root: Path) -> list:
    errors = []
    required = [
        root / "model_index.json",
        root / "scheduler" / "scheduler_config.json",
        root / "tokenizer",
        root / "text_encoder" / "config.json",
        root / "vae" / "config.json",
        root / "unet" / "config.json",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        errors.append(f"missing required files: {missing}")
        return errors

    try:
        vae_config = read_json(root / "vae" / "config.json")
        vae_down_blocks = vae_config.get("down_block_types", [])
        vae_up_blocks = vae_config.get("up_block_types", [])
        if any("CrossAttn" in str(x) for x in vae_down_blocks + vae_up_blocks):
            errors.append("vae/config.json looks like a conditional UNet config, not an AutoencoderKL config")
        if not any("DownEncoderBlock2D" in str(x) for x in vae_down_blocks):
            errors.append(f"vae/config.json has suspicious down_block_types={vae_down_blocks}")
    except Exception as exc:
        errors.append(f"failed to read vae/config.json: {exc}")

    try:
        unet_config = read_json(root / "unet" / "config.json")
        unet_down_blocks = unet_config.get("down_block_types", [])
        if not any("CrossAttnDownBlock2D" in str(x) for x in unet_down_blocks):
            errors.append(f"unet/config.json has suspicious down_block_types={unet_down_blocks}")
        if "cross_attention_dim" not in unet_config:
            errors.append("unet/config.json missing cross_attention_dim")
        if int(unet_config.get("in_channels", -1)) != 9:
            errors.append(f"unet/config.json in_channels={unet_config.get('in_channels')} but inpainting UNet should be 9")
    except Exception as exc:
        errors.append(f"failed to read unet/config.json: {exc}")

    weight_options = [
        [
            root / "text_encoder" / "pytorch_model.bin",
            root / "text_encoder" / "model.safetensors",
            root / "text_encoder" / "pytorch_model.fp16.bin",
            root / "text_encoder" / "model.fp16.safetensors",
        ],
        [
            root / "vae" / "diffusion_pytorch_model.bin",
            root / "vae" / "diffusion_pytorch_model.safetensors",
            root / "vae" / "diffusion_pytorch_model.fp16.bin",
            root / "vae" / "diffusion_pytorch_model.fp16.safetensors",
        ],
        [
            root / "unet" / "diffusion_pytorch_model.bin",
            root / "unet" / "diffusion_pytorch_model.safetensors",
            root / "unet" / "diffusion_pytorch_model.fp16.bin",
            root / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        ],
    ]
    for options in weight_options:
        if not any(p.exists() for p in options):
            errors.append(f"missing weight file; expected one of {[str(p.relative_to(root)) for p in options]}")
    return errors


def sd_ready(root: Path) -> bool:
    return not sd_health_errors(root)


def safe_repo_dir_name(repo_id: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in repo_id)


def publish_sd_root(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name == ".cache":
            continue
        link_or_copy(child, dst / child.name)
    normalize_sd_fp16_names(dst)


def hf_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def candidate_sd_roots(base: Path, repo_ids: list):
    if not base.exists():
        return
    if (base / "model_index.json").exists():
        yield base
    if (base / "hub").exists():
        yield from candidate_sd_roots(base / "hub", repo_ids)
    for child in base.glob("sd_inpaint_*"):
        if child.is_dir():
            if (child / "model_index.json").exists():
                yield child
            sd_child = find_sd_root(child)
            if sd_child is not None:
                yield sd_child
    for repo_id in repo_ids:
        snapshots = base / hf_cache_name(repo_id) / "snapshots"
        if snapshots.exists():
            for model_index in snapshots.glob("*/model_index.json"):
                yield model_index.parent


def find_valid_local_sd(repo_ids: list, search_dirs: list) -> Path | None:
    seen = set()
    invalid_count = 0
    for base in search_dirs:
        for candidate in candidate_sd_roots(base, repo_ids):
            try:
                key = candidate.resolve()
            except Exception:
                key = candidate
            if key in seen:
                continue
            seen.add(key)
            errors = sd_health_errors(candidate)
            if not errors:
                print(f"[sd] found valid local SD inpainting candidate: {candidate}")
                return candidate
            if invalid_count < 6:
                print(f"[sd][warn] local SD candidate invalid: {candidate}; errors={errors}")
                invalid_count += 1
    return None


weights_root = Path(os.environ["WEIGHTS_ROOT"])
download_root = Path(os.environ["DOWNLOAD_ROOT"])
cococo_repo = os.environ["COCOCO_HF_REPO"]
cococo_repo_type = os.environ["COCOCO_HF_REPO_TYPE"]
cococo_filename = os.environ["COCOCO_HF_FILENAME"]
sd_repo = os.environ["SD_INPAINT_REPO"]
sd_repos = [x.strip() for x in os.environ.get("SD_INPAINT_REPOS", sd_repo).split(",") if x.strip()]
sd_inpaint_variant = os.environ.get("SD_INPAINT_VARIANT", "fp16").strip().lower()
minimax_repo = os.environ["MINIMAX_HF_REPO"]
cococo_local_zip = os.environ.get("COCOCO_LOCAL_ZIP", "").strip()
sd_inpaint_local_dir = os.environ.get("SD_INPAINT_LOCAL_DIR", "").strip()
sd_inpaint_search_dirs_raw = os.environ.get("SD_INPAINT_SEARCH_DIRS", "").strip()
minimax_local_dir = os.environ.get("MINIMAX_LOCAL_DIR", "").strip()
local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
hf_snapshot_max_workers = int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", "1"))

hf_hub_download, snapshot_download = ensure_hf_hub()

cococo_bundle = weights_root / "COCOCO_weight"
cococo_ckpt_dir = cococo_bundle / "cococo"
sd_target = cococo_bundle / "stable-diffusion-v1-5-inpainting"
cococo_extract = download_root / "cococo_hf_extract"
cococo_zip = download_root / cococo_filename
sd_search_dirs = [download_root]
for env_name in ["HF_HUB_CACHE", "HF_HOME"]:
    value = os.environ.get(env_name, "").strip()
    if value:
        sd_search_dirs.append(Path(value).expanduser())
sd_search_dirs.extend([
    Path.home() / ".cache" / "huggingface" / "hub",
    Path("/home/ubuntu/.cache/huggingface/hub"),
    Path("/home/nvme03/ubuntu_home_redirect/.cache/huggingface/hub"),
])
for raw_path in sd_inpaint_search_dirs_raw.split(os.pathsep):
    if raw_path.strip():
        sd_search_dirs.append(Path(raw_path.strip()).expanduser())

if cococo_local_zip:
    cococo_zip = Path(cococo_local_zip).expanduser().resolve()
    if not cococo_zip.exists():
        raise SystemExit(f"COCOCO_LOCAL_ZIP does not exist: {cococo_zip}")
    print(f"[cococo] use local zip: {cococo_zip}")
elif cococo_zip.exists():
    print(f"[cococo] reuse local zip: {cococo_zip}")
else:
    print(f"[cococo] download {cococo_repo}/{cococo_filename}")
    zip_path = Path(
        hf_hub_download(
            repo_id=cococo_repo,
            repo_type=cococo_repo_type,
            filename=cococo_filename,
            local_dir=str(download_root),
            local_dir_use_symlinks=False,
            local_files_only=local_files_only,
        )
    )
    if zip_path != cococo_zip and zip_path.exists():
        cococo_zip = zip_path

marker = cococo_extract / ".extracted.ok"
if not marker.exists():
    if cococo_extract.exists():
        shutil.rmtree(cococo_extract)
    cococo_extract.mkdir(parents=True, exist_ok=True)
    print(f"[cococo] extract {cococo_zip} -> {cococo_extract}")
    with zipfile.ZipFile(cococo_zip) as zf:
        zf.extractall(cococo_extract)
    marker.write_text("ok\n", encoding="utf-8")
else:
    print(f"[cococo] reuse extracted files: {cococo_extract}")

ckpts = find_cococo_ckpts(cococo_extract)
missing = [f"model_{i}.pth" for i in range(4) if f"model_{i}.pth" not in ckpts]
if missing:
    raise SystemExit(f"missing COCOCO checkpoints in HF zip: {missing}")
for name, src in sorted(ckpts.items()):
    link_or_copy(src, cococo_ckpt_dir / name)
print(f"[cococo] checkpoints ready: {cococo_ckpt_dir}")

sd_src = find_sd_root(cococo_extract)
if sd_src is not None:
    print(f"[sd] found SD in zip: {sd_src}")
    publish_sd_root(sd_src, sd_target)
elif sd_inpaint_local_dir:
    sd_src = Path(sd_inpaint_local_dir).expanduser().resolve()
    if not (sd_src / "model_index.json").exists():
        raise SystemExit(f"SD_INPAINT_LOCAL_DIR is not a diffusers model folder: {sd_src}")
    print(f"[sd] use local SD inpainting dir: {sd_src}")
    publish_sd_root(sd_src, sd_target)
elif sd_ready(sd_target):
    print(f"[sd] reuse existing valid SD inpainting weights: {sd_target}")
else:
    if sd_target.exists():
        print(f"[sd][warn] existing SD target is invalid and will be rebuilt: {sd_target}")
        for err in sd_health_errors(sd_target):
            print(f"[sd][warn]   {err}")
    local_sd = find_valid_local_sd(sd_repos, sd_search_dirs)
    if local_sd is not None:
        publish_sd_root(local_sd, sd_target)
    else:
        print("[sd] SD inpainting folder not found in zip/cache; try Hugging Face repos")
        last_error = None
        allow_patterns, ignore_patterns = sd_patterns(sd_inpaint_variant)
        for repo_id in sd_repos:
            try:
                repo_work = download_root / f"sd_inpaint_{safe_repo_dir_name(repo_id)}_{sd_inpaint_variant}"
                print(f"[sd] download {repo_id} ({sd_inpaint_variant}) -> {repo_work}")
                if repo_work.exists() and sd_health_errors(repo_work):
                    print(f"[sd][warn] keep partial SD download for resume: {repo_work}")
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    local_dir=str(repo_work),
                    local_dir_use_symlinks=False,
                    local_files_only=local_files_only,
                    max_workers=hf_snapshot_max_workers,
                )
                normalize_sd_fp16_names(repo_work)
                errors = sd_health_errors(repo_work)
                if errors:
                    raise RuntimeError(f"downloaded SD repo is not compatible: {errors}")
                publish_sd_root(repo_work, sd_target)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"[sd][warn] failed {repo_id}: {exc}")
        if last_error is not None:
            local_sd = find_valid_local_sd(sd_repos, sd_search_dirs)
            if local_sd is not None:
                publish_sd_root(local_sd, sd_target)
            else:
                raise last_error
sd_errors = sd_health_errors(sd_target)
if sd_errors:
    raise SystemExit(f"SD inpainting model not ready: {sd_target}; errors={sd_errors}")
print(f"[sd] ready: {sd_target}")

minimax_target = weights_root / "minimax"
if minimax_local_dir:
    minimax_src = Path(minimax_local_dir).expanduser().resolve()
    print(f"[minimax] use local MiniMax dir: {minimax_src}")
    for child in ["vae", "transformer", "scheduler"]:
        if not (minimax_src / child).exists():
            raise SystemExit(f"MINIMAX_LOCAL_DIR missing {child}/: {minimax_src}")
        link_or_copy(minimax_src / child, minimax_target / child)
elif all((minimax_target / child).exists() for child in ["vae", "transformer", "scheduler"]):
    print(f"[minimax] reuse existing weights: {minimax_target}")
else:
    print(f"[minimax] download {minimax_repo}")
    snapshot_download(
        repo_id=minimax_repo,
        allow_patterns=["vae/**", "transformer/**", "scheduler/**"],
        local_dir=str(minimax_target),
        local_dir_use_symlinks=False,
        local_files_only=local_files_only,
        max_workers=hf_snapshot_max_workers,
    )
for child in ["vae", "transformer", "scheduler"]:
    if not (minimax_target / child).exists():
        raise SystemExit(f"MiniMax folder missing after download: {minimax_target / child}")
print(f"[minimax] ready: {minimax_target}")

project_propainter = Path(os.environ["PROJECT_ROOT"]) / "weights" / "propainter"
propainter_target = weights_root / "propainter"
propainter_target.mkdir(parents=True, exist_ok=True)
if project_propainter.exists():
    for name in ["ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth"]:
        src = project_propainter / name
        if src.exists():
            link_or_copy(src, propainter_target / name)
missing_propainter = [name for name in ["ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth"] if not (propainter_target / name).exists()]
if missing_propainter:
    print(f"[propainter][warn] missing {missing_propainter}; ProPainter may auto-download during inference.")
else:
    print(f"[propainter] ready: {propainter_target}")

manifest = {
    "cococo": str(cococo_ckpt_dir),
    "stable_diffusion_inpainting": str(sd_target),
    "minimax": str(minimax_target),
    "propainter": str(propainter_target),
}
(weights_root / "multimodel_weights_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"[done] manifest: {weights_root / 'multimodel_weights_manifest.json'}")
PY

"${PYTHON_RUN[@]}" "${PY_SCRIPT}"

cat <<EOF

[done] Weight folders:
  ${WEIGHTS_ROOT}/COCOCO_weight/cococo
  ${WEIGHTS_ROOT}/COCOCO_weight/stable-diffusion-v1-5-inpainting
  ${WEIGHTS_ROOT}/minimax
  ${WEIGHTS_ROOT}/propainter
EOF
