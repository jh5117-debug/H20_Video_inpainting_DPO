#!/usr/bin/env bash
set -Eeuo pipefail

# Restart multimodel DPO data generation on GPUs 4-7 only, then run a
# 1-hour runtime monitor and print a compact health summary.

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/DPO_Finetune_Data_Multimodel_v1}"
RUNTIME_SEC="${RUNTIME_SEC:-3600}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"
STOP_EXISTING="${STOP_EXISTING:-1}"
FORCE_KILL="${FORCE_KILL:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/DPO_Finetune_Data_Multimodel_v1.4gpu.${TIMESTAMP}.stdout.log}"
MONITOR_TXT="${MONITOR_TXT:-${LOG_PATH%.log}.monitor.txt}"
MONITOR_JSON="${MONITOR_JSON:-${LOG_PATH%.log}.monitor.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export GPUS="${GPUS:-4,5,6,7}"
export PARALLEL_METHODS="${PARALLEL_METHODS:-4}"
export PARALLEL_VIDEOS="${PARALLEL_VIDEOS:-4}"
export MASK_SEEDS_PER_VIDEO="${MASK_SEEDS_PER_VIDEO:-1}"
export ENABLE_LPIPS="${ENABLE_LPIPS:-1}"
export ENABLE_VBENCH="${ENABLE_VBENCH:-1}"
export CANDIDATE_RETENTION="${CANDIDATE_RETENTION:-none}"
export CLEANUP_FAILED="${CLEANUP_FAILED:-1}"
export METHOD_GPU_MAP="${METHOD_GPU_MAP:-cococo=4,5;diffueraser=6,7;propainter=7,6,5,4;minimax=5,4,7,6}"

find_generation_pids() {
  ps -efww | awk '/run_multimodel_dpo_generation_h20.sh|generate_multimodel_dpo_dataset.py/ && !/awk/ {print $2}'
}

if [[ "${STOP_EXISTING}" == "1" ]]; then
  PIDS="$(find_generation_pids || true)"
  if [[ -n "${PIDS}" ]]; then
    echo "[stop] existing generation pids: ${PIDS//$'\n'/ }"
    xargs -r kill <<<"${PIDS}" || true
    sleep 5
    STILL="$(find_generation_pids || true)"
    if [[ -n "${STILL}" && "${FORCE_KILL}" == "1" ]]; then
      echo "[stop] force kill lingering pids: ${STILL//$'\n'/ }"
      xargs -r kill -9 <<<"${STILL}" || true
      sleep 2
    fi
  else
    echo "[stop] no existing generation process found"
  fi
fi

mkdir -p "$(dirname "${LOG_PATH}")"
mkdir -p "${OUT_ROOT}"

echo "[launch] project_root=${PROJECT_ROOT}"
echo "[launch] output_root=${OUT_ROOT}"
echo "[launch] log_path=${LOG_PATH}"
echo "[launch] gpus=${GPUS}"
echo "[launch] parallel_videos=${PARALLEL_VIDEOS}"
echo "[launch] method_gpu_map=${METHOD_GPU_MAP}"

nohup bash "${PROJECT_ROOT}/DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh" \
  >"${LOG_PATH}" 2>&1 &
LAUNCHER_PID=$!

echo "[launch] launcher_pid=${LAUNCHER_PID}"
sleep 15

echo
echo "=== Immediate Check: command line ==="
ps -efww | grep generate_multimodel_dpo_dataset.py | grep -v grep || true

echo
echo "=== Immediate Check: scheduler / errors ==="
grep -a -nE '\[run\]|\[scheduler\]|Traceback|RuntimeError|ERROR|command failed|failed:' "${LOG_PATH}" | tail -n 80 || true

echo
echo "=== Immediate Check: GPU snapshot ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

echo
echo "[monitor] duration=${RUNTIME_SEC}s interval=${MONITOR_INTERVAL}s"
python "${PROJECT_ROOT}/DPO_finetune/scripts/monitor_multimodel_runtime_h20.py" \
  --duration "${RUNTIME_SEC}" \
  --interval "${MONITOR_INTERVAL}" \
  --output-root "${OUT_ROOT}" \
  --log-path "${LOG_PATH}" \
  --gpus "${GPUS}" \
  --match H20_Video_inpainting_DPO \
  --json-out "${MONITOR_JSON}" | tee "${MONITOR_TXT}"

echo
echo "=== Post Check: recent log errors ==="
grep -a -nE 'Traceback|RuntimeError|ERROR|command failed|failed:' "${LOG_PATH}" | tail -n 80 || true

echo
echo "=== Post Check: usable entries ==="
python - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
manifest_path = out_root / "manifest.json"
usable_entries = len(json.load(open(manifest_path))) if manifest_path.exists() else 0
print(f"usable_entries={usable_entries}")
PY

cat <<EOF

[done] 4-GPU 1h check finished.
  log:          ${LOG_PATH}
  monitor text: ${MONITOR_TXT}
  monitor json: ${MONITOR_JSON}

Most useful sections to review:
  - === Runtime Window ===
  - === Log Stats ===
  - === GPU Summary ===
  - === Method Peak Summary ===
EOF
