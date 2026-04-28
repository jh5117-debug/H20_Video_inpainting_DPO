#!/usr/bin/env python3
"""Sample GPU memory/utilization and attribute peaks to multimodel DPO workers.

Designed for H20 dataset generation runs. It watches live GPU processes via
``nvidia-smi`` and groups them into:

- propainter
- cococo
- diffueraser
- minimax
- scorer_orchestrator (the main generate_multimodel_dpo_dataset.py process)
- other / unknown

Example:

    python DPO_finetune/scripts/profile_multimodel_peakmem_h20.py \
      --duration 600 --gpus 4,5,6,7 \
      --match Video_inpainting_DPO \
      --json-out /tmp/multimodel_peakmem.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ProcPeak:
    pid: int
    gpu: int
    label: str
    process_name: str
    max_used_mib: int
    first_seen_ts: float
    last_seen_ts: float
    samples: int
    cmdline: str


def run_cmd(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def parse_csv_lines(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append([part.strip() for part in line.split(",")])
    return rows


def gpu_inventory() -> Dict[str, Dict[str, int]]:
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    inventory: Dict[str, Dict[str, int]] = {}
    for idx_s, uuid, total_s in parse_csv_lines(out):
        inventory[uuid] = {"index": int(idx_s), "memory_total_mib": int(total_s)}
    return inventory


def gpu_status() -> Dict[int, Dict[str, int]]:
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    status: Dict[int, Dict[str, int]] = {}
    for idx_s, used_s, util_s in parse_csv_lines(out):
        status[int(idx_s)] = {
            "memory_used_mib": int(used_s),
            "util_gpu_pct": int(util_s),
        }
    return status


def compute_apps() -> List[Tuple[str, int, str, int]]:
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: List[Tuple[str, int, str, int]] = []
    for uuid, pid_s, process_name, used_s in parse_csv_lines(out):
        rows.append((uuid, int(pid_s), process_name, int(used_s)))
    return rows


def cmdline_for_pid(pid: int) -> str:
    try:
        return run_cmd(["ps", "-p", str(pid), "-o", "args="]).strip()
    except Exception:
        return ""


def classify_process(cmdline: str, process_name: str) -> str:
    text = f"{process_name} {cmdline}"
    if "generate_multimodel_dpo_dataset.py" in text:
        return "scorer_orchestrator"
    if "infer_propainter_candidate.py" in text:
        return "propainter"
    if "infer_cococo_candidate.py" in text or "valid_code_release" in text:
        return "cococo"
    if "infer_diffueraser_candidate.py" in text or "inference/run_OR.py" in text:
        return "diffueraser"
    if "infer_minimax_candidate.py" in text or "pipeline_minimax_remover" in text:
        return "minimax"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile multimodel GPU peak memory on H20.")
    parser.add_argument("--duration", type=float, default=300.0, help="Seconds to monitor.")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds.")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU indices to include. Empty = all.")
    parser.add_argument(
        "--match",
        default="Video_inpainting_DPO",
        help="Only attribute processes whose cmdline contains this substring. Empty = no filter.",
    )
    parser.add_argument("--json-out", default="", help="Optional JSON summary output path.")
    args = parser.parse_args()

    include_gpus = {
        int(x.strip()) for x in args.gpus.split(",") if x.strip()
    } if args.gpus.strip() else None
    match_substring = args.match.strip()

    inventory = gpu_inventory()
    gpu_peaks: Dict[int, Dict[str, int]] = {}
    proc_peaks: Dict[Tuple[int, int], ProcPeak] = {}
    method_gpu_peaks: Dict[Tuple[str, int], int] = {}
    started = time.time()
    deadline = started + args.duration

    while time.time() < deadline:
        try:
            status = gpu_status()
            apps = compute_apps()
        except Exception as exc:
            print(f"[warn] sampling failed: {exc}", file=sys.stderr)
            time.sleep(args.interval)
            continue

        now = time.time()

        for gpu_idx, st in status.items():
            if include_gpus is not None and gpu_idx not in include_gpus:
                continue
            peak = gpu_peaks.setdefault(
                gpu_idx,
                {
                    "peak_total_memory_used_mib": 0,
                    "peak_util_gpu_pct": 0,
                    "memory_total_mib": 0,
                },
            )
            peak["peak_total_memory_used_mib"] = max(peak["peak_total_memory_used_mib"], st["memory_used_mib"])
            peak["peak_util_gpu_pct"] = max(peak["peak_util_gpu_pct"], st["util_gpu_pct"])
            for uuid, meta in inventory.items():
                if meta["index"] == gpu_idx:
                    peak["memory_total_mib"] = meta["memory_total_mib"]
                    break

        for uuid, pid, process_name, used_mib in apps:
            meta = inventory.get(uuid)
            if meta is None:
                continue
            gpu_idx = meta["index"]
            if include_gpus is not None and gpu_idx not in include_gpus:
                continue

            cmdline = cmdline_for_pid(pid)
            if match_substring and match_substring not in cmdline:
                continue

            label = classify_process(cmdline, process_name)
            key = (pid, gpu_idx)
            existing = proc_peaks.get(key)
            if existing is None:
                proc_peaks[key] = ProcPeak(
                    pid=pid,
                    gpu=gpu_idx,
                    label=label,
                    process_name=process_name,
                    max_used_mib=used_mib,
                    first_seen_ts=now,
                    last_seen_ts=now,
                    samples=1,
                    cmdline=cmdline,
                )
            else:
                existing.max_used_mib = max(existing.max_used_mib, used_mib)
                existing.last_seen_ts = now
                existing.samples += 1

            mg_key = (label, gpu_idx)
            method_gpu_peaks[mg_key] = max(method_gpu_peaks.get(mg_key, 0), used_mib)

        time.sleep(args.interval)

    summary = {
        "duration_sec": args.duration,
        "interval_sec": args.interval,
        "match_substring": match_substring,
        "gpu_peaks": gpu_peaks,
        "method_gpu_peaks": {
            f"{label}@gpu{gpu}": peak for (label, gpu), peak in sorted(method_gpu_peaks.items())
        },
        "process_peaks": [
            asdict(p) for p in sorted(
                proc_peaks.values(),
                key=lambda item: (item.gpu, item.label, -item.max_used_mib, item.pid),
            )
        ],
    }

    print("\n=== GPU Peak Summary ===")
    for gpu in sorted(gpu_peaks):
        info = gpu_peaks[gpu]
        total = info.get("memory_total_mib", 0)
        used = info["peak_total_memory_used_mib"]
        util = info["peak_util_gpu_pct"]
        frac = (used / total * 100.0) if total else 0.0
        print(
            f"GPU {gpu}: peak_total_mem={used} MiB / {total} MiB "
            f"({frac:.1f}%), peak_util={util}%"
        )

    print("\n=== Method Peak Summary ===")
    if not method_gpu_peaks:
        print("(no matching GPU processes found)")
    else:
        for (label, gpu), peak in sorted(method_gpu_peaks.items()):
            print(f"{label:20s} gpu{gpu}: {peak:6d} MiB")

    print("\n=== Process Peak Summary ===")
    if not proc_peaks:
        print("(no matching GPU processes found)")
    else:
        for proc in sorted(proc_peaks.values(), key=lambda item: (item.gpu, item.label, -item.max_used_mib)):
            tail = proc.cmdline
            if len(tail) > 140:
                tail = "..." + tail[-137:]
            print(
                f"gpu{proc.gpu} pid={proc.pid:<8d} {proc.label:20s} "
                f"peak={proc.max_used_mib:6d} MiB samples={proc.samples:4d} cmd={tail}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[done] wrote JSON summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
