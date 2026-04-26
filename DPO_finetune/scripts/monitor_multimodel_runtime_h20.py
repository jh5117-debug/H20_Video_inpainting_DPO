#!/usr/bin/env python3
"""Monitor multimodel DPO generation throughput and GPU activity on H20.

Example:

    python DPO_finetune/scripts/monitor_multimodel_runtime_h20.py \
      --duration 3600 \
      --interval 2 \
      --output-root /home/nvme01/H20_Video_inpainting_DPO/DPO_Finetune_Data_Multimodel_v1 \
      --log-path /home/nvme01/H20_Video_inpainting_DPO/DPO_Finetune_Data_Multimodel_v1.repair_short_diffueraser.stdout.log \
      --gpus 4,5,6,7 \
      --match H20_Video_inpainting_DPO \
      --json-out /tmp/h20_multimodel_monitor.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple


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


_CMDLINE_CACHE: Dict[int, str] = {}


def cmdline_for_pid(pid: int) -> str:
    cached = _CMDLINE_CACHE.get(pid)
    if cached is not None:
        return cached
    try:
        value = run_cmd(["ps", "-p", str(pid), "-o", "args="]).strip()
    except Exception:
        value = ""
    _CMDLINE_CACHE[pid] = value
    return value


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


def read_manifest_entries(output_root: Path) -> int:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        return 0
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return 0
    return len(manifest)


def consume_new_log_lines(log_path: Path, state: Dict[str, object]) -> List[str]:
    if not log_path.exists():
        return []
    offset = int(state.get("offset", 0))
    partial = str(state.get("partial", ""))
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        chunk = f.read()
        state["offset"] = f.tell()
    if not chunk:
        return []
    text = partial + chunk
    if text.endswith("\n"):
        state["partial"] = ""
        return text.splitlines()
    if "\n" not in text:
        state["partial"] = text
        return []
    body, tail = text.rsplit("\n", 1)
    state["partial"] = tail
    return body.splitlines()


def update_log_stats(lines: List[str], stats: Dict[str, object]) -> None:
    infer_re = re.compile(r"^\[infer\]\s+\S+\s+(\w+)\s+on GPU\s+(\d+)")
    cand_re = re.compile(r"^\[candidate\]\s+\S+\s+(\w+):\s+(ok|failed:.*)$")

    for line in lines:
        if line.startswith("[done] "):
            stats["done_count"] = int(stats["done_count"]) + 1
        if line.startswith("[skip] ") and "already complete" in line:
            stats["skip_complete_count"] = int(stats["skip_complete_count"]) + 1
        if "padded short clip from" in line:
            stats["padded_short_clip_count"] = int(stats["padded_short_clip_count"]) + 1
        if line.startswith("[resume] seeded source_counts"):
            stats["resume_seed_line"] = line

        match = infer_re.match(line)
        if match:
            method, gpu = match.groups()
            infer_counts = stats["infer_counts"]
            infer_counts[method] = infer_counts.get(method, 0) + 1
            infer_gpu_counts = stats["infer_gpu_counts"]
            key = f"{method}@gpu{gpu}"
            infer_gpu_counts[key] = infer_gpu_counts.get(key, 0) + 1
            continue

        match = cand_re.match(line)
        if match:
            method, status = match.groups()
            if status == "ok":
                ok_counts = stats["candidate_ok_counts"]
                ok_counts[method] = ok_counts.get(method, 0) + 1
            else:
                fail_counts = stats["candidate_fail_counts"]
                fail_counts[method] = fail_counts.get(method, 0) + 1


def compute_recent_rate(entries_timeline: List[Tuple[float, int]], window_sec: float) -> float:
    if len(entries_timeline) < 2:
        return 0.0
    end_ts, end_entries = entries_timeline[-1]
    start_ts = end_ts - window_sec
    earlier = None
    for ts, entries in reversed(entries_timeline):
        if ts <= start_ts:
            earlier = (ts, entries)
            break
    if earlier is None:
        earlier = entries_timeline[0]
    ts0, e0 = earlier
    dt = end_ts - ts0
    if dt <= 0:
        return 0.0
    return (end_entries - e0) * 3600.0 / dt


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor multimodel DPO runtime and throughput on H20.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--match", default="H20_Video_inpainting_DPO")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    log_path = Path(args.log_path).resolve()
    include_gpus = {int(x.strip()) for x in args.gpus.split(",") if x.strip()}
    inventory = gpu_inventory()

    log_state: Dict[str, object] = {"offset": log_path.stat().st_size if log_path.exists() else 0, "partial": ""}
    log_stats: Dict[str, object] = {
        "done_count": 0,
        "skip_complete_count": 0,
        "padded_short_clip_count": 0,
        "resume_seed_line": "",
        "infer_counts": {},
        "infer_gpu_counts": {},
        "candidate_ok_counts": {},
        "candidate_fail_counts": {},
    }

    gpu_stats: Dict[int, Dict[str, float]] = {
        gpu: {
            "samples": 0,
            "util_sum": 0.0,
            "mem_sum": 0.0,
            "peak_util": 0.0,
            "peak_mem": 0.0,
            "active_over_10": 0.0,
            "busy_over_80": 0.0,
            "memory_total_mib": 0.0,
        }
        for gpu in include_gpus
    }
    for uuid, meta in inventory.items():
        gpu = meta["index"]
        if gpu in gpu_stats:
            gpu_stats[gpu]["memory_total_mib"] = float(meta["memory_total_mib"])

    proc_peaks: Dict[Tuple[int, int], ProcPeak] = {}
    method_gpu_peaks: Dict[Tuple[str, int], int] = {}
    method_gpu_samples: DefaultDict[Tuple[str, int], int] = defaultdict(int)
    entries_timeline: List[Tuple[float, int]] = []

    start_ts = time.time()
    end_ts = start_ts + args.duration
    start_entries = read_manifest_entries(output_root)
    entries_timeline.append((start_ts, start_entries))

    while time.time() < end_ts:
        now = time.time()

        lines = consume_new_log_lines(log_path, log_state)
        if lines:
            update_log_stats(lines, log_stats)

        try:
            status = gpu_status()
            apps = compute_apps()
        except Exception as exc:
            print(f"[warn] sampling failed: {exc}", file=sys.stderr)
            time.sleep(args.interval)
            continue

        entries = read_manifest_entries(output_root)
        entries_timeline.append((now, entries))

        for gpu in include_gpus:
            st = status.get(gpu)
            if st is None:
                continue
            g = gpu_stats[gpu]
            g["samples"] += 1
            g["util_sum"] += st["util_gpu_pct"]
            g["mem_sum"] += st["memory_used_mib"]
            g["peak_util"] = max(g["peak_util"], float(st["util_gpu_pct"]))
            g["peak_mem"] = max(g["peak_mem"], float(st["memory_used_mib"]))
            if st["util_gpu_pct"] >= 10:
                g["active_over_10"] += 1
            if st["util_gpu_pct"] >= 80:
                g["busy_over_80"] += 1

        for uuid, pid, process_name, used_mib in apps:
            meta = inventory.get(uuid)
            if meta is None:
                continue
            gpu = meta["index"]
            if gpu not in include_gpus:
                continue
            cmdline = cmdline_for_pid(pid)
            if args.match and args.match not in cmdline:
                continue
            label = classify_process(cmdline, process_name)
            key = (pid, gpu)
            existing = proc_peaks.get(key)
            if existing is None:
                proc_peaks[key] = ProcPeak(
                    pid=pid,
                    gpu=gpu,
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
            method_gpu_peaks[(label, gpu)] = max(method_gpu_peaks.get((label, gpu), 0), used_mib)
            method_gpu_samples[(label, gpu)] += 1

        time.sleep(args.interval)

    final_ts = time.time()
    final_entries = read_manifest_entries(output_root)
    entries_timeline.append((final_ts, final_entries))
    lines = consume_new_log_lines(log_path, log_state)
    if lines:
        update_log_stats(lines, log_stats)

    duration_sec = max(1e-6, final_ts - start_ts)
    entries_delta = final_entries - start_entries
    entries_per_hour = entries_delta * 3600.0 / duration_sec
    recent_10m_per_hour = compute_recent_rate(entries_timeline, 600.0)

    summary = {
        "window": {
            "start_time_epoch": start_ts,
            "end_time_epoch": final_ts,
            "duration_sec": duration_sec,
        },
        "paths": {
            "output_root": str(output_root),
            "log_path": str(log_path),
        },
        "entries": {
            "start": start_entries,
            "end": final_entries,
            "delta": entries_delta,
            "entries_per_hour": entries_per_hour,
            "recent_10m_entries_per_hour": recent_10m_per_hour,
        },
        "log_stats": log_stats,
        "gpu_summary": {},
        "method_gpu_peaks": {
            f"{label}@gpu{gpu}": peak for (label, gpu), peak in sorted(method_gpu_peaks.items())
        },
        "method_gpu_sample_counts": {
            f"{label}@gpu{gpu}": count for (label, gpu), count in sorted(method_gpu_samples.items())
        },
        "process_peaks": [
            asdict(p)
            for p in sorted(proc_peaks.values(), key=lambda item: (item.gpu, item.label, -item.max_used_mib, item.pid))
        ],
    }

    print("\n=== Runtime Window ===")
    print(f"duration_sec={duration_sec:.1f}")
    print(f"entries_start={start_entries}")
    print(f"entries_end={final_entries}")
    print(f"entries_delta={entries_delta}")
    print(f"entries_per_hour={entries_per_hour:.2f}")
    print(f"recent_10m_entries_per_hour={recent_10m_per_hour:.2f}")

    print("\n=== Log Stats ===")
    print(f"done_count={log_stats['done_count']}")
    print(f"skip_complete_count={log_stats['skip_complete_count']}")
    print(f"padded_short_clip_count={log_stats['padded_short_clip_count']}")
    if log_stats["resume_seed_line"]:
        print(log_stats["resume_seed_line"])
    print(f"candidate_ok_counts={json.dumps(log_stats['candidate_ok_counts'], ensure_ascii=False)}")
    print(f"candidate_fail_counts={json.dumps(log_stats['candidate_fail_counts'], ensure_ascii=False)}")
    print(f"infer_counts={json.dumps(log_stats['infer_counts'], ensure_ascii=False)}")

    print("\n=== GPU Summary ===")
    for gpu in sorted(gpu_stats):
        g = gpu_stats[gpu]
        samples = max(1.0, g["samples"])
        avg_util = g["util_sum"] / samples
        avg_mem = g["mem_sum"] / samples
        active_ratio = 100.0 * g["active_over_10"] / samples
        busy_ratio = 100.0 * g["busy_over_80"] / samples
        total_mem = g["memory_total_mib"]
        peak_mem_frac = 100.0 * g["peak_mem"] / total_mem if total_mem else 0.0
        summary["gpu_summary"][gpu] = {
            "samples": int(g["samples"]),
            "avg_util_gpu_pct": avg_util,
            "peak_util_gpu_pct": g["peak_util"],
            "avg_memory_used_mib": avg_mem,
            "peak_memory_used_mib": g["peak_mem"],
            "peak_memory_used_pct_of_total": peak_mem_frac,
            "active_ratio_over_10pct": active_ratio,
            "busy_ratio_over_80pct": busy_ratio,
            "memory_total_mib": total_mem,
        }
        print(
            f"GPU {gpu}: avg_util={avg_util:.1f}% peak_util={g['peak_util']:.0f}% "
            f"avg_mem={avg_mem:.0f}MiB peak_mem={g['peak_mem']:.0f}MiB "
            f"active>10%={active_ratio:.1f}% busy>80%={busy_ratio:.1f}%"
        )

    print("\n=== Method Peak Summary ===")
    if not method_gpu_peaks:
        print("(no matching GPU processes found)")
    else:
        for (label, gpu), peak in sorted(method_gpu_peaks.items()):
            count = method_gpu_samples[(label, gpu)]
            print(f"{label:20s} gpu{gpu}: peak={peak:6d} MiB samples={count:4d}")

    print("\n=== Process Peak Summary ===")
    if not proc_peaks:
        print("(no matching GPU processes found)")
    else:
        for proc in sorted(proc_peaks.values(), key=lambda item: (item.gpu, item.label, -item.max_used_mib, item.pid)):
            tail = proc.cmdline
            if len(tail) > 140:
                tail = "..." + tail[-137:]
            print(
                f"gpu{proc.gpu} pid={proc.pid:<8d} {proc.label:20s} "
                f"peak={proc.max_used_mib:6d} MiB samples={proc.samples:4d} cmd={tail}"
            )

    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[done] wrote JSON summary to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
