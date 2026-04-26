#!/usr/bin/env python3
"""Diagnose multimodel DPO generation health, with extra focus on one method.

This script combines two views:

1. Output-root state (`*/meta.json`):
   - whether each method actually succeeded
   - whether it was later selected into `neg_frames_{1,2}`
   - quality-score and defect-bucket distribution
   - failure reasons

2. Generator log state (`[infer]`, `[candidate]`, `[done]`, `[skip]`):
   - how many launches each method received
   - how many `candidate ok` / `candidate fail` lines appeared
   - whether many launches are still unmatched (started but not finished in the log)

That lets us distinguish three common cases:
   - real method failures
   - low selection rate despite successful generation
   - low recent `candidate_ok` because the method is still in flight
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


INFER_RE = re.compile(r"^\[infer\]\s+(\S+)\s+(\w+)\s+on GPU\s+(\d+)")
CANDIDATE_RE = re.compile(r"^\[candidate\]\s+(\S+)\s+(\w+):\s+(ok|failed:.*)$")
DONE_RE = re.compile(r"^\[done\]\s+(\S+):\s+neg1=(\w+),\s+neg2=(\w+)")
SKIP_RE = re.compile(r"^\[skip\]\s+(\S+)(?::\s+(.*))?$")


def safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        x = float(value)
        if math.isfinite(x):
            return x
    return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shorten(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).strip().split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def normalize_reason(text: str) -> str:
    raw = shorten(text, limit=240)
    lowered = raw.lower()
    if not raw:
        return "unknown"
    if "adapter disabled" in lowered:
        return "adapter disabled"
    if "adapter command is still todo" in lowered:
        return "adapter command TODO"
    if "command failed with code" in lowered:
        match = re.search(r"command failed with code\s+(\d+)", lowered)
        if match:
            return f"command failed (exit {match.group(1)})"
        return "command failed"
    if "no output frames found" in lowered:
        return "no output frames found"
    if "cannot composite empty candidate" in lowered:
        return "empty candidate during compositing"
    if "failed to read image" in lowered or "failed to read mask" in lowered:
        return "failed to read generated asset"
    if "vbench failed" in lowered:
        return "vbench failed"
    if "need at least one successful model candidate" in lowered:
        return "no successful candidates to select"
    return raw


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(median(nums))


def fmt_float(value: Optional[float], ndigits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def fmt_pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "-"
    return f"{100.0 * numer / denom:.1f}%"


def top_counter(counter: Counter[str], limit: int = 5) -> List[Tuple[str, int]]:
    return counter.most_common(limit)


def parse_log(log_path: Optional[Path]) -> Dict[str, Any]:
    per_method: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "infer": 0,
            "candidate_ok": 0,
            "candidate_fail": 0,
            "unmatched": 0,
            "gpus": Counter(),
            "fail_reasons": Counter(),
            "sample_unmatched": [],
        }
    )
    per_key: DefaultDict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"infer": 0, "ok": 0, "fail": 0, "gpu": None}
    )
    summary = {
        "exists": bool(log_path and log_path.exists()),
        "done": 0,
        "skip_complete": 0,
        "skip_other": 0,
        "per_method": per_method,
    }
    if not log_path or not log_path.exists():
        return summary

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            match = INFER_RE.match(line)
            if match:
                video_id, method, gpu = match.groups()
                per_method[method]["infer"] += 1
                per_method[method]["gpus"][gpu] += 1
                key = (video_id, method)
                per_key[key]["infer"] += 1
                per_key[key]["gpu"] = gpu
                continue

            match = CANDIDATE_RE.match(line)
            if match:
                video_id, method, status = match.groups()
                key = (video_id, method)
                if status == "ok":
                    per_method[method]["candidate_ok"] += 1
                    per_key[key]["ok"] += 1
                else:
                    reason = normalize_reason(status.removeprefix("failed:").strip())
                    per_method[method]["candidate_fail"] += 1
                    per_method[method]["fail_reasons"][reason] += 1
                    per_key[key]["fail"] += 1
                continue

            if DONE_RE.match(line):
                summary["done"] += 1
                continue

            match = SKIP_RE.match(line)
            if match:
                _, reason = match.groups()
                if reason and "already complete" in reason:
                    summary["skip_complete"] += 1
                else:
                    summary["skip_other"] += 1

    for (video_id, method), stats in per_key.items():
        unmatched = max(0, stats["infer"] - stats["ok"] - stats["fail"])
        if unmatched:
            per_method[method]["unmatched"] += unmatched
            samples = per_method[method]["sample_unmatched"]
            if len(samples) < 8:
                samples.append(
                    {
                        "video_id": video_id,
                        "gpu": stats["gpu"],
                        "infer": stats["infer"],
                        "ok": stats["ok"],
                        "fail": stats["fail"],
                    }
                )
    return summary


def parse_output_root(output_root: Path) -> Dict[str, Any]:
    per_method: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "meta_entries": 0,
            "ok": 0,
            "fail": 0,
            "selected": 0,
            "quality_scores": [],
            "buckets": Counter(),
            "fail_reasons": Counter(),
            "score_key_presence": Counter(),
        }
    )
    video_failures: Counter[str] = Counter()
    total_meta = 0

    for meta_path in sorted(output_root.glob("*/meta.json")):
        total_meta += 1
        try:
            meta = read_json(meta_path)
        except Exception as exc:  # pragma: no cover - defensive
            video_failures[normalize_reason(str(exc))] += 1
            continue

        for selected_key in ("selected_neg_frames_1", "selected_neg_frames_2"):
            method = meta.get(selected_key)
            if method:
                per_method[method]["selected"] += 1

        if meta.get("error"):
            video_failures[normalize_reason(str(meta["error"]))] += 1

        for cand in meta.get("candidates", []):
            method = str(cand.get("method", "unknown"))
            stats = per_method[method]
            stats["meta_entries"] += 1
            if cand.get("ok"):
                stats["ok"] += 1
                q = safe_float(cand.get("quality_score"))
                if q is not None:
                    stats["quality_scores"].append(q)
                score = cand.get("score") or {}
                if isinstance(score, dict):
                    bucket = score.get("defect_bucket")
                    if bucket:
                        stats["buckets"][str(bucket)] += 1
                    for key, value in score.items():
                        if safe_float(value) is not None:
                            stats["score_key_presence"][str(key)] += 1
            else:
                stats["fail"] += 1
                stats["fail_reasons"][normalize_reason(str(cand.get("error", "")))] += 1

    manifest_entries = 0
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            if isinstance(manifest, dict):
                manifest_entries = len(manifest)
        except Exception:
            manifest_entries = 0

    return {
        "total_meta": total_meta,
        "manifest_entries": manifest_entries,
        "per_method": per_method,
        "video_failures": video_failures,
    }


def build_focus_notes(
    method: str,
    output_stats: Dict[str, Any],
    log_stats: Dict[str, Any],
) -> List[str]:
    notes: List[str] = []
    meta_ok = int(output_stats.get("ok", 0))
    meta_fail = int(output_stats.get("fail", 0))
    selected = int(output_stats.get("selected", 0))
    log_infer = int(log_stats.get("infer", 0))
    log_ok = int(log_stats.get("candidate_ok", 0))
    log_fail = int(log_stats.get("candidate_fail", 0))
    unmatched = int(log_stats.get("unmatched", 0))
    avg_q = mean_or_none(output_stats.get("quality_scores", []))
    bucket_counts: Counter[str] = output_stats.get("buckets", Counter())

    if log_infer > 0 and log_fail == 0 and unmatched > max(2, log_ok):
        notes.append(
            f"`{method}` has many launches without matching `[candidate]` lines yet "
            f"(`infer={log_infer}`, `ok={log_ok}`, `fail={log_fail}`, `unmatched={unmatched}`), "
            "so low recent `candidate_ok` looks more like in-flight backlog than method failure."
        )

    if meta_fail > 0 and meta_ok == 0:
        notes.append(
            f"`{method}` is failing at candidate generation time (`meta ok={meta_ok}`, `meta fail={meta_fail}`)."
        )
    elif meta_ok > 0 and selected == 0:
        notes.append(
            f"`{method}` is generating candidates, but none are being selected into negatives yet."
        )
    elif meta_ok > 0 and selected > 0:
        notes.append(
            f"`{method}` is generating usable candidates (`meta ok={meta_ok}`) and has been selected {selected} times."
        )

    if avg_q is not None:
        if avg_q > 0.75:
            notes.append(
                f"`{method}` average relative quality is high ({avg_q:.3f}); it may be landing closer to 'too good' than the target hard-negative band."
            )
        elif avg_q < 0.25:
            notes.append(
                f"`{method}` average relative quality is very low ({avg_q:.3f}); it may be too degraded to be a useful hard negative."
            )

    if bucket_counts:
        bucket, count = bucket_counts.most_common(1)[0]
        if count > 0:
            notes.append(f"`{method}` most common defect bucket is `{bucket}` ({count} samples).")

    if not notes:
        notes.append(f"No strong anomaly was detected for `{method}` from the available log/meta evidence.")
    return notes


def print_method_table(
    methods: List[str],
    output_data: Dict[str, Any],
    log_data: Dict[str, Any],
) -> None:
    header = (
        f"{'method':12s} {'meta_ok':>7s} {'meta_fail':>9s} {'selected':>9s} "
        f"{'sel/ok':>8s} {'avg_q':>7s} {'med_q':>7s} "
        f"{'log_infer':>9s} {'log_ok':>7s} {'log_fail':>9s} {'unmatched':>10s}"
    )
    print(header)
    print("-" * len(header))
    for method in methods:
        out_stats = output_data["per_method"].get(method, {})
        log_stats = log_data["per_method"].get(method, {})
        print(
            f"{method:12s} "
            f"{int(out_stats.get('ok', 0)):7d} "
            f"{int(out_stats.get('fail', 0)):9d} "
            f"{int(out_stats.get('selected', 0)):9d} "
            f"{fmt_pct(int(out_stats.get('selected', 0)), int(out_stats.get('ok', 0))):>8s} "
            f"{fmt_float(mean_or_none(out_stats.get('quality_scores', []))):>7s} "
            f"{fmt_float(median_or_none(out_stats.get('quality_scores', []))):>7s} "
            f"{int(log_stats.get('infer', 0)):9d} "
            f"{int(log_stats.get('candidate_ok', 0)):7d} "
            f"{int(log_stats.get('candidate_fail', 0)):9d} "
            f"{int(log_stats.get('unmatched', 0)):10d}"
        )


def build_summary_json(
    methods: List[str],
    output_data: Dict[str, Any],
    log_data: Dict[str, Any],
    focus_method: str,
) -> Dict[str, Any]:
    summary_methods: Dict[str, Any] = {}
    for method in methods:
        out_stats = output_data["per_method"].get(method, {})
        log_stats = log_data["per_method"].get(method, {})
        summary_methods[method] = {
            "meta_entries": int(out_stats.get("meta_entries", 0)),
            "meta_ok": int(out_stats.get("ok", 0)),
            "meta_fail": int(out_stats.get("fail", 0)),
            "selected": int(out_stats.get("selected", 0)),
            "selected_given_ok_pct": fmt_pct(int(out_stats.get("selected", 0)), int(out_stats.get("ok", 0))),
            "avg_quality_score": mean_or_none(out_stats.get("quality_scores", [])),
            "median_quality_score": median_or_none(out_stats.get("quality_scores", [])),
            "top_buckets": top_counter(out_stats.get("buckets", Counter())),
            "top_fail_reasons": top_counter(out_stats.get("fail_reasons", Counter())),
            "log_infer": int(log_stats.get("infer", 0)),
            "log_candidate_ok": int(log_stats.get("candidate_ok", 0)),
            "log_candidate_fail": int(log_stats.get("candidate_fail", 0)),
            "log_unmatched": int(log_stats.get("unmatched", 0)),
            "log_gpus": dict(log_stats.get("gpus", Counter())),
            "sample_unmatched": log_stats.get("sample_unmatched", []),
        }
    return {
        "total_meta": output_data["total_meta"],
        "manifest_entries": output_data["manifest_entries"],
        "log_exists": log_data["exists"],
        "log_done": log_data["done"],
        "log_skip_complete": log_data["skip_complete"],
        "log_skip_other": log_data["skip_other"],
        "methods": summary_methods,
        "video_failures": top_counter(output_data["video_failures"]),
        "focus_method": focus_method,
        "focus_notes": build_focus_notes(
            focus_method,
            output_data["per_method"].get(focus_method, {}),
            log_data["per_method"].get(focus_method, {}),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose multimodel DPO dataset generation state on H20.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--methods", default="propainter,cococo,diffueraser,minimax")
    parser.add_argument("--focus-method", default="propainter")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    log_path = Path(args.log_path).resolve() if args.log_path else None
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if args.focus_method not in methods:
        methods.append(args.focus_method)

    output_data = parse_output_root(output_root)
    log_data = parse_log(log_path)

    print("\n=== Output Root Summary ===")
    print(f"output_root={output_root}")
    print(f"meta_entries={output_data['total_meta']}")
    print(f"manifest_entries={output_data['manifest_entries']}")

    print("\n=== Log Summary ===")
    if log_data["exists"]:
        print(f"log_path={log_path}")
        print(f"done_count={log_data['done']}")
        print(f"skip_complete_count={log_data['skip_complete']}")
        print(f"skip_other_count={log_data['skip_other']}")
    else:
        print("log_path=(missing or not provided)")

    print("\n=== Method Summary ===")
    print_method_table(methods, output_data, log_data)

    focus_output = output_data["per_method"].get(args.focus_method, {})
    focus_log = log_data["per_method"].get(args.focus_method, {})
    print(f"\n=== Focus: {args.focus_method} ===")
    for note in build_focus_notes(args.focus_method, focus_output, focus_log):
        print(f"- {note}")

    focus_fail_reasons = top_counter(focus_output.get("fail_reasons", Counter()))
    if focus_fail_reasons:
        print("\nTop output-root failure reasons:")
        for reason, count in focus_fail_reasons:
            print(f"  {count:4d}  {reason}")

    focus_buckets = top_counter(focus_output.get("buckets", Counter()))
    if focus_buckets:
        print("\nTop quality buckets:")
        for bucket, count in focus_buckets:
            print(f"  {count:4d}  {bucket}")

    focus_log_fails = top_counter(focus_log.get("fail_reasons", Counter()))
    if focus_log_fails:
        print("\nTop log failure reasons:")
        for reason, count in focus_log_fails:
            print(f"  {count:4d}  {reason}")

    unmatched = focus_log.get("sample_unmatched", [])
    if unmatched:
        print("\nSample unmatched log launches:")
        for item in unmatched[:8]:
            print(
                f"  video_id={item['video_id']} gpu={item['gpu']} "
                f"infer={item['infer']} ok={item['ok']} fail={item['fail']}"
            )

    video_failures = top_counter(output_data["video_failures"])
    if video_failures:
        print("\nTop video-level selection / completion failures:")
        for reason, count in video_failures:
            print(f"  {count:4d}  {reason}")

    summary_json = build_summary_json(methods, output_data, log_data, args.focus_method)
    if args.json_out:
        json_out = Path(args.json_out).resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)
        print(f"\n[json] wrote {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
