"""Read-only multi-session KEEP-recall soak for the local Jev primary.

A tool result is a known KEEP-positive only when a later user or assistant message
repeats a distinctive fingerprint from it. Unreferenced results stay unlabeled.
Reports contain aggregate metrics and hashed identifiers, never transcript text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from plugins.context_engine import load_context_engine


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def fingerprints(text: str) -> set[str]:
    value = normalize(text)
    found: set[str] = set()
    for pattern in (
        r"(?<![0-9a-f])[0-9a-f]{16,64}(?![0-9a-f])",
        r"/(?:[a-z0-9_.-]+/){1,}[a-z0-9_.-]+",
        r"\b[a-z][a-z0-9_.-]{15,}\b",
    ):
        for item in re.findall(pattern, value):
            if len(set(item)) >= 5:
                found.add(item[:96])
    sample = value[:1600] + " " + value[-1600:]
    for start in range(0, max(0, len(sample) - 48), 24):
        item = sample[start:start + 48].strip()
        if len(item) >= 40 and sum(char.isalnum() for char in item) >= 24 and len(set(item)) >= 12:
            found.add(item)
    return found


def future_texts(messages: list[dict[str, Any]]) -> list[str]:
    output = [""] * len(messages)
    accumulated = ""
    for index in range(len(messages) - 1, -1, -1):
        output[index] = accumulated
        message = messages[index]
        if message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str):
            accumulated = normalize(message["content"]) + " " + accumulated
    return output


def positive_ids(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    futures = future_texts(messages)
    positives: dict[str, dict[str, Any]] = {}
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or not message.get("tool_call_id"):
            continue
        content = message.get("content") or ""
        if not isinstance(content, str) or len(content) < 80 or "jev-compaction:" in content:
            continue
        hits = [item for item in fingerprints(content) if item in futures[index]]
        if hits:
            positives[str(message["tool_call_id"])] = {
                "message_index": index,
                "result_chars": len(content),
                "strongest_fingerprint_chars": max(map(len, hits)),
            }
    return positives


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def spread_pick(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[0]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


def build_quality(rows: list[dict[str, Any]], *, dry_run: bool, keep_threshold: float) -> dict[str, Any]:
    if dry_run:
        return {
            "known_positive_keep_recall": None,
            "false_truncates": None,
            "zero_error_one_sided_95_upper_rate": None,
            "minimum_positive_result_score": None,
            "result_score_p05": None,
            "result_score_p50": None,
            "result_score_p95": None,
            "threshold_headroom_ratio": None,
        }
    scores = [float(row["result_score"]) for row in rows]
    false_truncates = [row for row in rows if row["production_action"] != "keep"]
    count = len(scores)
    return {
        "known_positive_keep_recall": (count - len(false_truncates)) / count if count else None,
        "false_truncates": len(false_truncates),
        "zero_error_one_sided_95_upper_rate": (
            1 - math.pow(0.05, 1 / count) if count and not false_truncates else None
        ),
        "minimum_positive_result_score": min(scores) if scores else None,
        "result_score_p05": percentile(scores, 0.05),
        "result_score_p50": percentile(scores, 0.50),
        "result_score_p95": percentile(scores, 0.95),
        "threshold_headroom_ratio": min(scores) / keep_threshold if scores else None,
    }


def select_sessions(candidates: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(candidates, key=lambda item: float(item.get("started_at") or 0)):
        by_source[str(row.get("source") or "unknown")].append(row)
    total = sum(len(rows) for rows in by_source.values())
    quotas = {
        source: min(len(rows), max(1, round(maximum * len(rows) / total)))
        for source, rows in by_source.items()
    }
    while sum(quotas.values()) > maximum:
        reducible = [key for key in quotas if quotas[key] > 1]
        if not reducible:
            break
        quotas[max(reducible, key=lambda key: quotas[key])] -= 1
    while sum(quotas.values()) < maximum:
        expandable = [key for key, rows in by_source.items() if quotas[key] < len(rows)]
        if not expandable:
            break
        key = max(expandable, key=lambda item: len(by_source[item]) - quotas[item])
        quotas[key] += 1
    selected = []
    for source, rows in by_source.items():
        selected.extend(spread_pick(rows, quotas[source]))
    return sorted(selected, key=lambda item: float(item.get("started_at") or 0))


def load_candidates(export_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    session_count = message_count = tool_count = positive_count = 0
    sources: Counter[str] = Counter()
    timestamps: list[float] = []
    with export_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            session_count += 1
            messages = row.get("messages") or []
            message_count += len(messages)
            tool_count += sum(message.get("role") == "tool" for message in messages)
            positives = positive_ids(messages)
            if positives:
                row["_positives"] = positives
                candidates.append(row)
                positive_count += len(positives)
                sources[str(row.get("source") or "unknown")] += 1
                timestamps.append(float(row.get("started_at") or 0))
    return candidates, {
        "export_sessions": session_count,
        "export_messages": message_count,
        "export_tool_results": tool_count,
        "positive_sites": positive_count,
        "positive_sessions": len(candidates),
        "positive_sessions_by_source": dict(sorted(sources.items())),
        "corpus_started_at": datetime.fromtimestamp(min(timestamps), timezone.utc).isoformat() if timestamps else None,
        "corpus_ended_at": datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat() if timestamps else None,
    }


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sessions", type=int, default=96)
    parser.add_argument("--max-sites-per-session", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates, inventory = load_candidates(args.export)
    selected = select_sessions(candidates, min(args.max_sessions, len(candidates)))
    config = yaml.safe_load(Path("HERMES_HOME/config.yaml").read_text(encoding="utf-8"))
    settings = dict(config["context_engine"]["settings"])
    if config["context"]["engine"] != "jev_compaction" or settings.get("decision_backend") != "local":
        raise SystemExit("production local Jev is not active")
    base = load_context_engine("jev_compaction")
    if base is None:
        raise SystemExit("jev_compaction unavailable")
    engine = base.__class__(settings)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    latencies: list[float] = []
    selected_by_source: Counter[str] = Counter()
    run_started = time.monotonic()
    for ordinal, session in enumerate(selected):
        messages = session.get("messages") or []
        engine_sites = {site["tid"]: site for site in engine._call_sites(messages)}
        eligible = [
            (tool_id, metadata, engine_sites[tool_id])
            for tool_id, metadata in session["_positives"].items()
            if tool_id in engine_sites
        ]
        eligible.sort(key=lambda item: (-item[1]["strongest_fingerprint_chars"], item[1]["message_index"]))
        eligible = eligible[:max(1, args.max_sites_per_session)]
        if not eligible:
            continue
        selected_by_source[str(session.get("source") or "unknown")] += 1
        if args.dry_run:
            for tool_id, metadata, _site in eligible:
                rows.append({
                    "session": short_hash(str(session["id"])),
                    "source": session.get("source"),
                    "tool": short_hash(tool_id),
                    "result_chars": metadata["result_chars"],
                    "fingerprint_chars": metadata["strongest_fingerprint_chars"],
                })
            continue
        state, state_tokens = engine._fit_state(messages)
        questions = engine._questions([item[2] for item in eligible])
        request_started = time.monotonic()
        try:
            answers = engine._local_ask(state, questions, float(settings["local_timeout_s"]))
        except Exception as exc:
            failures.append({"session": short_hash(str(session["id"])), "error": type(exc).__name__})
            continue
        latencies.append(time.monotonic() - request_started)
        for tool_id, metadata, _site in eligible:
            result_score = float(answers[f"keep_result_{tool_id}"]["noul"])
            call_score = float(answers[f"keep_call_{tool_id}"]["noul"])
            keep = float(settings["local_keep_threshold"])
            truncate = float(settings["local_truncate_threshold"])
            rows.append({
                "session": short_hash(str(session["id"])),
                "source": session.get("source"),
                "tool": short_hash(tool_id),
                "result_chars": metadata["result_chars"],
                "fingerprint_chars": metadata["strongest_fingerprint_chars"],
                "state_tokens_est": state_tokens,
                "call_score": call_score,
                "result_score": result_score,
                "production_action": "keep" if result_score >= keep else (
                    "marker" if call_score < truncate and result_score < truncate else "truncate"
                ),
            })
        print(json.dumps({"progress": ordinal + 1, "sessions": len(selected), "scored": len(rows)}), flush=True)

    quality = build_quality(rows, dry_run=args.dry_run, keep_threshold=float(settings["local_keep_threshold"]))
    false_rows = [] if args.dry_run else [row for row in rows if row["production_action"] != "keep"]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic future-reference KEEP positives; unlabeled sites excluded",
        "privacy": "redacted local export; aggregate/hash-only report; no transcript content",
        "production": {
            "head": "b4bd8177bd8cd3d6ceda9c71557d99c272dba189",
            "keep_threshold": settings["local_keep_threshold"],
            "truncate_threshold": settings["local_truncate_threshold"],
            "decision_backend": settings["decision_backend"],
        },
        "inventory": inventory,
        "sample": {
            "selected_sessions": len(selected),
            "selected_sessions_by_source": dict(sorted(selected_by_source.items())),
            "max_sites_per_session": args.max_sites_per_session,
            "scored_positive_sites": 0 if args.dry_run else len(rows),
            "dry_run_candidate_sites": len(rows) if args.dry_run else None,
            "failed_sessions": len(failures),
        },
        "quality": quality,
        "performance": {
            "elapsed_seconds": time.monotonic() - run_started,
            "request_count": len(latencies),
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "latency_max_seconds": max(latencies) if latencies else None,
        },
        "false_truncate_rows": false_rows,
        "failures": failures,
        "rows": rows,
        "dry_run": args.dry_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "sessions": len(selected),
        "rows": len(rows),
        "false_truncates": quality["false_truncates"],
        "failures": len(failures),
        "elapsed_seconds": round(report["performance"]["elapsed_seconds"], 3),
    }, sort_keys=True))
    return 0 if args.dry_run or (rows and not failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
