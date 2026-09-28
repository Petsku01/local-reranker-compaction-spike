from __future__ import annotations

import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

MODEL = "Qwen/Qwen3-Reranker-0.6B"
OUT = Path(__file__).with_name("results.json")
INSTRUCTION = (
    "Decide whether the document must remain available to correctly continue the current task. "
    "Keep current evidence, errors, constraints, decisions, and verification. "
    "Discard unrelated or superseded output."
)

SCENARIOS = [
    {
        "task": "Investigate why the PostgreSQL backup cron fails with permission denied and fix it without losing existing archives.",
        "items": [
            ("backup_error", "Current cron error: PermissionError: [Errno 13] opening /srv/backups/db.dump", 1),
            ("backup_permissions", "ls -ld /srv/backups => drwx------ root root /srv/backups", 1),
            ("backup_verified", "Fresh backup completed; archive checksum verified and restore listing succeeded.", 1),
            ("backup_constraint_fi", "Käyttäjän rajoite: olemassa olevia varmuuskopioarkistoja ei saa poistaa.", 1),
            ("weather", "Turun sää: 12 C, heikkoa sadetta.", 0),
            ("npm_output", "npm install completed: added 418 packages and audited 419 packages.", 0),
        ],
    },
    {
        "task": "Fix the Python API returning HTTP 500 when email is null and add a regression test without changing the database schema.",
        "items": [
            ("stack_trace", "Traceback: normalize_email calls email.lower(); AttributeError: 'NoneType' object has no attribute 'lower'.", 1),
            ("source_bug", "def normalize_email(email): return email.strip().lower()", 1),
            ("test_verified", "Regression test test_null_email_returns_422 passed; targeted suite: 12 passed.", 1),
            ("schema_constraint", "User decision: do not add a database migration for this fix.", 1),
            ("css_diff", "Changed .hero-title font-size from 42px to 40px on mobile.", 0),
            ("old_git_status", "git status from another repository: README.md modified.", 0),
        ],
    },
    {
        "task": "Find why the Samsung Odyssey G9 wallpaper is blurry and improve it without changing display resolution until the user approves.",
        "items": [
            ("display_mode", "DP-2 primary mode is 3840x1080@120 Hz with scale 1.25; 5120x1440@60 is also advertised.", 1),
            ("wallpaper_size", "Current wallpaper file dimensions: 1920x1080 pixels, JPEG quality 72.", 1),
            ("stop_constraint_fi", "Käyttäjä sanoi: pysähdy. Näyttöasetuksia ei saa muuttaa ennen uutta lupaa.", 1),
            ("network_ping", "ping 1.1.1.1: 12 ms, 0% packet loss.", 0),
            ("discord_log", "Discord gateway connected as Kuu#2777.", 0),
        ],
    },
]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def best_threshold(scores: list[float], expected: list[int]) -> tuple[float, float]:
    values = sorted(set(scores))
    candidates = [0.0, 1.0]
    candidates.extend((a + b) / 2 for a, b in zip(values, values[1:]))
    best = (0.0, -1.0)
    for threshold in candidates:
        accuracy = sum(int(score >= threshold) == label for score, label in zip(scores, expected)) / len(scores)
        if accuracy > best[1]:
            best = (threshold, accuracy)
    return best


def auc(scores: list[float], expected: list[int]) -> float:
    positives = [score for score, label in zip(scores, expected) if label == 1]
    negatives = [score for score, label in zip(scores, expected) if label == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def call_jev(scenario: dict) -> tuple[dict[str, float], float]:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    state_lines = [f"[user] {scenario['task']}"]
    questions = {}
    for ident, text, _expected in scenario["items"]:
        state_lines.append(f"[tool_result {ident}] {text}")
        questions[f"keep_{ident}"] = {
            "type": "noul",
            "instructions": (
                f"The tool result {ident} appears in the state. Knowing its exact content still matters "
                "for correctly continuing the user's current task. Should it stay available?"
            ),
        }
    payload = json.dumps({"state": "\n".join(state_lines), "model": "jev-latest", "questions": questions}).encode()
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    answers = body.get("answers") or {}
    scores = {}
    for ident, _text, _expected in scenario["items"]:
        value = (answers.get(f"keep_{ident}") or {}).get("noul")
        scores[ident] = float(value) if value is not None else float("nan")
    return scores, elapsed


def main() -> None:
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = CrossEncoder(
        MODEL,
        device="cuda",
        model_kwargs={"dtype": torch.bfloat16},
        max_length=4096,
    )
    load_s = time.perf_counter() - load_started

    rows = []
    local_times = []
    jev_times = []
    for scenario in SCENARIOS:
        pairs = [(scenario["task"], text) for _ident, text, _expected in scenario["items"]]
        started = time.perf_counter()
        raw_scores = model.predict(
            pairs,
            prompt=INSTRUCTION,
            batch_size=len(pairs),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        local_times.append(time.perf_counter() - started)
        jev_scores, jev_s = call_jev(scenario)
        jev_times.append(jev_s)
        for (ident, text, expected), raw in zip(scenario["items"], raw_scores):
            raw_f = float(raw)
            rows.append(
                {
                    "task": scenario["task"],
                    "id": ident,
                    "text": text,
                    "expected": expected,
                    "local_raw": raw_f,
                    "local_sigmoid": sigmoid(raw_f),
                    "jev_noul": jev_scores[ident],
                }
            )

    expected = [r["expected"] for r in rows]
    local = [r["local_sigmoid"] for r in rows]
    jev = [r["jev_noul"] for r in rows]
    local_pred = [int(v >= 0.5) for v in local]
    jev_pred = [int(v >= 0.6) for v in jev]
    calibrated_threshold, calibrated_accuracy = best_threshold(local, expected)
    calibrated_pred = [int(v >= calibrated_threshold) for v in local]
    summary = {
        "model": MODEL,
        "device": torch.cuda.get_device_name(0),
        "cases": len(rows),
        "load_seconds": load_s,
        "local_scenario_seconds": local_times,
        "jev_scenario_seconds": jev_times,
        "local_accuracy_at_0.5": sum(a == b for a, b in zip(expected, local_pred)) / len(rows),
        "local_calibrated_threshold": calibrated_threshold,
        "local_calibrated_accuracy": calibrated_accuracy,
        "local_auc": auc(local, expected),
        "local_min_relevant": min(score for score, label in zip(local, expected) if label == 1),
        "local_max_irrelevant": max(score for score, label in zip(local, expected) if label == 0),
        "jev_accuracy_at_0.6": sum(a == b for a, b in zip(expected, jev_pred)) / len(rows),
        "local_vs_jev_spearman": spearman(local, jev),
        "local_vs_jev_decision_agreement_at_calibrated_threshold": sum(
            a == b for a, b in zip(calibrated_pred, jev_pred)
        ) / len(rows),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nCASES")
    for row in rows:
        print(
            f"{row['id']:22s} expected={row['expected']} "
            f"local={row['local_sigmoid']:.4f} raw={row['local_raw']:.3f} "
            f"jev={row['jev_noul']:.4f}"
        )
    print(f"\nresults={OUT}")


if __name__ == "__main__":
    main()
