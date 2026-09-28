# Local Reranker for Compaction — Feasibility Spike

**Date:** 2026-09-24 (runs) · **Host:** consumer laptop (Ryzen AI 9 HX 370, 93 GiB RAM, RTX 5070 Laptop 12 GB)
**Question:** can a small local cross-encoder replace an external scoring API for context-compaction evidence ranking?

Public research artifact from a weekend spike. The production system's compaction engine
("TypeSafe JEV") calls an external scoring API; this spike measured whether a locally-hosted
reranker could take over that role: faster, free, and with **zero conversation egress**.

## TL;DR

- **Feasibility: YES.** The model runs reliably on a 12 GB laptop GPU; warm scoring is
  ~15× faster than the network API call (0.04 s vs 0.64–0.76 s per case); VRAM ~1.3 GB.
- **Drop-in replacement: NO (yet).** Raw sigmoid scores are not calibrated like the API's
  (threshold 0.5 is unusable — calibrated threshold was 0.000185); holdout accuracy 80% vs
  API's 93%; naive 4k truncation silently destroys critical evidence at the end of long
  tool results.
- **Error profile is the interesting part:** every local error on the holdout was a
  *conservative false positive* (kept too much). The external API made one *dangerous
  false negative* (dropped a current failing assertion). For compaction, failing
  "keep too much" is the safe direction.

## Method

Three-stage evaluation, isolated venv, **no production config or dependencies touched**:

1. **Calibration** (`benchmark.py`) — 17 synthetic cases (backup debugging, Python API
   debugging, display troubleshooting; Finnish + English; constraints, errors, verification,
   unrelated output). Both engines scored every case.
2. **Holdout** (`holdout.py`) — 15 fresh cases (port conflict, memory dedup, SMTP failure),
   including superseded evidence and a tool-result prompt injection.
3. **Long-result probe** (`long_context.py`) — same critical fact placed at start vs end of a
   19,843-token tool result.

### Calibration (17/17 both engines after threshold calibration)

| Metric | Local reranker | External API |
|---|---|---|
| Accuracy (naive 0.5 / 0.6 threshold) | 52.9% | 100% |
| Accuracy (calibrated threshold) | **100%** | 100% |
| Spearman (local vs API, warm) | 0.74 | — |
| Scoring latency | 0.039–0.041 s | 0.64–0.76 s |
| Cold model load | 28.2 s | n/a |
| Cached model load | 4.1 s | n/a |
| Peak VRAM | ~1.35 GB | n/a |

### Holdout (15 unseen cases)

| Metric | Local | External API |
|---|---|---|
| Accuracy | 80% (12/15) | 93.3% (14/15) |
| Decision agreement | 73.3% | — |
| KEEP precision / recall | 75% / **100%** | — |
| Confusion (TP/FP/FN/TN) | 9/3/**0**/3 | — |
| Dangerous false drops | **0** | **1** (dropped a current failing assertion) |

### Long tool result (19,843 tokens, `max_length=4096`)

| Placement | Raw sigmoid | Head+tail transform |
|---|---|---|
| Short relevant | 0.3775 | — |
| Long, fact at **start** | 0.6514 | — |
| Long, fact at **end** | **0.0260** (evidence destroyed) | **0.9579** |
| Long, irrelevant | 0.0260 | — |

## Second model, second round (2026-09-26/27): SemIf / Qwen3.5-4B

A follow-up probe tried a different model class (`SemIf` 4B, option-selection readout instead
of a cross-encoder). 144 authored cases across three decision families:

| Family | Accuracy | Balanced accuracy |
|---|---|---|
| Evidence interpretation | 85.4% | 87.1% |
| Rule application | 87.5% | 87.9% |
| Candidate selection | 68.8% | 69.0% |
| **Mean** | — | **81.3%** |

A multi-session soak test hit a crash late on day 2 (diagnosis scripts included;
`accelerate==1.12.0` dependency issue was one blocker). This part is **unfinished** —
included as honest work-in-progress, not a result.

## What a real build needs (from the spike's verdict)

1. Persistent model worker / lazy singleton (no per-turn reload; cold load 28 s is a killer)
2. Head+tail or chunked scoring, aggregated with max — any critical segment keeps the result
3. Separate call vs result scores, preserving the existing decision table
4. **Fail-open to KEEP** on model errors, OOM, unknown scores, or uncertain calibration
5. Drop threshold learned on a much larger shadow corpus — never reuse a spike constant
6. External API remains decision-maker during shadow evaluation; collect metrics, not raw text
7. Promote only after real-session evaluation: zero critical false drops + acceptable yield

## Repo layout

```
benchmark.py                  calibration harness (17 cases, both engines)
holdout.py                    15-case holdout
long_context.py               19.8k-token placement probe
results*.json, holdout-results.json, long-context-results.json
authored144-local-report.json SemIf 144-case report (per-family accuracy)
authored144-local.jsonl       per-case timing/probability log (prompt_sha256 only, no text)
multi_session_soak.py + test  soak harness (WIP, crash late in day 2)
diagnose_*.py                 crash-site diagnostics for the soak failure
probe_semif_4b.py             SemIf load/value-shape probe
download_semif_4b.py          pinned-commit model fetch
release/                      semif_phase1 0.1.0 wheel + requirements.lock + provenance
hermes-jev-local-primary.service.staging  systemd unit sketch (staging-only, not deployed)
README.md                     this file
```

Models are re-downloadable (pinned commits); nothing in this repo contains conversation
content, credentials, or private paths (checked by an automated scan — paths/usernames were
anonymized; see `docs/scan-report.md`).

## Why publish a "partial"?

Because the negative results are the payload: **the calibration cliff (100% → 80% after a
17-case calibration), the tail-evidence loss under naive truncation, and the
conservative-error asymmetry** are exactly the failure modes that a "yes it ran" blog post
would hide. Local-first compaction ranking is viable; it is not trivial; here is the map.