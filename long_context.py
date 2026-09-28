from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

MODEL = "Qwen/Qwen3-Reranker-0.6B"
TASK = "Fix the Python API returning HTTP 500 when email is null; keep exact evidence needed for the fix."
INSTRUCTION = (
    "Decide whether the document must remain available to correctly continue the current task. "
    "Keep current errors, source evidence, constraints, decisions, and verification. Discard unrelated output."
)
FACT = "CRITICAL CURRENT ERROR: normalize_email calls email.lower() and crashes because email is None."
FILLER_LINE = "Historical unrelated package build completed successfully with no API errors. "
FILLER = FILLER_LINE * 1800


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def head_tail(text: str, chars: int = 6000) -> str:
    if len(text) <= chars * 2:
        return text
    return text[:chars] + "\n[... middle omitted ...]\n" + text[-chars:]


def main() -> None:
    model = CrossEncoder(
        MODEL,
        device="cuda",
        model_kwargs={"dtype": torch.bfloat16},
        max_length=4096,
    )
    docs = {
        "short_relevant": FACT,
        "long_fact_at_start": FACT + "\n" + FILLER,
        "long_fact_at_end": FILLER + "\n" + FACT,
        "long_irrelevant": FILLER,
        "head_tail_fact_at_end": head_tail(FILLER + "\n" + FACT),
    }
    full_tokens = {
        name: len(model.tokenizer(TASK, doc, truncation=False)["input_ids"])
        for name, doc in docs.items()
    }
    started = time.perf_counter()
    raw = model.predict(
        [(TASK, doc) for doc in docs.values()],
        prompt=INSTRUCTION,
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - started
    rows = []
    for (name, doc), score in zip(docs.items(), raw):
        rows.append({
            "name": name,
            "characters": len(doc),
            "full_tokens": full_tokens[name],
            "raw": float(score),
            "sigmoid": sigmoid(float(score)),
        })
    result = {"elapsed_seconds": elapsed, "rows": rows}
    path = Path(__file__).with_name("long-context-results.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"results={path}")


if __name__ == "__main__":
    main()
