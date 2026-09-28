import json
import time
import torch

from plugins.context_engine.jev_compaction.local_primary_service import SemIfDecisionModel

snapshot = "<spike_dir>/hf-cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
started = time.monotonic()
model = SemIfDecisionModel(
    model_id=snapshot,
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    device="cuda",
    dtype="bfloat16",
    max_input_tokens=6000,
)
loaded_s = time.monotonic() - started
state = (
    "Goal: deploy API-free local Jev safely. "
    "The final acceptance criterion requires preserving the latest test evidence. "
    "An earlier temporary download estimate was superseded after the model download completed. "
    + "Context detail for capacity testing. " * 800
)
questions = {
    "keep_latest_test": {
        "type": "noul",
        "instructions": "Does the latest passing test evidence remain necessary for the goal?",
    },
    "keep_old_estimate": {
        "type": "noul",
        "instructions": "Must the superseded temporary download estimate remain in context?",
    },
}
for index in range(30):
    questions[f"capacity_{index:02d}"] = {
        "type": "noul",
        "instructions": f"Does context detail number {index} remain necessary for the final goal?",
    }
score_started = time.monotonic()
answers = model.decide(state, questions)
score_s = time.monotonic() - score_started
important = answers["keep_latest_test"]["noul"]
stale = answers["keep_old_estimate"]["noul"]
result = {
    "loaded_s": round(loaded_s, 3),
    "score_s": round(score_s, 3),
    "question_count": len(answers),
    "important_score": important,
    "stale_score": stale,
    "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
}
print(json.dumps(result, sort_keys=True), flush=True)
if len(answers) != 32 or important < 0.20 or stale >= 0.10:
    raise SystemExit(2)
