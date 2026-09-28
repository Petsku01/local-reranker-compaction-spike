import json
from pathlib import Path
import yaml

from multi_session_soak import load_candidates, select_sessions, short_hash
from plugins.context_engine import load_context_engine
from plugins.context_engine.jev_compaction.local_primary_service import SemIfDecisionModel

TARGET = "611729d0851fcd12"
candidates, _ = load_candidates(Path("/tmp/jev-soak-sessions.jsonl"))
selected = select_sessions(candidates, 96)
settings = yaml.safe_load(Path("HERMES_HOME/config.yaml").read_text())["context_engine"]["settings"]
base = load_context_engine("jev_compaction")
engine = base.__class__(settings)
for session in selected:
    if short_hash(str(session["id"])) != TARGET:
        continue
    messages = session.get("messages") or []
    state, state_tokens = engine._fit_state(messages)
    sites = {site["tid"]: site for site in engine._call_sites(messages)}
    eligible = [(tid, meta, sites[tid]) for tid, meta in session["_positives"].items() if tid in sites]
    eligible.sort(key=lambda item: (-item[1]["strongest_fingerprint_chars"], item[1]["message_index"]))
    tool_id, metadata, site = eligible[0]
    questions = engine._questions([site])
    state = state + "\n[END OF EVIDENCE]\n"
    model = SemIfDecisionModel(
        model_id="<spike_dir>/hf-cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        device="cuda",
        dtype="bfloat16",
        max_input_tokens=6000,
        decision_batch_size=4,
    )
    answers = model.decide(state, questions)
    print(json.dumps({"ok": True, "answers": len(answers), "state_tokens_est": state_tokens, "tool_hash": short_hash(tool_id)}, sort_keys=True))
    raise SystemExit(0)
raise SystemExit("target not found")
