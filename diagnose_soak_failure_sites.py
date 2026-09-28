import json
import socket
from pathlib import Path

import yaml

from multi_session_soak import load_candidates, select_sessions, short_hash
from plugins.context_engine import load_context_engine

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
    for ordinal, (tool_id, metadata, site) in enumerate(eligible[:4]):
        questions = engine._questions([site])
        request = {"type": "local_decide", "state": state, "questions": questions}
        payload = json.dumps(request, separators=(",", ":")).encode() + b"\n"
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(45)
        client.connect("/run/user/1000/hermes-jev-local.sock")
        client.sendall(payload)
        decoded = json.loads(client.recv(4096))
        client.close()
        print(json.dumps({
            "ordinal": ordinal,
            "tool_hash": short_hash(tool_id),
            "result_chars": metadata["result_chars"],
            "question_chars": [len(item["instructions"]) for item in questions.values()],
            "state_tokens_est": state_tokens,
            "ok": decoded.get("ok"),
            "error": decoded.get("error"),
        }, sort_keys=True), flush=True)
    break
else:
    raise SystemExit("target not found")
