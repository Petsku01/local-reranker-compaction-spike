import json
import socket
from pathlib import Path

import yaml

from multi_session_soak import load_candidates, select_sessions, short_hash
from plugins.context_engine import load_context_engine

TARGET = "3bea3dced5f648e2"
candidates, _ = load_candidates(Path("/tmp/jev-soak-sessions.jsonl"))
selected = select_sessions(candidates, 96)
config = yaml.safe_load(Path("HERMES_HOME/config.yaml").read_text())
settings = dict(config["context_engine"]["settings"])
base = load_context_engine("jev_compaction")
engine = base.__class__(settings)
for session in selected:
    if short_hash(str(session["id"])) != TARGET:
        continue
    messages = session.get("messages") or []
    sites = {site["tid"]: site for site in engine._call_sites(messages)}
    eligible = [
        (tool_id, metadata, sites[tool_id])
        for tool_id, metadata in session["_positives"].items()
        if tool_id in sites
    ]
    eligible.sort(key=lambda item: (-item[1]["strongest_fingerprint_chars"], item[1]["message_index"]))
    eligible = eligible[:4]
    state, state_tokens = engine._fit_state(messages)
    request = {
        "type": "local_decide",
        "state": state,
        "questions": engine._questions([item[2] for item in eligible]),
    }
    payload = json.dumps(request, separators=(",", ":")).encode() + b"\n"
    from plugins.context_engine.jev_compaction.local_primary_service import validate_decision_request
    local_validation = "ok"
    try:
        validate_decision_request(request)
    except Exception as exc:
        local_validation = f"{type(exc).__name__}:{exc}"
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(45)
    client.connect("/run/user/1000/hermes-jev-local.sock")
    client.sendall(payload)
    response = client.recv(4096)
    client.close()
    decoded = json.loads(response)
    print(json.dumps({
        "session": TARGET,
        "state_tokens_est": state_tokens,
        "state_chars": len(state),
        "sites": len(eligible),
        "questions": len(request["questions"]),
        "payload_bytes": len(payload),
        "local_validation": local_validation,
        "max_question_id_chars": max(map(len, request["questions"])),
        "max_instruction_chars": max(len(item["instructions"]) for item in request["questions"].values()),
        "response_ok": decoded.get("ok"),
        "server_error_type": decoded.get("error"),
    }, sort_keys=True))
    break
else:
    raise SystemExit("target not found")
