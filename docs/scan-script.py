#!/usr/bin/env python3
"""Anonymize staging files for public release: /home/Petsku → generic, Petsku → generic,
/scratch paths → generic. Preserves everything else byte-for-byte. Reports all replacements."""
import re
from pathlib import Path

STAGE = Path("/tmp/jev-spike-public-staging")

# Vain 6 skriptiä + service-unit sisältävät polkuja/nimiä (jsonl:ssä /scratch on
# logihakemistopolku — myös se anonymisoidaan)
TARGETS = [
    "multi_session_soak.py", "test_multi_session_soak.py",
    "diagnose_semif_value_error.py", "diagnose_soak_failure.py",
    "diagnose_soak_failure_sites.py", "probe_semif_4b.py",
    "download_semif_4b.py", "download_semif_4b.log",
    "hermes-jev-local-primary.service.staging",
    "authored144-local.jsonl",
]

REPLACEMENTS = [
    # (regex, korvaus, kuvaus)
    (re.compile(r"/home/Petsku/\.hermes"), "HERMES_HOME", "hermes-home → HERMES_HOME placeholder"),
    (re.compile(r"/home/Petsku"), "/home/<user>", "kotihakemisto → <user>"),
    (re.compile(r"/scratch/build/jev-local-reranker-spike"), "<spike_dir>", "spike-polku → placeholder"),
    (re.compile(r"/scratch"), "/<scratch>", "scratch-polku → placeholder"),
    (re.compile(r"\bPetsku01\b"), "<user>", "GitHub-tunnus → placeholder"),
    (re.compile(r"\bPetsku\b"), "<user>", "käyttäjänimi → placeholder"),
    (re.compile(r"\bZiggurat\b"), "<hostname>", "hostname → placeholder"),
]

total = {}
for name in TARGETS:
    p = STAGE / name
    if not p.exists():
        continue
    if name == "authored144-local.jsonl":
        # jsonl: korvaus riveittäin ettei isoa tiedostoa ladata kokonaan muistiin turhaan
        out = []
        n_hits = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                orig = line
                for pat, rep, _ in REPLACEMENTS:
                    line = pat.sub(repl=rep, string=line)
                if line != orig:
                    n_hits += 1
                out.append(line)
        p.write_text("".join(out), encoding="utf-8")
        if n_hits:
            total[name] = f"{n_hits} riviä korjattu"
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    changed = []
    for pat, rep, desc in REPLACEMENTS:
        n = len(pat.findall(text))
        if n:
            text = pat.sub(repl=rep, string=text)
            changed.append(f"{desc}: {n}")
    if changed:
        p.write_text(text, encoding="utf-8")
        total[name] = "; ".join(changed)

print("=== ANONYMOUSOINTI ===")
for k, v in sorted(total.items()):
    print(f"{k}: {v}")
if not total:
    print("ei korjattavaa")

# re-scan: pitäisi olla puhtaa
import sys
sys.path.insert(0, "/tmp")
PATTERNS_CHECK = {
    "kotipolku": r"/home/Petsku", "Petsku": r"\bPetsku(01)?\b",
    "scratch": r"/scratch", "hostname": r"Ziggurat",
}
print("\n=== UUDELLEENSKANNI ===")
clean = True
for name in TARGETS:
    p = STAGE / name
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    for label, pat in PATTERNS_CHECK.items():
        n = len(re.findall(pat, text))
        if n:
            clean = False
            print(f"JÄLJELLÄ {label}: {name} ×{n}")
if clean:
    print("PUHTAA — kaikki anonymisoitu")