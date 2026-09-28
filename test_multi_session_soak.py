import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("multi_session_soak.py")
spec = importlib.util.spec_from_file_location("multi_session_soak", MODULE_PATH)
soak = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(soak)


def test_positive_ids_requires_later_narrative_reference():
    fingerprint = "deployment-proof-64373124f8768477"
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {"name": "probe", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": f"result {fingerprint} " + "x" * 100},
        {"role": "assistant", "content": f"The final report relies on {fingerprint}."},
        {"role": "assistant", "tool_calls": [{"id": "call-2", "function": {"name": "probe", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-2", "content": "unreferenced result " + "y" * 100},
    ]

    positives = soak.positive_ids(messages)

    assert set(positives) == {"call-1"}
    assert positives["call-1"]["result_chars"] > 100


def test_dry_run_quality_makes_no_accuracy_claim():
    rows = [{"session": "abc", "source": "cli", "tool": "def"}]

    quality = soak.build_quality(rows, dry_run=True, keep_threshold=0.01)

    assert quality["known_positive_keep_recall"] is None
    assert quality["false_truncates"] is None
    assert quality["zero_error_one_sided_95_upper_rate"] is None


def test_live_quality_counts_false_truncates_and_zero_error_bound():
    safe = [
        {"result_score": 0.9, "production_action": "keep"},
        {"result_score": 0.2, "production_action": "keep"},
    ]
    risky = safe + [{"result_score": 0.001, "production_action": "truncate"}]

    safe_quality = soak.build_quality(safe, dry_run=False, keep_threshold=0.01)
    risky_quality = soak.build_quality(risky, dry_run=False, keep_threshold=0.01)

    assert safe_quality["known_positive_keep_recall"] == 1.0
    assert safe_quality["false_truncates"] == 0
    assert safe_quality["zero_error_one_sided_95_upper_rate"] == pytest.approx(1 - 0.05 ** 0.5)
    assert risky_quality["known_positive_keep_recall"] == pytest.approx(2 / 3)
    assert risky_quality["false_truncates"] == 1
    assert risky_quality["zero_error_one_sided_95_upper_rate"] is None


def test_spread_pick_keeps_temporal_endpoints():
    rows = [{"started_at": value} for value in range(10)]

    selected = soak.spread_pick(rows, 4)

    assert selected[0]["started_at"] == 0
    assert selected[-1]["started_at"] == 9
    assert len(selected) == 4
