"""Offline evaluation tests (scripted fake LLM; verifies wiring, not model quality)."""
import json

import evaluate
from evaluate import CASES, format_report, main, run_evaluation


def test_offline_evaluation_numbers_and_verifier_probe():
    s = run_evaluation()
    assert s["mode"] == "OFFLINE_SCRIPTED" and s["total"] == len(CASES) == 6 and s["errors"] == []
    assert all(r["expected"] == r["actual"] for r in s["cases"]) and s["accuracy"] == 1.0
    probe = next(r for r in s["cases"] if r["id"] == "fabricated_citation")
    assert probe["grounded"] is False and probe["actual"] == "INSUFFICIENT_EVIDENCE"  # verifier fail-closed
    assert round(s["grounding_rate"], 4) == round(5 / 6, 4) and s["insufficient_count"] == 2
    assert s["avg_latency_ms"] >= 0


def test_five_required_case_kinds_present():
    ids = {c.id for c in CASES}
    assert {"retention", "location", "compliant", "ambiguous", "out_of_scope"} <= ids


def test_report_is_honestly_labeled():
    text = format_report(run_evaluation())
    assert "OFFLINE SCRIPTED" in text and "by construction" in text and "legal accuracy" in text


def test_live_without_key_exits_cleanly_without_running(monkeypatch, capsys):
    monkeypatch.setattr("src.llm_clients.settings.GROQ_API_KEY", "")
    assert main(["--live"]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().out


def test_json_output(tmp_path, capsys):
    p = tmp_path / "r.json"
    assert main(["--json", str(p)]) == 0
    assert json.loads(p.read_text())["total"] == 6