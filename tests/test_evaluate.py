"""Offline evaluation tests (scripted fake LLM; verifies wiring, not model quality)."""
import json

import pytest

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


def test_live_mode_skips_the_scripted_only_verifier_probe_and_says_so():
    assert [c.id for c in evaluate.cases_for_mode(False)] == [c.id for c in CASES] and len(CASES) == 6
    live_ids = [c.id for c in evaluate.cases_for_mode(True)]
    assert "fabricated_citation" not in live_ids and len(live_ids) == 5
    assert run_evaluation()["skipped"] == []


def test_live_path_wiring_with_a_fake_model_never_touches_the_network(monkeypatch):
    from src.schemas import DocumentGrade
    from tests.fakes import FakeLLM
    fake = FakeLLM({"DocumentGrade": [DocumentGrade(is_relevant=False, confidence_score=0.5, rationale="r")]},
                   texts=["retained days", "access logs"])
    monkeypatch.setattr(evaluate, "make_default_llm", lambda: fake)
    s = run_evaluation(live=True)  # web fallback stays disabled (no --web)
    assert s["mode"] == "LIVE" and s["total"] == 5 and s["skipped"] == ["fabricated_citation"] and s["errors"] == []
    assert s["accuracy"] == pytest.approx(1 / 5)  # an always-irrelevant grader is right only for the out-of-scope case
    text = format_report(s)
    assert "LIVE RUN" in text and "skipped in live mode" in text and "fabricated_citation" in text


def test_case_filter_and_verbose_details_offline():
    s = run_evaluation(only=["out_of_scope"])
    assert s["total"] == 1 and s["skipped"] == []
    assert s["cases"][0]["path"][:2] == ["retrieve", "grade"] and "web_fallback" in s["cases"][0]["path"]
    text = format_report(s, verbose=True)
    assert "[out_of_scope] path: retrieve > grade" in text and "last grade rationale" in text and "gap analysis" in text
    assert "path:" not in format_report(s)  # details only with --verbose


def test_classified_case_records_path_and_gap_analysis():
    row = run_evaluation(only=["retention"])["cases"][0]
    assert row["path"] == ["retrieve", "grade", "audit", "verify"] and "30 days" in row["gap_analysis"]


def test_cli_case_option_unknown_case_and_live_probe_refusal(capsys, monkeypatch):
    assert main(["--case", "retention"]) == 0
    out = capsys.readouterr().out
    assert "retention" in out and "location" not in out
    with pytest.raises(SystemExit) as e:
        main(["--case", "nope"])
    assert e.value.code == 2
    monkeypatch.setattr(evaluate, "make_default_llm", lambda: (_ for _ in ()).throw(AssertionError("must not be reached")))
    assert main(["--live", "--case", "fabricated_citation"]) == 2
    assert "scripted-only" in capsys.readouterr().out