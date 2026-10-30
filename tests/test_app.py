"""Headless UI smoke tests using Streamlit's AppTest (offline demo mode; no keys, no network)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import evaluate  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


def _app(monkeypatch):
    monkeypatch.setattr("src.config.settings.GROQ_API_KEY", "")
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    return at


def _run_sample(at, case):
    at.selectbox(key="sample").select(case.title)
    at.run()
    at.button[0].click()
    at.run()
    return at


def test_app_loads_with_disclaimer_and_defaults_to_demo(monkeypatch):
    at = _app(monkeypatch)
    assert not at.exception
    assert any("not legal advice" in c.value for c in at.caption)
    assert any("ONE policy clause" in i.value for i in at.info)
    assert at.sidebar.radio[0].value == "Offline scripted demo"


def test_demo_audit_shows_status_and_severity_separately_and_real_steps(monkeypatch):
    at = _run_sample(_app(monkeypatch), evaluate.CASES[0])
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Compliance status"] == "NON COMPLIANT" and m["Risk severity"] == "HIGH"
    texts = " ".join(str(getattr(e, "value", "")) for e in at.markdown)
    assert "Completed step: **retrieve**" in texts and "Completed step: **verify**" in texts
    assert any("SCRIPTED DEMO" in w.value for w in at.warning)
    assert at.expander and any("syn-r1" in e.label for e in at.expander)


def test_verifier_probe_shows_failure_warning_and_hides_quote(monkeypatch):
    case = next(c for c in evaluate.CASES if c.id == "fabricated_citation")
    at = _run_sample(_app(monkeypatch), case)
    assert not at.exception
    assert any("Verification failed" in w.value for w in at.warning)
    assert not any("365 days" in str(c.value) for c in at.code)


def test_empty_input_and_unsupported_text_in_demo(monkeypatch):
    at = _app(monkeypatch)
    at.button[0].click(); at.run()
    assert any("Enter a clause ID" in w.value for w in at.warning)
    at.text_area(key="clause_text").set_value("Some custom clause."); at.run()
    at.button[0].click(); at.run()
    assert any("only supports the built-in sample" in w.value for w in at.warning)


def test_live_mode_without_key_shows_safe_error(monkeypatch):
    at = _app(monkeypatch)
    at.sidebar.radio[0].set_value("Live model"); at.run()
    at.text_area(key="clause_text").set_value("A clause."); at.run()
    at.button[0].click(); at.run()
    assert any("API key is not configured" in e.value for e in at.error)
    assert not any("gsk_" in e.value or "Traceback" in e.value for e in at.error)