"""Tests for UI view-model helpers (no Streamlit needed)."""
from langgraph.errors import GraphRecursionError

from src.llm_clients import LLMOutputError, MissingCredentialError
from src.retriever import EmbeddingModelError, RetrievalIndexError
from src.report import MSG_INSUFFICIENT, MSG_NOT_VERIFIED, MSG_WEB, build_report_view, safe_error_message
from src.schemas import (AuditFinding, ComplianceStatus, EvidenceChunk, RiskSeverity, SourceType)

LOC = EvidenceChunk(id="loc-1", document_name="RULEBOOK", page_number=4, section_title="Retention", text="Keep 90 days.")
WEB = EvidenceChunk(id="web-1", document_name="Reg Site", page_number=None, section_title="Web Search Fallback",
                    text="Keep 90 days.", source_type=SourceType.WEB, source_url="https://example.gov/r")


def f(**kw):
    b = dict(policy_clause_id="P", policy_text_excerpt="e", status=ComplianceStatus.NON_COMPLIANT,
             severity=RiskSeverity.HIGH, statutory_clause="Rule 1", statutory_verbatim_quote="Keep 90 days.",
             evidence_ids=["loc-1"], gap_analysis="gap", remediation_action="fix")
    b.update(kw)
    return AuditFinding(**b)


def state(report, grounded=True, docs=(LOC,), **kw):
    s = {"audit_report": report, "is_grounded": grounded, "retrieved_docs": list(docs), "retry_count": 1,
         "web_fallback_attempted": False, "web_fallback_used": False, "error_message": None}
    s.update(kw)
    return s


def test_status_and_severity_are_separate_fields_and_evidence_provenance():
    v = build_report_view(state(f()))
    assert v["status"] == "NON_COMPLIANT" and v["severity"] == "HIGH" and v["warnings"] == []
    e = v["evidence"][0]
    assert (e["id"], e["title"], e["section"], e["page"], e["url"]) == ("loc-1", "RULEBOOK", "Retention", 4, None)
    assert v["quote"] == "Keep 90 days." and v["statutory_clause"] == "Rule 1"


def test_web_evidence_shows_url_label_and_warning():
    v = build_report_view(state(f(evidence_ids=["web-1"]), docs=(WEB,), web_fallback_attempted=True, web_fallback_used=True))
    e = v["evidence"][0]
    assert e["url"] == "https://example.gov/r" and "WEB" in e["source"] and e["page"] == "n/a"
    assert MSG_WEB in v["warnings"]


def test_unverified_quote_is_never_shown_and_failure_is_flagged():
    v = build_report_view(state(f(), grounded=False, error_message="Citation verification failed: x"))
    assert v["quote"] is None and v["statutory_clause"] is None
    assert MSG_NOT_VERIFIED in v["warnings"] and v["verification_note"].startswith("Citation verification failed")


def test_insufficient_evidence_warning_only_when_grounded_vacuously():
    ins = f(status=ComplianceStatus.INSUFFICIENT_EVIDENCE, severity=RiskSeverity.INFORMATIONAL, statutory_clause=None,
            statutory_verbatim_quote=None, evidence_ids=[])
    assert MSG_INSUFFICIENT in build_report_view(state(ins))["warnings"]
    assert MSG_INSUFFICIENT not in build_report_view(state(ins, grounded=False))["warnings"]


def test_web_attempted_but_empty_is_reported():
    ins = f(status=ComplianceStatus.INSUFFICIENT_EVIDENCE, severity=RiskSeverity.INFORMATIONAL, statutory_clause=None,
            statutory_verbatim_quote=None, evidence_ids=[])
    v = build_report_view(state(ins, web_fallback_attempted=True))
    assert any("web search was attempted" in w for w in v["warnings"])


def test_missing_result():
    v = build_report_view(state(None))
    assert v["has_result"] is False and v["warnings"]


def test_safe_errors_leak_nothing():
    secret = RuntimeError("Traceback /home/x/.env GROQ_API_KEY=gsk_123")
    for exc in (secret, MissingCredentialError("GROQ_API_KEY=gsk_123"), LLMOutputError("raw"), GraphRecursionError("x")):
        msg = safe_error_message(exc)
        assert "gsk_" not in msg and "Traceback" not in msg and "/home" not in msg
    assert "API key" in safe_error_message(MissingCredentialError())


def test_index_and_embedding_errors_have_actionable_safe_messages():
    idx_msg = safe_error_message(RetrievalIndexError("No index found in C:\\secret\\path. Index documents first"))
    assert "index" in idx_msg and "ingestion" in idx_msg and "secret" not in idx_msg and "C:" not in idx_msg
    emb_msg = safe_error_message(EmbeddingModelError("HF download failed at https://x"))
    assert "embedding model" in emb_msg and "internet" in emb_msg and "https" not in emb_msg


def _groq_status_error(cls_name, status):
    import groq
    import httpx
    resp = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/x"),
                          json={"error": {"message": "SECRET-DETAIL gsk_abc"}})
    return getattr(groq, cls_name)("SECRET-DETAIL gsk_abc", response=resp, body=None)


def test_groq_failures_get_specific_safe_messages():
    import groq
    import httpx
    cases = [("AuthenticationError", 401, "rejected the API key"), ("NotFoundError", 404, "GROQ_MODEL_NAME"),
             ("RateLimitError", 429, "rate limit"), ("BadRequestError", 400, "JSON output"),
             ("InternalServerError", 500, "temporary problem")]
    for cls, status, expected in cases:
        msg = safe_error_message(_groq_status_error(cls, status))
        assert expected in msg and "SECRET" not in msg and "gsk_" not in msg and "Traceback" not in msg
    conn = groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))
    assert "Could not reach the model provider" in safe_error_message(conn)


def test_unrecognised_errors_still_get_the_generic_message():
    msg = safe_error_message(ValueError("whatever gsk_x"))
    assert "Something went wrong" in msg and "gsk_" not in msg