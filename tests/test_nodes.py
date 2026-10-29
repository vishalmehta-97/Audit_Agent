"""Node tests. Currently covers retriever_node; grader/rewriter/web_fallback/auditor tests come with those nodes."""
import pytest

import src.nodes as nodes
from src.nodes import retriever_node
from src.schemas import (AuditFinding, ComplianceStatus, DocumentGrade, EvidenceChunk, RiskSeverity)
from tests.mocks.mock_retriever import MockRegulatoryRetriever

CH = [EvidenceChunk(id="a", document_name="D", text="logs retention days"),
      EvidenceChunk(id="b", document_name="D", text="retention rules")]


def stale_state(query="logs retention"):
    f = AuditFinding(policy_clause_id="P", policy_text_excerpt="e", status=ComplianceStatus.COMPLIANT,
                     severity=RiskSeverity.LOW, gap_analysis="g", remediation_action="r")
    return {"policy_clause_id": "P", "policy_text": "T", "current_query": query, "retrieved_docs": [],
            "document_grade": DocumentGrade(is_relevant=True, confidence_score=1.0, rationale="old"),
            "retry_count": 1, "web_fallback_used": True, "web_fallback_attempted": True,
            "is_grounded": True, "audit_report": f, "error_message": "old error"}


def test_retriever_node_returns_chunks_and_uses_settings_top_k(monkeypatch):
    monkeypatch.setattr(nodes.settings, "RETRIEVAL_TOP_K", 1)
    r = MockRegulatoryRetriever(CH)
    out = retriever_node(stale_state(), r)
    assert len(out["retrieved_docs"]) == 1 and isinstance(out["retrieved_docs"][0], EvidenceChunk)
    assert r.calls[-1]["top_k"] == 1


def test_stale_fields_reset_but_essential_fields_untouched():
    out = retriever_node(stale_state(), MockRegulatoryRetriever(CH))
    assert out["document_grade"] is None and out["is_grounded"] is False
    assert out["audit_report"] is None and out["error_message"] is None
    for key in ("policy_clause_id", "policy_text", "retry_count", "web_fallback_used", "web_fallback_attempted"):
        assert key not in out


@pytest.mark.parametrize("q", ["", "   ", None])
def test_blank_query_skips_retriever_with_clear_error(q):
    r = MockRegulatoryRetriever(CH)
    out = retriever_node(stale_state(q), r)
    assert r.calls == [] and out["retrieved_docs"] == []
    assert "empty" in out["error_message"].lower()


def test_no_match_returns_empty_without_error():
    out = retriever_node(stale_state("zzz qqq"), MockRegulatoryRetriever(CH))
    assert out["retrieved_docs"] == [] and out["error_message"] is None


def test_unexpected_retriever_exception_is_not_hidden():
    class Boom(MockRegulatoryRetriever):
        def retrieve(self, query, top_k=4):
            raise RuntimeError("index corrupted")
    with pytest.raises(RuntimeError):
        retriever_node(stale_state(), Boom())