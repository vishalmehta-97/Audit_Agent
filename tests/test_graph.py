"""Offline graph tests: fake LLM-ish nodes + MockRegulatoryRetriever. Proves bounded termination."""
import pytest

import src.graph as graph_mod
from src.graph import create_audit_graph, make_initial_state, route_after_grade, run_audit
from src.nodes import make_insufficient_finding, verifier_node
from src.schemas import (AuditFinding, AuditGraphState, ComplianceStatus, DocumentGrade, EvidenceChunk,
                         RiskSeverity, SourceType)
from tests.mocks.mock_retriever import MockRegulatoryRetriever

LOCAL = EvidenceChunk(id="loc-1", document_name="SYNTHETIC RULEBOOK", page_number=1, section_title="Retention",
                      text="Records must be retained for a minimum of 90 days.")
WEB = EvidenceChunk(id="web-1", document_name="Example Regulator", page_number=None,
                    section_title="Web Search Fallback", text="Providers must keep logs for 90 days.",
                    source_type=SourceType.WEB, source_url="https://example.org/rule")
CLAUSE = "We keep logs for retention of 30 days."


class TracingRetriever(MockRegulatoryRetriever):
    def __init__(self, chunks, order):
        super().__init__(chunks)
        self.order = order

    def retrieve(self, query, top_k=4):
        self.order.append("retrieve")
        return super().retrieve(query, top_k)


def make_grader(order, relevant_calls=(), log=None):
    n = {"i": 0}

    def grader(state):
        n["i"] += 1
        order.append("grade")
        if log is not None:
            log.append((state["retry_count"], len(state["retrieved_docs"])))
        return {"document_grade": DocumentGrade(is_relevant=n["i"] in relevant_calls,
                                                confidence_score=0.9, rationale="fake")}
    return grader


def make_rewriter(order):
    def rewriter(state):
        order.append("rewrite")
        n = state["retry_count"] + 1
        return {"retry_count": n, "current_query": f"{state['current_query']} retention{n}",
                "document_grade": None, "audit_report": None, "is_grounded": False, "error_message": None}
    return rewriter


def make_fallback(order, results=()):
    def fallback(state):
        order.append("web_fallback")
        return {"web_fallback_attempted": True, "web_fallback_used": bool(results), "document_grade": None,
                "retrieved_docs": list(state["retrieved_docs"]) + list(results)}
    return fallback


def make_auditor(order):
    def auditor(state):
        order.append("audit")
        grade, docs = state["document_grade"], state["retrieved_docs"]
        if grade and grade.is_relevant and docs:
            c = docs[-1]
            f = AuditFinding(policy_clause_id=state["policy_clause_id"], policy_text_excerpt=state["policy_text"],
                             status=ComplianceStatus.NON_COMPLIANT, severity=RiskSeverity.HIGH,
                             statutory_verbatim_quote=c.text, evidence_ids=[c.id],
                             gap_analysis="gap", remediation_action="fix")
        else:
            f = make_insufficient_finding(state["policy_clause_id"], state["policy_text"], "no evidence")
        return {"audit_report": f}
    return auditor


def make_verifier(order):
    def verifier(state):
        order.append("verify")
        return verifier_node(state)
    return verifier


def build(chunks, relevant_calls=(), fallback_results=(), auditor=None, grade_log=None):
    order = []
    r = TracingRetriever(chunks, order)
    g = create_audit_graph(r, grader=make_grader(order, relevant_calls, grade_log), rewriter=make_rewriter(order),
                           web_fallback=make_fallback(order, fallback_results),
                           auditor=auditor or make_auditor(order), verifier=make_verifier(order))
    return g, r, order


def test_initial_state_has_every_contract_field():
    s = make_initial_state("P-1", "text")
    assert set(s) == set(AuditGraphState.__annotations__)
    assert s["retrieved_docs"] == [] and s["document_grade"] is None and s["retry_count"] == 0
    assert s["web_fallback_used"] is False and s["web_fallback_attempted"] is False
    assert s["is_grounded"] is False and s["audit_report"] is None and s["error_message"] is None
    assert s["current_query"] == "text"


def test_happy_path_exact_node_order():
    g, r, order = build([LOCAL], relevant_calls=(1,))
    out = run_audit(g, "P-1", CLAUSE)
    assert order == ["retrieve", "grade", "audit", "verify"]
    assert len(r.calls) == 1 and out["retry_count"] == 0
    assert out["is_grounded"] is True and out["audit_report"].status == ComplianceStatus.NON_COMPLIANT


def test_always_irrelevant_exact_sequence_and_counts():
    g, r, order = build([LOCAL])  # non-empty evidence, grader always says irrelevant
    out = run_audit(g, "P-1", CLAUSE)
    assert order == ["retrieve", "grade", "rewrite", "retrieve", "grade", "rewrite", "retrieve", "grade",
                     "web_fallback", "grade", "audit", "verify"]
    assert order.count("rewrite") == 2 and order.count("retrieve") == 3 and len(r.calls) == 3
    assert order.count("web_fallback") == 1 and order.count("grade") == 4
    assert out["retry_count"] == 2 and out["web_fallback_attempted"] is True and out["web_fallback_used"] is False
    assert out["audit_report"].status == ComplianceStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("m", [0, 1, 2, 3])
def test_counts_follow_max_retry_count(monkeypatch, m):
    monkeypatch.setattr(graph_mod.settings, "MAX_RETRY_COUNT", m)
    g, r, order = build([LOCAL])
    out = run_audit(g, "P-1", CLAUSE)
    assert order.count("rewrite") == m and order.count("retrieve") == m + 1 and len(r.calls) == m + 1
    assert order.count("web_fallback") == 1 and order.count("grade") == m + 2
    assert order.count("audit") == 1 and order.count("verify") == 1 and out["retry_count"] == m


def test_retry_count_seen_by_each_grading():
    log = []
    g, _, _ = build([LOCAL], grade_log=log)
    run_audit(g, "P-1", CLAUSE)
    assert [c for c, _ in log] == [0, 1, 2, 2]


def test_missing_tavily_key_terminates_safely():
    g, r, order = build([], fallback_results=())  # fallback runs, finds nothing (no key / failure)
    out = run_audit(g, "P-1", CLAUSE)
    assert order.count("web_fallback") == 1 and order.count("retrieve") == 3 and len(r.calls) == 3
    assert order[-3:] == ["grade", "audit", "verify"]
    assert out["web_fallback_attempted"] is True and out["web_fallback_used"] is False
    assert out["audit_report"].status == ComplianceStatus.INSUFFICIENT_EVIDENCE


def test_fallback_results_are_graded_before_audit_and_cited_as_web():
    g, _, order = build([], relevant_calls=(4,), fallback_results=(WEB,))
    out = run_audit(g, "P-1", CLAUSE)
    i = order.index("web_fallback")
    assert order[i:] == ["web_fallback", "grade", "audit", "verify"]
    assert out["web_fallback_used"] is True and out["is_grounded"] is True
    assert out["audit_report"].evidence_ids == ["web-1"]
    assert out["retrieved_docs"][-1].source_url == "https://example.org/rule"


def test_fallback_results_still_irrelevant_end_insufficient():
    g, _, order = build([], fallback_results=(WEB,))
    out = run_audit(g, "P-1", CLAUSE)
    assert order.count("web_fallback") == 1 and order.count("rewrite") == 2
    assert out["web_fallback_used"] is True
    assert out["audit_report"].status == ComplianceStatus.INSUFFICIENT_EVIDENCE


def test_non_empty_irrelevant_evidence_never_goes_straight_to_audit():
    log = []
    g, _, order = build([LOCAL], grade_log=log)
    out = run_audit(g, "P-1", CLAUSE)
    assert all(n > 0 for _, n in log[:3])  # evidence really was non-empty
    assert order.index("audit") > order.index("web_fallback")
    assert out["audit_report"].status != ComplianceStatus.COMPLIANT
    assert out["audit_report"].status == ComplianceStatus.INSUFFICIENT_EVIDENCE


def test_verifier_failure_terminates_safely_without_looping():
    def bad_auditor(state):
        return {"audit_report": AuditFinding(
            policy_clause_id="P-1", policy_text_excerpt="x", status=ComplianceStatus.NON_COMPLIANT,
            severity=RiskSeverity.HIGH, statutory_verbatim_quote="made up", evidence_ids=["ghost"],
            gap_analysis="g", remediation_action="r")}
    g, r, order = build([LOCAL], relevant_calls=(1,), auditor=bad_auditor)
    out = run_audit(g, "P-1", CLAUSE)
    assert order == ["retrieve", "grade", "verify"] and len(r.calls) == 1
    assert out["is_grounded"] is False
    rep = out["audit_report"]
    assert rep.status == ComplianceStatus.INSUFFICIENT_EVIDENCE and rep.evidence_ids == []
    assert rep.statutory_verbatim_quote is None


def test_blank_query_does_not_call_retriever_and_still_terminates():
    g, r, order = build([LOCAL])
    out = run_audit(g, "P-1", CLAUSE, current_query="   ")
    # first retrieval is skipped; the (fake) rewriter then supplies a non-empty query
    assert len(r.calls) == 2 and order.count("web_fallback") == 1
    assert out["audit_report"].status == ComplianceStatus.INSUFFICIENT_EVIDENCE


# --- router unit tests -------------------------------------------------------------------
def _st(relevant=None, attempted=False, retry=0, grounded=False):
    grade = None if relevant is None else DocumentGrade(is_relevant=relevant, confidence_score=0.5, rationale="r")
    return {"document_grade": grade, "web_fallback_attempted": attempted, "retry_count": retry,
            "is_grounded": grounded, "retrieved_docs": [LOCAL]}


def test_router_decisions():
    assert route_after_grade(_st(True)) == "audit"
    assert route_after_grade(_st(False, retry=0)) == "rewrite"
    assert route_after_grade(_st(False, retry=1)) == "rewrite"
    assert route_after_grade(_st(False, retry=2)) == "web_fallback"
    assert route_after_grade(_st(None, retry=0)) == "rewrite"          # missing grade == not relevant
    assert route_after_grade(_st(False, attempted=True, retry=0)) == "audit"   # never rewrite after fallback
    assert route_after_grade(_st(False, attempted=True, retry=2)) == "audit"


def test_router_ignores_is_grounded_and_doc_count():
    assert route_after_grade(_st(False, grounded=True)) == "rewrite"
    assert route_after_grade(_st(True, grounded=False)) == "audit"


def test_default_nodes_fail_clearly_when_not_implemented():
    with pytest.raises(NotImplementedError):
        create_audit_graph(MockRegulatoryRetriever([]))