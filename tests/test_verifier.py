"""Offline tests for the deterministic citation verifier."""
from src.nodes import normalize_text, verifier_node
from src.schemas import (AuditFinding, ComplianceStatus, EvidenceChunk, RiskSeverity, SourceType)

QUOTE = "Records must be retained for a minimum of 90 days."


def chunk(cid="c1", text=f"Section 4.2. {QUOTE} Other text follows.", **kw):
    return EvidenceChunk(id=cid, document_name="SYNTHETIC RULEBOOK", page_number=kw.pop("page_number", 3),
                         section_title=kw.pop("section_title", "Retention"), text=text, **kw)


def finding(**kw):
    base = dict(policy_clause_id="P-1", policy_text_excerpt="We keep logs 30 days.",
                status=ComplianceStatus.NON_COMPLIANT, severity=RiskSeverity.HIGH,
                statutory_clause="Section 4.2", statutory_verbatim_quote=QUOTE,
                evidence_ids=["c1"], gap_analysis="30 < 90", remediation_action="Retain 90 days")
    base.update(kw)
    return AuditFinding(**base)


def run(f, docs, **extra):
    state = {"policy_clause_id": "P-1", "policy_text": "We keep logs 30 days.",
             "audit_report": f, "retrieved_docs": docs, **extra}
    return verifier_node(state), state


def assert_safe(out):
    r = out["audit_report"]
    assert out["is_grounded"] is False
    assert r.status == ComplianceStatus.INSUFFICIENT_EVIDENCE and r.severity == RiskSeverity.INFORMATIONAL
    assert r.evidence_ids == [] and r.statutory_verbatim_quote is None and r.statutory_clause is None
    assert "verification failed" in out["error_message"].lower()


def test_valid_id_and_exact_quote_pass():
    out, _ = run(finding(), [chunk()])
    assert out == {"is_grounded": True}


def test_fabricated_id_fails():
    out, _ = run(finding(evidence_ids=["ghost"]), [chunk()])
    assert_safe(out)
    assert "ghost" in out["error_message"]


def test_fabricated_quote_fails():
    out, _ = run(finding(statutory_verbatim_quote="Records must be kept forever."), [chunk()])
    assert_safe(out)


def test_quote_in_non_cited_chunk_fails():
    docs = [chunk("c1", text="Section 4.2. Unrelated text only."), chunk("c2")]  # quote lives in c2
    out, _ = run(finding(evidence_ids=["c1"]), docs)
    assert_safe(out)


def test_unsupported_statutory_clause_fails():
    out, _ = run(finding(statutory_clause="Section 99.9"), [chunk()])
    assert_safe(out)


def test_clause_supported_by_section_title():
    docs = [chunk(text=QUOTE, section_title="Section 4.2 Retention")]
    out, _ = run(finding(), docs)
    assert out == {"is_grounded": True}


def test_web_source_url_remains_traceable():
    web = chunk("w1", source_type=SourceType.WEB, page_number=None, source_url="https://example.org/rule")
    out, _ = run(finding(evidence_ids=["w1"]), [web])
    assert out == {"is_grounded": True}
    assert web.source_url == "https://example.org/rule"


def test_web_without_url_or_with_page_fails():
    no_url = chunk("w1", source_type=SourceType.WEB, page_number=None, source_url=None)
    assert_safe(run(finding(evidence_ids=["w1"]), [no_url])[0])
    paged = chunk("w2", source_type=SourceType.WEB, page_number=2, source_url="https://example.org/x")
    assert_safe(run(finding(evidence_ids=["w2"]), [paged])[0])


def test_failed_verification_returns_safe_finding_keeps_policy_context():
    out, _ = run(finding(evidence_ids=["ghost"]), [chunk()])
    r = out["audit_report"]
    assert r.policy_clause_id == "P-1" and r.policy_text_excerpt == "We keep logs 30 days."
    assert "withheld" in r.gap_analysis and r.remediation_action


def test_typographic_and_whitespace_differences_pass():
    docs = [chunk(text="The \u201cdata\u201d must be kept\u00a0for\n  90\u2013day periods.")]
    f = finding(statutory_clause=None, statutory_verbatim_quote='The "data" must be kept for 90-day periods.')
    assert run(f, docs)[0] == {"is_grounded": True}


def test_no_fuzzy_matching_case_or_word_changes_fail():
    assert_safe(run(finding(statutory_verbatim_quote=QUOTE.lower()), [chunk()])[0])
    assert_safe(run(finding(statutory_verbatim_quote="Records must be retained for at least 90 days."), [chunk()])[0])


def test_blank_quote_fails():
    assert_safe(run(finding(statutory_verbatim_quote="   "), [chunk()])[0])


def test_substantive_finding_without_evidence_fails():
    assert_safe(run(finding(evidence_ids=[], statutory_clause=None, statutory_verbatim_quote=None), [chunk()])[0])


def test_insufficient_finding_without_citations_passes():
    f = finding(status=ComplianceStatus.INSUFFICIENT_EVIDENCE, severity=RiskSeverity.INFORMATIONAL,
                statutory_clause=None, statutory_verbatim_quote=None, evidence_ids=[])
    assert run(f, [])[0] == {"is_grounded": True}


def test_missing_report_fails_closed():
    out, _ = run(None, [chunk()])
    assert_safe(out)


def test_error_message_preserved_and_state_not_mutated():
    f = finding(evidence_ids=["ghost"])
    out, state = run(f, [chunk()], error_message="earlier problem")
    assert out["error_message"].startswith("earlier problem | ")
    assert state["audit_report"] is f and f.evidence_ids == ["ghost"]


def test_deterministic():
    a, _ = run(finding(evidence_ids=["ghost"]), [chunk()])
    b, _ = run(finding(evidence_ids=["ghost"]), [chunk()])
    assert a == b


def test_normalize_does_not_change_case_or_words():
    assert normalize_text("  A\u2019s  \n B ") == "A's B"