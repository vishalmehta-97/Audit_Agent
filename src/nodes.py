# """Graph nodes (Developer B). This file currently holds the deterministic verifier only;
# retriever/grader/rewriter/web_fallback/auditor nodes are added in later steps.
# No live clients are created at import time.
# """
# from __future__ import annotations

# import logging
# from typing import Dict, List, Optional

# from src.schemas import (AuditFinding, ComplianceStatus, EvidenceChunk, RiskSeverity, SourceType)

# logger = logging.getLogger(__name__)

# # --- conservative normalization: whitespace + typographic characters ONLY -------------------
# _TYPO_MAP = str.maketrans({
#     "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
#     "\u201c": '"', "\u201d": '"', "\u201e": '"',
#     "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
#     "\u00a0": " ", "\u2009": " ", "\u202f": " ",
#     "\u200b": None, "\u00ad": None, "\ufeff": None,          # invisible characters
#     "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",  # PDF ligatures
# })


# def normalize_text(text: str) -> str:
#     """Normalize whitespace and typographic variants. Case and wording are NOT changed."""
#     return " ".join(text.translate(_TYPO_MAP).split())


# def _is_real_url(url: Optional[str]) -> bool:
#     return bool(url) and url.strip().lower().startswith(("http://", "https://")) and len(url.strip()) > 8


# def check_citations(finding: AuditFinding, retrieved_docs: List[EvidenceChunk]) -> List[str]:
#     """Return a list of verification failures (empty list == grounded). Pure and deterministic."""
#     problems: List[str] = []
#     by_id: Dict[str, EvidenceChunk] = {}
#     for d in retrieved_docs:
#         by_id.setdefault(d.id, d)

#     substantive = finding.status != ComplianceStatus.INSUFFICIENT_EVIDENCE
#     if substantive and not finding.evidence_ids:
#         problems.append("a substantive finding must cite at least one evidence chunk")

#     cited: List[EvidenceChunk] = []
#     for eid in finding.evidence_ids:
#         chunk = by_id.get(eid)
#         if chunk is None:
#             problems.append(f"evidence id {eid[:80]!r} does not exist in the retrieved evidence")
#         else:
#             cited.append(chunk)

#     for c in cited:  # web evidence must stay traceable
#         if c.source_type == SourceType.WEB:
#             if not _is_real_url(c.source_url):
#                 problems.append(f"web evidence {c.id[:80]!r} has no valid source_url")
#             if c.page_number is not None:
#                 problems.append(f"web evidence {c.id[:80]!r} must not have a page number")

#     quote = finding.statutory_verbatim_quote
#     if quote is not None:
#         nq = normalize_text(quote)
#         if not nq:
#             problems.append("the statutory quote is blank")
#         elif not any(nq in normalize_text(c.text) for c in cited):  # CITED chunks only
#             problems.append("the statutory quote was not found in any cited evidence chunk")

#     clause = finding.statutory_clause
#     if clause is not None:
#         nc = normalize_text(clause).casefold()
#         if not nc:
#             problems.append("the statutory clause identifier is blank")
#         elif not any(nc in normalize_text(c.text + " " + c.section_title).casefold() for c in cited):
#             problems.append("the statutory clause was not found in any cited evidence chunk")
#     return problems


# def make_insufficient_finding(clause_id: str, policy_text: str, gap_analysis: str,
#                               remediation: Optional[str] = None) -> AuditFinding:
#     """Safe fallback finding: no citations, no quote, no clause."""
#     return AuditFinding(
#         policy_clause_id=clause_id,
#         policy_text_excerpt=policy_text[:300],
#         status=ComplianceStatus.INSUFFICIENT_EVIDENCE,
#         severity=RiskSeverity.INFORMATIONAL,
#         statutory_clause=None,
#         statutory_verbatim_quote=None,
#         evidence_ids=[],
#         gap_analysis=gap_analysis,
#         remediation_action=remediation or
#         "Review this clause manually against the primary source; supply fuller regulatory text.",
#     )


# def verifier_node(state: dict) -> dict:
#     """Deterministically verify citations. Fails closed: any problem -> safe INSUFFICIENT_EVIDENCE."""
#     finding: Optional[AuditFinding] = state.get("audit_report")
#     docs: List[EvidenceChunk] = state.get("retrieved_docs") or []
#     clause_id = state.get("policy_clause_id", "UNKNOWN")
#     policy_text = state.get("policy_text", "")

#     if finding is None:
#         problems = ["no audit finding was produced"]
#     else:
#         problems = check_citations(finding, docs)

#     if not problems:
#         return {"is_grounded": True}

#     message = "Citation verification failed: " + "; ".join(problems)
#     logger.warning("verifier rejected finding for %s: %s", clause_id, message)
#     safe = make_insufficient_finding(
#         clause_id,
#         finding.policy_text_excerpt if finding else policy_text,
#         gap_analysis=("The automated finding could not be verified against the retrieved evidence, so it "
#                       "is withheld. " + message),
#     )
#     prior = state.get("error_message")
#     return {"audit_report": safe, "is_grounded": False,
#             "error_message": f"{prior} | {message}" if prior else message}




"""Graph nodes (Developer B). This file currently holds the deterministic verifier only;
retriever/grader/rewriter/web_fallback/auditor nodes are added in later steps.
No live clients are created at import time.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.config import settings
from src.retriever_interface import BaseRegulatoryRetriever
from src.schemas import (AuditFinding, ComplianceStatus, EvidenceChunk, RiskSeverity, SourceType)

logger = logging.getLogger(__name__)

# --- conservative normalization: whitespace + typographic characters ONLY -------------------
_TYPO_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",
    "\u200b": None, "\u00ad": None, "\ufeff": None,          # invisible characters
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",  # PDF ligatures
})


def normalize_text(text: str) -> str:
    """Normalize whitespace and typographic variants. Case and wording are NOT changed."""
    return " ".join(text.translate(_TYPO_MAP).split())


def _is_real_url(url: Optional[str]) -> bool:
    return bool(url) and url.strip().lower().startswith(("http://", "https://")) and len(url.strip()) > 8


def check_citations(finding: AuditFinding, retrieved_docs: List[EvidenceChunk]) -> List[str]:
    """Return a list of verification failures (empty list == grounded). Pure and deterministic."""
    problems: List[str] = []
    by_id: Dict[str, EvidenceChunk] = {}
    for d in retrieved_docs:
        by_id.setdefault(d.id, d)

    substantive = finding.status != ComplianceStatus.INSUFFICIENT_EVIDENCE
    if substantive and not finding.evidence_ids:
        problems.append("a substantive finding must cite at least one evidence chunk")

    cited: List[EvidenceChunk] = []
    for eid in finding.evidence_ids:
        chunk = by_id.get(eid)
        if chunk is None:
            problems.append(f"evidence id {eid[:80]!r} does not exist in the retrieved evidence")
        else:
            cited.append(chunk)

    for c in cited:  # web evidence must stay traceable
        if c.source_type == SourceType.WEB:
            if not _is_real_url(c.source_url):
                problems.append(f"web evidence {c.id[:80]!r} has no valid source_url")
            if c.page_number is not None:
                problems.append(f"web evidence {c.id[:80]!r} must not have a page number")

    quote = finding.statutory_verbatim_quote
    if quote is not None:
        nq = normalize_text(quote)
        if not nq:
            problems.append("the statutory quote is blank")
        elif not any(nq in normalize_text(c.text) for c in cited):  # CITED chunks only
            problems.append("the statutory quote was not found in any cited evidence chunk")

    clause = finding.statutory_clause
    if clause is not None:
        nc = normalize_text(clause).casefold()
        if not nc:
            problems.append("the statutory clause identifier is blank")
        elif not any(nc in normalize_text(c.text + " " + c.section_title).casefold() for c in cited):
            problems.append("the statutory clause was not found in any cited evidence chunk")
    return problems


def make_insufficient_finding(clause_id: str, policy_text: str, gap_analysis: str,
                              remediation: Optional[str] = None) -> AuditFinding:
    """Safe fallback finding: no citations, no quote, no clause."""
    return AuditFinding(
        policy_clause_id=clause_id,
        policy_text_excerpt=policy_text[:300],
        status=ComplianceStatus.INSUFFICIENT_EVIDENCE,
        severity=RiskSeverity.INFORMATIONAL,
        statutory_clause=None,
        statutory_verbatim_quote=None,
        evidence_ids=[],
        gap_analysis=gap_analysis,
        remediation_action=remediation or
        "Review this clause manually against the primary source; supply fuller regulatory text.",
    )


def verifier_node(state: dict) -> dict:
    """Deterministically verify citations. Fails closed: any problem -> safe INSUFFICIENT_EVIDENCE."""
    finding: Optional[AuditFinding] = state.get("audit_report")
    docs: List[EvidenceChunk] = state.get("retrieved_docs") or []
    clause_id = state.get("policy_clause_id", "UNKNOWN")
    policy_text = state.get("policy_text", "")

    if finding is None:
        problems = ["no audit finding was produced"]
    else:
        problems = check_citations(finding, docs)

    if not problems:
        return {"is_grounded": True}

    message = "Citation verification failed: " + "; ".join(problems)
    logger.warning("verifier rejected finding for %s: %s", clause_id, message)
    safe = make_insufficient_finding(
        clause_id,
        finding.policy_text_excerpt if finding else policy_text,
        gap_analysis=("The automated finding could not be verified against the retrieved evidence, so it "
                      "is withheld. " + message),
    )
    prior = state.get("error_message")
    return {"audit_report": safe, "is_grounded": False,
            "error_message": f"{prior} | {message}" if prior else message}


def retriever_node(state: dict, retriever: BaseRegulatoryRetriever) -> dict:
    """Run one local retrieval attempt.

    Resets stale per-attempt fields (grade, grounding, report, error) but never touches the policy
    input, retry_count or the web-fallback flags. A blank query is handled without calling the
    retriever. Unexpected retriever exceptions propagate (they are not hidden).
    """
    reset = {"document_grade": None, "is_grounded": False, "audit_report": None, "error_message": None}
    query = (state.get("current_query") or "").strip()
    if not query:
        return {**reset, "retrieved_docs": [],
                "error_message": "Retrieval skipped: the search query is empty."}
    docs = retriever.retrieve(query, top_k=settings.RETRIEVAL_TOP_K)
    return {**reset, "retrieved_docs": list(docs)}