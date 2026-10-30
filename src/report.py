"""Pure view-model helpers for the UI (no Streamlit import, fully unit-testable)."""
from __future__ import annotations

from typing import Any, Dict, List

from langgraph.errors import GraphRecursionError

from src.llm_clients import LLMOutputError, MissingCredentialError
from src.schemas import ComplianceStatus, SourceType

DISCLAIMER = ("AuditAgent is an assistive compliance-support prototype, not legal advice. Findings are generated "
              "from retrieved text and must be reviewed by a qualified person. Test rules may be synthetic and are "
              "not real law.")
MSG_NOT_VERIFIED = ("Verification failed: the automated finding could not be checked against the evidence, so it "
                    "has been withheld. Treat this clause as NOT reviewed.")
MSG_INSUFFICIENT = ("Insufficient evidence: the retrieved material was not enough to assess this clause. "
                    "This is not a finding of compliance or non-compliance.")
MSG_WEB = ("Some cited evidence comes from a web search. Web results are secondary and untrusted: confirm them "
           "against the official source before relying on them.")
MSG_WEB_EMPTY = "A web search was attempted but returned no usable results."


def safe_error_message(exc: BaseException) -> str:
    """User-safe text only: never include exception text, paths or keys (those go to the server log)."""
    if isinstance(exc, MissingCredentialError):
        return "A required API key is not configured. Add it to your local .env file and restart the app."
    if isinstance(exc, LLMOutputError):
        return "The language model returned an unusable response. Please try again."
    if isinstance(exc, GraphRecursionError):
        return "The audit stopped because it exceeded its safety step limit."
    return "Something went wrong while running the audit. Technical details were written to the server log."


def build_report_view(state: dict) -> Dict[str, Any]:
    report = state.get("audit_report")
    grounded = bool(state.get("is_grounded"))
    docs = {d.id: d for d in (state.get("retrieved_docs") or [])}
    view: Dict[str, Any] = {
        "has_result": report is not None, "warnings": [], "evidence": [], "grounded": grounded,
        "retries": state.get("retry_count", 0), "web_attempted": bool(state.get("web_fallback_attempted")),
        "web_used": bool(state.get("web_fallback_used")), "quote": None, "verification_note": None,
    }
    if report is None:
        view["warnings"].append("No result was produced.")
        return view

    cited = [docs[i] for i in report.evidence_ids if i in docs]
    for d in cited:
        view["evidence"].append({
            "id": d.id, "title": d.document_name, "section": d.section_title,
            "page": d.page_number if d.page_number is not None else "n/a",
            "source": "WEB (secondary, untrusted)" if d.source_type == SourceType.WEB else "INTERNAL corpus",
            "url": d.source_url if d.source_type == SourceType.WEB else None, "text": d.text})
    uses_web = any(d.source_type == SourceType.WEB for d in cited)

    if not grounded:
        view["warnings"].append(MSG_NOT_VERIFIED)
        view["verification_note"] = state.get("error_message")
    elif report.status == ComplianceStatus.INSUFFICIENT_EVIDENCE:
        view["warnings"].append(MSG_INSUFFICIENT)
    if uses_web:
        view["warnings"].append(MSG_WEB)
    elif view["web_attempted"] and not view["web_used"]:
        view["warnings"].append(MSG_WEB_EMPTY)

    view.update({
        "clause_id": report.policy_clause_id, "policy_excerpt": report.policy_text_excerpt,
        "status": report.status.value, "severity": report.severity.value,
        "gap_analysis": report.gap_analysis, "remediation": report.remediation_action,
        "statutory_clause": report.statutory_clause if grounded else None,
        "quote": report.statutory_verbatim_quote if grounded else None,  # never show an unverified quote
    })
    return view