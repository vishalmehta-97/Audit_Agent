"""Evaluation on SYNTHETIC cases. Default run is OFFLINE and deterministic.

IMPORTANT: the offline run uses a scripted fake LLM that returns pre-written answers. It checks that the
pipeline (routing, retries, fallback, verifier) is wired correctly. It does NOT measure model quality, and
the synthetic rules are NOT real law, so nothing here is a measure of legal accuracy.

Usage:
    python evaluate.py                 # offline, deterministic
    python evaluate.py --live          # LIVE model (needs GROQ_API_KEY); clearly labeled, opt-in; UNTESTED here
    python evaluate.py --live --web    # additionally allow live web fallback (needs TAVILY_API_KEY)
    python evaluate.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.graph import create_audit_graph, run_audit
from src.llm_clients import MissingCredentialError, make_default_llm
from src.schemas import (AuditFinding, ComplianceStatus, DocumentGrade, EvidenceChunk, RiskSeverity)
from tests.fakes import FakeLLM, FakeSearch
from tests.mocks.mock_retriever import MockRegulatoryRetriever

DOC = "SYNTHETIC TEST RULEBOOK (not real law)"
SYNTHETIC_CORPUS: List[EvidenceChunk] = [
    EvidenceChunk(id="syn-r1", document_name=DOC, page_number=1, section_title="Synthetic Rule R-1 Log Retention",
                  text="Synthetic Rule R-1: Access logs must be retained for a minimum of 90 days."),
    EvidenceChunk(id="syn-r2", document_name=DOC, page_number=2, section_title="Synthetic Rule R-2 Data Location",
                  text="Synthetic Rule R-2: Personal data must be stored on servers located within Region Alpha."),
    EvidenceChunk(id="syn-r3", document_name=DOC, page_number=3, section_title="Synthetic Rule R-3 Encryption",
                  text="Synthetic Rule R-3: Personal data must be encrypted at rest using an approved algorithm."),
    EvidenceChunk(id="syn-r4", document_name=DOC, page_number=4, section_title="Synthetic Rule R-4 Access Review",
                  text="Synthetic Rule R-4: Access rights must be reviewed periodically."),
]
_CHUNK = {c.id: c for c in SYNTHETIC_CORPUS}


@dataclass(frozen=True)
class EvalCase:
    id: str
    title: str
    clause_text: str
    query: str  # explicit retrieval query so the offline mock retriever behaves predictably
    expected: ComplianceStatus  # explicit SYNTHETIC ground truth
    kind: str


CASES: List[EvalCase] = [
    EvalCase("retention", "Retention mismatch (synthetic)", "We retain access logs for 30 days.",
             "access logs retained days", ComplianceStatus.NON_COMPLIANT, "classification"),
    EvalCase("location", "Data-location mismatch (synthetic)", "Customer personal data is stored on servers in Region Beta.",
             "personal data stored servers region", ComplianceStatus.NON_COMPLIANT, "classification"),
    EvalCase("compliant", "Compliant clause (synthetic)", "Personal data is encrypted at rest using an approved algorithm.",
             "personal data encrypted rest approved algorithm", ComplianceStatus.COMPLIANT, "classification"),
    EvalCase("ambiguous", "Ambiguous wording (synthetic)", "Access rights are reviewed from time to time.",
             "access rights reviewed", ComplianceStatus.PARTIALLY_COMPLIANT, "classification"),
    EvalCase("out_of_scope", "Out-of-scope clause (synthetic)", "Our cafeteria serves vegetarian meals on Fridays.",
             "cafeteria vegetarian meals fridays", ComplianceStatus.INSUFFICIENT_EVIDENCE, "classification"),
    EvalCase("fabricated_citation", "Verifier probe: fabricated quote (synthetic)", "We retain access logs for 30 days.",
             "access logs retained days", ComplianceStatus.INSUFFICIENT_EVIDENCE, "verifier-check"),
]

# case id -> (cited chunk id, status, severity, gap, remediation, quote override)
_SCRIPT = {
    "retention": ("syn-r1", ComplianceStatus.NON_COMPLIANT, RiskSeverity.HIGH,
                  "The clause retains logs for 30 days; the rule requires at least 90.", "Retain access logs for 90 days or more.", None),
    "location": ("syn-r2", ComplianceStatus.NON_COMPLIANT, RiskSeverity.HIGH,
                 "Data is stored in Region Beta; the rule requires Region Alpha.", "Move storage to servers in Region Alpha.", None),
    "compliant": ("syn-r3", ComplianceStatus.COMPLIANT, RiskSeverity.INFORMATIONAL,
                  "The clause states encryption at rest with an approved algorithm, as the rule requires.", "No action needed.", None),
    "ambiguous": ("syn-r4", ComplianceStatus.PARTIALLY_COMPLIANT, RiskSeverity.MEDIUM,
                  "A review is mentioned but 'from time to time' is not clearly 'periodic'.", "State a fixed review interval.", None),
    "fabricated_citation": ("syn-r1", ComplianceStatus.NON_COMPLIANT, RiskSeverity.HIGH,
                            "Scripted to cite a quote that is NOT in the evidence.", "n/a",
                            "Access logs must be retained for 365 days."),
}


def _grade(rel: bool) -> DocumentGrade:
    return DocumentGrade(is_relevant=rel, confidence_score=0.9, rationale="scripted")


def _scripted_llm(case: EvalCase) -> FakeLLM:
    if case.id == "out_of_scope":
        return FakeLLM({"DocumentGrade": [_grade(False)]}, texts=["cafeteria vegetarian menu", "friday meal service"])
    cid, status, sev, gap, fix, quote = _SCRIPT[case.id]
    chunk = _CHUNK[cid]
    finding = AuditFinding(policy_clause_id=case.id, policy_text_excerpt=case.clause_text, status=status, severity=sev,
                           statutory_clause=chunk.text.split(":")[0], statutory_verbatim_quote=quote or chunk.text,
                           evidence_ids=[cid], gap_analysis=gap, remediation_action=fix)
    return FakeLLM({"DocumentGrade": [_grade(True)], "AuditFinding": [finding]}, texts=["unused"])


def build_offline_graph(case: EvalCase):
    """Fresh graph per case: mock retriever + scripted LLM + search that never touches the network."""
    return create_audit_graph(MockRegulatoryRetriever(SYNTHETIC_CORPUS), llm=_scripted_llm(case), search=FakeSearch([]))


def _build_live_graph(llm, web: bool):
    return create_audit_graph(MockRegulatoryRetriever(SYNTHETIC_CORPUS), llm=llm, search=None if web else (lambda q, n: []))


def run_case(case: EvalCase, graph) -> Dict[str, Any]:
    t0 = time.perf_counter()
    row: Dict[str, Any] = {"id": case.id, "title": case.title, "kind": case.kind, "expected": case.expected.value,
                           "actual": None, "correct": False, "grounded": False, "error": None}
    try:
        out = run_audit(graph, case.id, case.clause_text, case.query)
        row["actual"] = out["audit_report"].status.value
        row["grounded"] = bool(out["is_grounded"])
        row["correct"] = row["actual"] == row["expected"]
    except MissingCredentialError:
        raise
    except Exception as exc:  # record, keep evaluating the remaining cases
        row["error"] = type(exc).__name__
    row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return row


def run_evaluation(live: bool = False, web: bool = False) -> Dict[str, Any]:
    llm = make_default_llm() if live else None  # raises MissingCredentialError if no key
    rows = []
    for case in CASES:
        graph = _build_live_graph(llm, web) if live else build_offline_graph(case)
        rows.append(run_case(case, graph))
    done = [r for r in rows if r["error"] is None]
    return {
        "mode": "LIVE" if live else "OFFLINE_SCRIPTED", "web_fallback_live": bool(live and web), "cases": rows,
        "total": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "grounding_rate": (sum(r["grounded"] for r in done) / len(done)) if done else 0.0,
        "avg_latency_ms": (sum(r["latency_ms"] for r in rows) / len(rows)),
        "insufficient_count": sum(r["actual"] == ComplianceStatus.INSUFFICIENT_EVIDENCE.value for r in rows),
        "errors": [{"id": r["id"], "error": r["error"]} for r in rows if r["error"]],
    }


def format_report(s: Dict[str, Any]) -> str:
    if s["mode"] == "LIVE":
        banner = "LIVE RUN (real model; synthetic cases and corpus; NOT a measure of legal accuracy)"
    else:
        banner = ("OFFLINE SCRIPTED RUN: a fake LLM returns pre-written answers. This checks pipeline and verifier\n"
                  "wiring only. It does NOT measure model quality or legal accuracy; 100% here is by construction.")
    lines = [banner, "", f"{'case':<22}{'expected':<24}{'actual':<24}{'ok':<5}{'grounded':<10}{'ms':>8}", "-" * 93]
    for r in s["cases"]:
        lines.append(f"{r['id']:<22}{r['expected']:<24}{str(r['actual']):<24}{('yes' if r['correct'] else 'NO'):<5}"
                     f"{('yes' if r['grounded'] else 'no'):<10}{r['latency_ms']:>8}")
    lines += ["", f"classification accuracy vs synthetic ground truth: {s['accuracy']:.0%} ({round(s['accuracy'] * s['total'])}/{s['total']})",
              f"citation grounding rate (completed cases): {s['grounding_rate']:.0%}",
              f"average latency per clause: {s['avg_latency_ms']:.1f} ms",
              f"insufficient-evidence results: {s['insufficient_count']}",
              f"errors/failures: {len(s['errors'])}" + (f" {s['errors']}" if s["errors"] else "")]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="use the real LLM (opt-in)")
    ap.add_argument("--web", action="store_true", help="with --live: also allow live web fallback")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = ap.parse_args(argv)
    try:
        summary = run_evaluation(live=args.live, web=args.web)
    except MissingCredentialError:
        print("Live evaluation needs GROQ_API_KEY in your .env file. Nothing was run.")
        return 2
    print(format_report(summary))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())