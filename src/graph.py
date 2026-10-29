"""LangGraph workflow (Developer B).

START -> retrieve -> grade -> {audit | rewrite -> retrieve | web_fallback -> grade}
audit -> verify -> END.   Routing uses document_grade.is_relevant, NEVER is_grounded or doc count.
"""
from __future__ import annotations

from typing import Callable, Optional

from langgraph.graph import END, START, StateGraph

import src.nodes as nodes
from src.config import settings
from src.retriever_interface import BaseRegulatoryRetriever
from src.llm_clients import LLMClient, SearchFn
from src.schemas import AuditGraphState

NodeFn = Callable[[dict], dict]


def make_initial_state(policy_clause_id: str, policy_text: str,
                       current_query: Optional[str] = None) -> AuditGraphState:
    """Every required field is initialized, per the frozen contract."""
    return {
        "policy_clause_id": policy_clause_id,
        "policy_text": policy_text,
        "current_query": policy_text if current_query is None else current_query,
        "retrieved_docs": [],
        "document_grade": None,
        "retry_count": 0,
        "web_fallback_used": False,
        "web_fallback_attempted": False,
        "is_grounded": False,
        "audit_report": None,
        "error_message": None,
    }


def route_after_grade(state: dict) -> str:
    """Return the next node name after grading. Order of checks matters."""
    grade = state.get("document_grade")
    if grade is not None and grade.is_relevant:
        return "audit"
    if state.get("web_fallback_attempted"):
        return "audit"  # still irrelevant after fallback: audit only to emit INSUFFICIENT_EVIDENCE
    if state.get("retry_count", 0) < settings.MAX_RETRY_COUNT:
        return "rewrite"
    return "web_fallback"


def _resolve(name: str, override: Optional[NodeFn], client=None) -> NodeFn:
    """Use the override if given, else the node from src.nodes with the optional client bound.
    The wrapper takes only `state`, and live clients are created lazily inside the node."""
    if override is not None:
        return override
    fn = getattr(nodes, name, None)
    if fn is None:
        raise NotImplementedError(f"src.nodes.{name} is not implemented yet; pass an override.")
    if client is None:
        return fn
    return lambda state: fn(state, client)


def create_audit_graph(retriever: BaseRegulatoryRetriever, *, grader: Optional[NodeFn] = None,
                       rewriter: Optional[NodeFn] = None, web_fallback: Optional[NodeFn] = None,
                       auditor: Optional[NodeFn] = None, verifier: Optional[NodeFn] = None,
                       llm: Optional[LLMClient] = None, search: Optional[SearchFn] = None):
    """Build and compile the audit graph. Keyword overrides exist so tests can inject fake nodes."""
    grade_fn = _resolve("grader_node", grader, llm)
    rewrite_fn = _resolve("rewriter_node", rewriter, llm)
    fallback_fn = _resolve("web_fallback_node", web_fallback, search)
    audit_fn = _resolve("auditor_node", auditor, llm)
    verify_fn = _resolve("verifier_node", verifier)

    def retrieve(state: dict) -> dict:
        return nodes.retriever_node(state, retriever)

    g = StateGraph(AuditGraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade_fn)
    g.add_node("rewrite", rewrite_fn)
    g.add_node("web_fallback", fallback_fn)
    g.add_node("audit", audit_fn)
    g.add_node("verify", verify_fn)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"audit": "audit", "rewrite": "rewrite", "web_fallback": "web_fallback"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web_fallback", "grade")
    g.add_edge("audit", "verify")
    g.add_edge("verify", END)
    return g.compile()


def run_audit(graph, policy_clause_id: str, policy_text: str, current_query: Optional[str] = None) -> dict:
    """Invoke the compiled graph on one clause. recursion_limit is a safety net, not the bound."""
    return graph.invoke(make_initial_state(policy_clause_id, policy_text, current_query),
                        config={"recursion_limit": 50})