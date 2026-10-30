"""AuditAgent Streamlit UI (minimal). Run:  streamlit run app.py"""
import logging

import streamlit as st

import evaluate
from src.config import settings
from src.graph import create_audit_graph, stream_audit
from src.llm_clients import MissingCredentialError
from src.report import DISCLAIMER, build_report_view, safe_error_message
from tests.mocks.mock_retriever import MockRegulatoryRetriever

logger = logging.getLogger("auditagent.app")
LIVE, DEMO = "Live model", "Offline scripted demo"
CASES = {c.title: c for c in evaluate.CASES}


def get_retriever():
    """HybridRetriever when Developer A's module is ready; otherwise the synthetic demo corpus (flagged)."""
    try:
        from src.retriever import HybridRetriever
        return HybridRetriever(), "hybrid"  # NOTE: confirm constructor arguments at integration time
    except Exception as exc:
        logger.warning("HybridRetriever unavailable (%s); using the synthetic demo corpus.", type(exc).__name__)
        return MockRegulatoryRetriever(evaluate.SYNTHETIC_CORPUS), "demo"


def apply_sample():
    case = CASES.get(st.session_state.get("sample"))
    if case:
        st.session_state["clause_id"] = case.id.upper()
        st.session_state["clause_text"] = case.clause_text


def render(v: dict) -> None:
    if not v["has_result"]:
        st.error("No result was produced.")
        return
    for w in v["warnings"]:
        st.warning(w)
    c1, c2 = st.columns(2)
    c1.metric("Compliance status", v["status"].replace("_", " "))
    c2.metric("Risk severity", v["severity"])
    st.subheader("Gap analysis")
    st.write(v["gap_analysis"])
    st.subheader("Recommended remediation")
    st.write(v["remediation"])
    if v["statutory_clause"]:
        st.caption(f"Cited clause: {v['statutory_clause']}")
    if v["quote"]:
        st.subheader("Verified quote")
        st.code(v["quote"], language=None)
    if v["evidence"]:
        st.subheader("Evidence")
        for e in v["evidence"]:
            with st.expander(f"{e['id']} | {e['title']}"):
                st.text(f"Source: {e['source']}\nSection: {e['section']}\nPage: {e['page']}")
                if e["url"]:
                    st.text(f"URL: {e['url']}")
                st.text(e["text"])
    st.caption(f"Query rewrites: {v['retries']} | web fallback attempted: {v['web_attempted']} | used: {v['web_used']}")
    if v["verification_note"]:
        with st.expander("Verification note"):
            st.text(v["verification_note"])


def run(clause_id: str, text: str, mode: str) -> None:
    query = None
    if mode == DEMO:
        chosen = CASES.get(st.session_state.get("sample"))  # prefer the selected sample (texts can repeat)
        case = chosen if chosen and chosen.clause_text == text else \
            next((c for c in evaluate.CASES if c.clause_text == text), None)
        if case is None:
            st.warning("The offline demo only supports the built-in sample clauses. Pick one from the list.")
            return
        st.warning("SCRIPTED DEMO: answers are pre-written for these synthetic samples. This is not a real audit.")
        graph, query = evaluate.build_offline_graph(case), case.query
    else:
        if not settings.GROQ_API_KEY:
            st.error(safe_error_message(MissingCredentialError()))
            return
        retriever, kind = get_retriever()
        if kind == "demo":
            st.warning("Using the built-in SYNTHETIC demo corpus (not real law); the real retriever is not available.")
        graph = create_audit_graph(retriever)
    final = None
    try:
        with st.status("Running audit...", expanded=True) as status:
            for node, snapshot in stream_audit(graph, clause_id, text, query):
                st.write(f"Completed step: **{node}**")  # genuine graph events
                final = snapshot
            status.update(label="Audit finished", state="complete")
    except Exception as exc:
        logger.exception("audit failed")  # details stay in the server log
        st.error(safe_error_message(exc))
        return
    render(build_report_view(final))


st.set_page_config(page_title="AuditAgent", layout="wide")
st.title("AuditAgent")
st.caption(DISCLAIMER)
st.info("This prototype audits ONE policy clause at a time. Whole-document auditing is not implemented.")

st.session_state.setdefault("clause_id", "CLAUSE-1")
st.session_state.setdefault("clause_text", "")
mode = st.sidebar.radio("Mode", [LIVE, DEMO], index=0 if settings.GROQ_API_KEY else 1, key="mode")
if mode == DEMO:
    st.sidebar.warning("Scripted demo: no model is called.")
st.selectbox("Sample synthetic clauses (optional)", [""] + list(CASES), key="sample", on_change=apply_sample)
st.text_input("Policy clause ID", key="clause_id")
st.text_area("Policy clause text (one clause)", key="clause_text", height=140)

if st.button("Run Audit", type="primary"):
    cid, ctext = st.session_state["clause_id"].strip(), st.session_state["clause_text"].strip()
    if not cid or not ctext:
        st.warning("Enter a clause ID and the clause text.")
    else:
        run(cid, ctext, mode)