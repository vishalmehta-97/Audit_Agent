"""End-to-end (offline): generated synthetic PDF -> ingestion -> real HybridRetriever (Chroma + BM25) -> real graph
nodes, with a fake LLM, fake search and fake embedder. Verifies the pieces fit together; says nothing about model
quality, real embeddings, or live Groq/Tavily behaviour."""
import re

import chromadb
import pytest

from src.graph import create_audit_graph, run_audit
from src.ingestion import chunk_pdf, ingest_pdf, main as ingest_main
from src.nodes import web_chunk_id
from src.retriever import HybridRetriever, RetrievalIndexError
from src.schemas import AuditFinding, ComplianceStatus, DocumentGrade, RiskSeverity, SourceType
from src.synthetic_corpus import write_synthetic_pdf
from tests.fake_embedder import HashingEmbedder
from tests.fakes import FakeLLM, FakeSearch

EVIDENCE = re.compile(r'<evidence id="([^"]+)"[^>]*>\n(.*?)\n</evidence>', re.DOTALL)
WEB_RES = {"title": "Official Rule", "url": "https://www.example.gov/rule",
           "content": "Providers must keep logs for 90 days."}


def fresh_process():
    chromadb.api.client.SharedSystemClient.clear_system_cache()


@pytest.fixture
def pdf(tmp_path):
    return write_synthetic_pdf(tmp_path / "synthetic_rulebook.pdf")


@pytest.fixture
def idx(tmp_path):
    return tmp_path / "idx"


@pytest.fixture
def retriever(pdf, idx):
    r = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder())
    ingest_pdf(pdf, r)
    return r


def grade(rel):
    return DocumentGrade(is_relevant=rel, confidence_score=0.9, rationale="fake")


def finding_citing(needle, quote, status=ComplianceStatus.NON_COMPLIANT, clause="Section 1"):
    """A fake auditor that cites whichever evidence block in ITS PROMPT contains `needle` (proves retrieval worked)."""
    def make(system, user):
        for cid, body in EVIDENCE.findall(user):
            if needle in " ".join(body.split()):
                return AuditFinding(policy_clause_id="x", policy_text_excerpt="x", status=status,
                                    severity=RiskSeverity.HIGH, statutory_clause=clause,
                                    statutory_verbatim_quote=quote, evidence_ids=[cid],
                                    gap_analysis="gap", remediation_action="fix")
        raise AssertionError(f"the expected evidence ({needle!r}) never reached the audit prompt")
    return make


def graph_for(retriever, llm, search=None):
    return create_audit_graph(retriever, llm=llm, search=search or FakeSearch([]))


# ------------------------------------------------------------------ the generated corpus
def test_synthetic_pdf_ingests_with_page_and_section_provenance(pdf):
    chunks = chunk_pdf(pdf)
    assert [(c.page_number, c.section_title) for c in chunks] == [
        (1, "Section 1 Log Retention"), (2, "Section 2 Data Location"),
        (3, "Section 3 Encryption"), (4, "Section 4 Access Review")]
    assert all("not real law" in c.text and c.document_name == "synthetic_rulebook.pdf" for c in chunks)
    assert all(c.source_type == SourceType.INTERNAL and c.source_url is None for c in chunks)


# ------------------------------------------------------------------ real retriever + real nodes
@pytest.mark.parametrize("clause,needle,quote,page,section,cited", [
    ("We retain access logs for 30 days.", "retained for a minimum of 90 days",
     "Access logs must be retained for a minimum of 90 days.", 1, "Section 1 Log Retention", "Section 1"),
    ("Customer personal data is stored on servers in Region Beta.", "located within Region Alpha",
     "Personal data must be stored on servers located within Region Alpha.", 2, "Section 2 Data Location", "Section 2"),
])
def test_clause_is_retrieved_audited_and_citations_verify(retriever, clause, needle, quote, page, section, cited):
    llm = FakeLLM({"DocumentGrade": [grade(True)], "AuditFinding": [finding_citing(needle, quote, clause=cited)]}, texts=["unused"])
    out = run_audit(graph_for(retriever, llm), "P-1", clause)
    assert out["retrieved_docs"][0].page_number == page and out["retrieved_docs"][0].section_title == section
    rep = out["audit_report"]
    assert out["is_grounded"] is True and rep.status == ComplianceStatus.NON_COMPLIANT
    assert rep.evidence_ids == [out["retrieved_docs"][0].id]
    assert llm.count("DocumentGrade") == 1 and out["retry_count"] == 0


def test_irrelevant_evidence_takes_the_bounded_retry_then_fallback_path_and_ends_insufficient(retriever, monkeypatch):
    sizes, real = [], retriever.retrieve
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=4: (lambda r: (sizes.append(len(r)), r)[1])(real(q, top_k)))
    llm = FakeLLM({"DocumentGrade": [grade(False)]}, texts=["rewritten query one", "rewritten query two"])
    search = FakeSearch([])
    out = run_audit(graph_for(retriever, llm, search), "P-1", "Our cafeteria serves vegetarian meals on Fridays.")
    assert len(sizes) == 3 and all(n > 0 for n in sizes)  # nearest neighbours came back each time: non-empty but irrelevant
    assert llm.count("DocumentGrade") == 4 and llm.count("text") == 2 and len(search.calls) == 1
    assert out["audit_report"].status == ComplianceStatus.INSUFFICIENT_EVIDENCE and out["retry_count"] == 2
    assert out["web_fallback_attempted"] and not out["web_fallback_used"]


def test_web_fallback_evidence_is_graded_cited_and_verified_alongside_the_real_index(retriever, monkeypatch):
    calls = []
    real = retriever.retrieve
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=4: (calls.append(q), real(q, top_k))[1])
    llm = FakeLLM({"DocumentGrade": [grade(False), grade(False), grade(False), grade(True)],
                   "AuditFinding": [finding_citing("Providers must keep logs for 90 days", WEB_RES["content"], clause="Web Search")]},
                  texts=["access log retention", "log storage period"])
    out = run_audit(graph_for(retriever, llm, FakeSearch([WEB_RES])), "P-1", "We retain access logs for 30 days.")
    assert len(calls) == 3 and llm.count("DocumentGrade") == 4
    assert out["is_grounded"] is True and out["web_fallback_used"] is True
    cited = next(d for d in out["retrieved_docs"] if d.id == out["audit_report"].evidence_ids[0])
    assert cited.id == web_chunk_id(WEB_RES["url"], WEB_RES["content"])
    assert (cited.source_type, cited.page_number, cited.source_url) == (SourceType.WEB, None, WEB_RES["url"])
    assert any(d.source_type == SourceType.INTERNAL for d in out["retrieved_docs"])  # local evidence kept alongside


def test_fabricated_citation_against_the_real_retriever_fails_closed(retriever):
    def fabricated(system, user):
        return AuditFinding(policy_clause_id="x", policy_text_excerpt="x", status=ComplianceStatus.NON_COMPLIANT,
                            severity=RiskSeverity.HIGH, statutory_clause="Section 1",
                            statutory_verbatim_quote="Access logs must be retained for 365 days.",
                            evidence_ids=[chunk_pdf_ids(retriever)[0]], gap_analysis="g", remediation_action="r")
    llm = FakeLLM({"DocumentGrade": [grade(True)], "AuditFinding": [fabricated]}, texts=["unused"])
    out = run_audit(graph_for(retriever, llm), "P-1", "We retain access logs for 30 days.")
    rep = out["audit_report"]
    assert out["is_grounded"] is False and rep.status == ComplianceStatus.INSUFFICIENT_EVIDENCE
    assert rep.evidence_ids == [] and rep.statutory_verbatim_quote is None


def chunk_pdf_ids(retriever):
    return [c.id for c in retriever.retrieve("access logs retained days", top_k=4)]


def test_audit_still_works_after_a_restart_from_disk(pdf, idx):
    r = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder())
    ingest_pdf(pdf, r)
    fresh_process()
    emb = HashingEmbedder()
    r2 = HybridRetriever(persist_directory=str(idx), embedder=emb)
    llm = FakeLLM({"DocumentGrade": [grade(True)], "AuditFinding": [
        finding_citing("retained for a minimum of 90 days", "Access logs must be retained for a minimum of 90 days.")]}, texts=["x"])
    out = run_audit(graph_for(r2, llm), "P-1", "We retain access logs for 30 days.")
    assert out["is_grounded"] is True and emb.doc_calls == 0  # nothing re-embedded after the restart


def test_ensure_ready_reports_missing_index_and_sees_later_ingestion(pdf, idx):
    r = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder())
    with pytest.raises(RetrievalIndexError, match="Index documents first"):
        r.ensure_ready()
    ingest_pdf(pdf, r)
    assert r.ensure_ready() is r and r.count() == 4


# ------------------------------------------------------------------ ingest command line
def test_cli_ingests_a_pdf_and_a_folder(pdf, idx, capsys):
    assert ingest_main([str(pdf.parent), "--persist-dir", str(idx)], embedder=HashingEmbedder()) == 0
    out = capsys.readouterr().out
    assert "read    synthetic_rulebook.pdf: 4 chunk(s)" in out and "the index now holds 4" in out
    fresh_process()
    assert HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder()).count() == 4


def test_cli_reingesting_the_same_file_does_not_re_embed(pdf, idx, capsys):
    emb = HashingEmbedder()
    assert ingest_main([str(pdf), "--persist-dir", str(idx)], embedder=emb) == 0
    before = emb.texts_embedded
    assert ingest_main([str(pdf), "--persist-dir", str(idx)], embedder=emb) == 0
    assert emb.texts_embedded == before


def test_cli_reports_bad_files_and_indexes_the_good_ones(pdf, idx, tmp_path, capsys):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    code = ingest_main([str(pdf), str(bad), str(tmp_path / "missing.pdf"), "--persist-dir", str(idx)], embedder=HashingEmbedder())
    out = capsys.readouterr().out
    assert code == 1 and out.count("FAILED") == 2 and "corrupt" in out and "not found" in out
    fresh_process()
    assert HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder()).count() == 4


def test_cli_replace_is_refused_when_any_file_failed(pdf, idx, tmp_path, capsys):
    assert ingest_main([str(pdf), "--persist-dir", str(idx)], embedder=HashingEmbedder()) == 0
    capsys.readouterr()
    code = ingest_main([str(tmp_path / "missing.pdf"), "--replace", "--persist-dir", str(idx)], embedder=HashingEmbedder())
    assert code == 1 and "Not replacing" in capsys.readouterr().out
    fresh_process()
    assert HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder()).count() == 4


def test_cli_usage_and_rebuild(pdf, idx, capsys):
    assert ingest_main(["--persist-dir", str(idx)], embedder=HashingEmbedder()) == 2
    assert ingest_main(["--rebuild", "--persist-dir", str(idx)], embedder=HashingEmbedder()) == 1
    assert "nothing to rebuild" in capsys.readouterr().out
    assert ingest_main([str(pdf), "--persist-dir", str(idx)], embedder=HashingEmbedder()) == 0
    (idx / "auditagent_evidence.manifest.json").unlink()
    fresh_process()
    assert ingest_main(["--rebuild", "--persist-dir", str(idx)], embedder=HashingEmbedder()) == 0
    assert "Rebuilt the index" in capsys.readouterr().out