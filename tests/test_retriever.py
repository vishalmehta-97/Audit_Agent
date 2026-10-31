"""Retriever tests: pure building blocks, then HybridRetriever (real ChromaDB in temp dirs, fake embedder)."""
import inspect
import json
import logging
import typing

import chromadb
import pytest

from src.retriever import (Bm25Ranker, FastEmbedEmbedder, HybridRetriever, candidate_count, corpus_revision,
                           index_text, rrf_fuse, tokenize)
from src.retriever_interface import BaseRegulatoryRetriever
from src.schemas import EvidenceChunk, SourceType
from tests.fake_embedder import HashingEmbedder, MapEmbedder
from tests.mocks.mock_retriever import MockRegulatoryRetriever


# ------------------------------------------------------------------ tokenizer
def test_legal_identifiers_are_preserved_with_family_prefixes():
    assert tokenize("Section 14.2(b)(ii) applies") == ["section", "14.2(b)(ii)", "14.2(b)", "14.2", "applies"]
    t = tokenize("see 14.2(b).")
    assert "14.2(b)" in t and "14.2" in t and "14.2." not in t


def test_dotted_numbers_expand_but_never_to_a_bare_digit():
    t = tokenize("clause 4.2.1")
    assert "4.2.1" in t and "4.2" in t and "4" not in t


def test_hyphenated_words_and_identifiers():
    assert tokenize("log-retention") == ["log-retention", "log", "retention"]
    assert tokenize("Synthetic Rule R-1:") == ["synthetic", "rule", "r-1"]
    assert tokenize("2024-01-05") == ["2024-01-05"]


def test_case_punctuation_unicode_and_empty():
    assert tokenize("Logs, retained; for 90 DAYS.") == ["logs", "retained", "for", "90", "days"]
    assert tokenize("Donn\u00e9es personnelles") == ["donn\u00e9es", "personnelles"]
    assert tokenize("") == [] and tokenize("   ...  ") == [] and tokenize(None) == []
    assert tokenize("section 3 (b)") == ["section", "3", "b"]  # a detached '(b)' is just 'b'


def test_no_stopword_removal_and_duplicates_kept():
    assert tokenize("the the cat") == ["the", "the", "cat"]


def test_tokenizer_is_deterministic():
    s = "Section 14.2(b): logs must be kept 90 days."
    assert tokenize(s) == tokenize(s)


# ------------------------------------------------------------------ RRF
def test_rrf_exact_math_and_order():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63) and fused["d"] == pytest.approx(1 / 62)
    assert [i for i, _ in rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60)] == ["b", "a", "d", "c"]


def test_rrf_ranks_are_one_indexed():
    assert rrf_fuse([["only"]], k=60) == [("only", pytest.approx(1 / 61))]
    assert rrf_fuse([["only"]], k=1) == [("only", pytest.approx(0.5))]


def test_item_in_one_list_gets_a_single_contribution():
    fused = dict(rrf_fuse([["a", "b"], ["c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62) and fused["c"] == pytest.approx(1 / 61)


def test_ties_break_by_id_ascending_regardless_of_list_order():
    assert [i for i, _ in rrf_fuse([["z"], ["a"]], k=60)] == ["a", "z"]
    assert [i for i, _ in rrf_fuse([["a"], ["z"]], k=60)] == ["a", "z"]


def test_duplicates_within_a_list_are_dropped_before_ranking():
    fused = dict(rrf_fuse([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61) and fused["b"] == pytest.approx(1 / 62)


def test_rrf_empty_inputs_top_n_validation_and_purity():
    assert rrf_fuse([], k=60) == [] and rrf_fuse([[], []], k=60) == []
    assert [i for i, _ in rrf_fuse([[], ["x", "y"]], k=60)] == ["x", "y"]
    assert [i for i, _ in rrf_fuse([["a", "b", "c"]], k=60, top_n=2)] == ["a", "b"]
    assert rrf_fuse([["a"]], k=60, top_n=0) == []
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)
    lists = (("a", "b"), ("b", "a"))
    assert rrf_fuse(lists, k=60) == rrf_fuse(lists, k=60) and lists == (("a", "b"), ("b", "a"))


def test_candidate_count_is_double_top_k_bounded_by_corpus():
    assert candidate_count(4, 100) == 8 and candidate_count(4, 5) == 5
    assert candidate_count(4, 0) == 0 and candidate_count(1, 1) == 1


# ------------------------------------------------------------------ BM25
def _ranker(docs):
    return Bm25Ranker([i for i, _ in docs], [tokenize(t) for _, t in docs])


DOCS = [("d1", "Access logs must be retained for a minimum of 90 days."),
        ("d2", "Personal data must be stored on servers within Region Alpha."),
        ("d3", "Personal data must be encrypted at rest using an approved algorithm.")]


def test_bm25_ranks_the_relevant_document_first():
    hits = _ranker(DOCS).rank(tokenize("access logs retained days"), limit=3)
    assert hits[0][0] == "d1" and [h[0] for h in hits] == ["d1"]  # only d1 shares tokens
    hits = _ranker(DOCS).rank(tokenize("personal data encrypted"), limit=3)
    assert [h[0] for h in hits][:2] == ["d3", "d2"]
    assert all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1))


def test_bm25_no_overlap_returns_empty():
    assert _ranker(DOCS).rank(tokenize("cafeteria vegetarian"), limit=5) == []


def test_bm25_single_document_corpus_still_matches():
    # Okapi idf can be <= 0 here; the document must still be returned
    assert [h[0] for h in _ranker([("only", "logs kept 90 days")]).rank(tokenize("logs"), 3)] == ["only"]


def test_bm25_empty_corpus_empty_query_and_limit():
    assert Bm25Ranker([], []).rank(["x"], 3) == [] and len(Bm25Ranker([], [])) == 0
    assert _ranker(DOCS).rank([], 3) == [] and _ranker(DOCS).rank(["personal"], 0) == []
    assert len(_ranker(DOCS).rank(tokenize("personal data"), limit=1)) == 1


def test_bm25_ties_are_deterministic_by_id():
    r = _ranker([("b", "same text here"), ("a", "same text here"), ("c", "different words")])
    assert [h[0] for h in r.rank(tokenize("same text"), 5)] == ["a", "b"]


def test_bm25_documents_with_no_tokens_do_not_crash():
    r = Bm25Ranker(["x", "y"], [[], tokenize("logs kept")])
    assert [h[0] for h in r.rank(["logs"], 5)] == ["y"]
    assert Bm25Ranker(["x"], [[]]).rank(["logs"], 5) == []


def test_bm25_length_mismatch_rejected():
    with pytest.raises(ValueError):
        Bm25Ranker(["a", "b"], [["x"]])


def test_identifier_queries_exact_subclause_beats_sibling_and_family_query_finds_all():
    docs = [("b", "Retention rule 14.2(b) logs ninety days"), ("c", "Retention rule 14.2(c) logs sixty days"),
            ("plain", "Retention rule 14.2 general text")]
    r = _ranker(docs)
    assert r.rank(tokenize("14.2(b)"), 3)[0][0] == "b"
    hits = dict(r.rank(tokenize("14.2(b)"), 3))
    assert set(hits) == {"b", "c", "plain"}            # all share the family token 14.2
    assert hits["b"] > hits["c"] and hits["b"] > hits["plain"]  # but the exact sub-clause scores strictly higher
    assert {h[0] for h in r.rank(tokenize("14.2"), 5)} == {"b", "c", "plain"}
    assert "c" not in [h[0] for h in r.rank(tokenize("14.2(b)"), 1)]


# =====================================================================================================
# HybridRetriever
# =====================================================================================================
DOC = "SYNTHETIC RULEBOOK (not real law)"


def mk(cid, text, section="GENERAL", page=1, doc=DOC, **kw):
    return EvidenceChunk(id=cid, document_name=doc, page_number=page, section_title=section, text=text, **kw)


CHUNKS = [
    mk("r1", "Access logs must be retained for a minimum of 90 days.", "Synthetic Rule R-1 Log Retention", 1),
    mk("r2", "Personal data must be stored on servers located within Region Alpha.", "Synthetic Rule R-2 Data Location", 2),
    mk("r3", "Personal data must be encrypted at rest using an approved algorithm.", "Synthetic Rule R-3 Encryption", 3),
    mk("r4", "Access rights must be reviewed periodically.", "Synthetic Rule R-4 Access Review", 4),
    mk("w1", "Providers must keep logs for 90 days.", "Web Search Fallback", None, doc="Example Regulator",
       source_type=SourceType.WEB, source_url="https://example.org/rule"),
]


def fresh_process():
    """Make chromadb forget its in-process clients so a new HybridRetriever really reloads from disk."""
    chromadb.api.client.SharedSystemClient.clear_system_cache()


@pytest.fixture
def idx(tmp_path):
    return tmp_path / "idx"


def build(idx, chunks=CHUNKS, embedder=None, **kw):
    r = HybridRetriever(persist_directory=str(idx), embedder=embedder or HashingEmbedder(), **kw)
    r.index_documents(chunks)
    return r


def ids(chunks):
    return [c.id for c in chunks]


# ------------------------------------------------------------------ interface parity / lifecycle errors
def test_real_and_mock_retrievers_share_the_interface(idx):
    r = build(idx)
    assert isinstance(r, BaseRegulatoryRetriever) and isinstance(MockRegulatoryRetriever([]), BaseRegulatoryRetriever)
    a, b = HybridRetriever.retrieve, MockRegulatoryRetriever.retrieve
    sa, sb = inspect.signature(a), inspect.signature(b)
    assert [(p.name, p.default) for p in sa.parameters.values()] == [(p.name, p.default) for p in sb.parameters.values()]
    assert typing.get_type_hints(a) == typing.get_type_hints(b)
    out = r.retrieve("access logs", top_k=2)
    assert all(isinstance(c, EvidenceChunk) for c in out)


def test_default_embedder_is_lazy_and_needs_no_download_to_construct(idx):
    r = HybridRetriever(persist_directory=str(idx))
    assert isinstance(r._embedder, FastEmbedEmbedder) and r._embedder._model is None
    assert r._embedder.model_name == "BAAI/bge-small-en-v1.5"


def test_empty_query_and_bad_top_k_are_value_errors_even_without_an_index(idx):
    r = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder())
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError):
            r.retrieve(bad)
    with pytest.raises(ValueError):
        r.retrieve("logs", top_k=0)


def test_retrieve_without_an_index_fails_with_actionable_error(idx, caplog):
    r = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder())
    with pytest.raises(RuntimeError, match="Index documents first"):
        r.retrieve("logs")
    with caplog.at_level(logging.WARNING):
        r.index_documents([])  # no-op: still no index
    assert "nothing was changed" in caplog.text
    with pytest.raises(RuntimeError, match="Index documents first"):
        r.retrieve("logs")
    with pytest.raises(RuntimeError, match="nothing to rebuild"):
        r.rebuild()


def test_collection_name_validation(idx):
    with pytest.raises(ValueError):
        HybridRetriever(persist_directory=str(idx), collection_name="a b", embedder=HashingEmbedder())


# ------------------------------------------------------------------ search behaviour
def test_relevant_chunk_ranks_first_with_provenance_intact(idx):
    r = build(idx)
    top = r.retrieve("access logs retained minimum days", top_k=3)
    assert top[0].id == "r1" and top[0].page_number == 1 and top[0].document_name == DOC
    assert top[0].source_type == SourceType.INTERNAL and top[0].source_url is None
    assert top[0].text == CHUNKS[0].text


def test_identifier_query_finds_the_exact_subclause(idx):
    chunks = [mk("b", "Retention rule 14.2(b) logs ninety days"), mk("c", "Retention rule 14.2(c) logs sixty days"),
              mk("o", "Unrelated cafeteria menu text"), mk("p", "Encryption requirement 9.1 for stored data")]
    assert build(idx, chunks).retrieve("14.2(b)", top_k=1)[0].id == "b"


def test_top_k_bound_uniqueness_and_descending_scores(idx):
    r = build(idx)
    for k in (1, 2, 3, 5, 20):
        out = r.retrieve("personal data access logs", top_k=k)
        assert len(out) == min(k, 5) and len(set(ids(out))) == len(out)
        assert all(out[i].score >= out[i + 1].score for i in range(len(out) - 1))


def test_retrieval_is_deterministic(idx):
    r = build(idx)
    a, b = r.retrieve("personal data region", top_k=4), r.retrieve("personal data region", top_k=4)
    assert [(c.id, c.score) for c in a] == [(c.id, c.score) for c in b]


def test_query_with_no_shared_words_still_returns_nearest_neighbours_not_an_error(idx):
    out = build(idx).retrieve("cafeteria vegetarian meals", top_k=3)
    assert 1 <= len(out) <= 3 and all(isinstance(c, EvidenceChunk) for c in out)


def test_exact_rrf_fusion_through_the_retriever(idx):
    chunks = [mk("A", "alpha one"), mk("B", "zeta zeta two"), mk("C", "zeta"), mk("D", "other words")]
    vec = {"alpha one": [1.0, 0, 0], "zeta zeta two": [0.8, 0.6, 0], "zeta": [0, 1.0, 0], "other words": [0, 0, 1.0]}
    r = build(idx, chunks, embedder=MapEmbedder({**vec, "zeta query": [1.0, 0, 0]}))
    dense, sparse = r._dense_rank("zeta query", 4), r._sparse_rank("zeta query", 4)
    assert dense == ["A", "B", "C", "D"] and set(sparse) == {"B", "C"}  # B and C are the only 'zeta' chunks
    expected = dict(rrf_fuse([dense, sparse], k=60))
    out = r.retrieve("zeta query", top_k=4)
    assert {c.id: c.score for c in out} == pytest.approx(expected)
    assert ids(out) == [cid for cid, _ in rrf_fuse([dense, sparse], k=60)]
    assert out[0].id in ("B", "C")  # in both lists, so it beats the dense-only 'A'
    only_dense = next(c for c in out if c.id == "A")
    assert only_dense.score == pytest.approx(1 / 61)  # one contribution only


def test_candidate_pool_is_top_k_times_two_bounded_by_corpus(idx, monkeypatch):
    r = build(idx)
    seen = []
    orig_dense, orig_rank = r._dense_rank, Bm25Ranker.rank
    monkeypatch.setattr(r, "_dense_rank", lambda q, n: (seen.append(("dense", n)), orig_dense(q, n))[1])
    monkeypatch.setattr(Bm25Ranker, "rank", lambda self, q, limit: (seen.append(("sparse", limit)), orig_rank(self, q, limit))[1])
    r.retrieve("access logs", top_k=2)
    r.retrieve("access logs", top_k=4)
    assert seen == [("dense", 4), ("sparse", 4), ("dense", 5), ("sparse", 5)]


def test_rrf_k_comes_from_settings_or_constructor(idx, monkeypatch):
    r = build(idx)
    base = r.retrieve("access logs", top_k=1)[0].score
    monkeypatch.setattr("src.retriever.settings.RRF_K", 1)
    assert r.retrieve("access logs", top_k=1)[0].score > base
    assert build(idx.parent / "idx2", rrf_k=1).retrieve("access logs", top_k=1)[0].score > base


# ------------------------------------------------------------------ metadata / persistence
def test_metadata_is_stored_and_reconstructed_including_optional_fields(idx):
    r = build(idx)
    col = r._client.get_collection("auditagent_evidence")
    got = col.get(ids=["w1", "r1"], include=["metadatas", "documents"])
    meta = dict(zip(got["ids"], got["metadatas"]))
    assert meta["w1"] == {"document_name": "Example Regulator", "section_title": "Web Search Fallback",
                          "source_type": "WEB", "source_url": "https://example.org/rule"}  # no page_number key
    assert meta["r1"]["page_number"] == 1 and "source_url" not in meta["r1"]
    assert dict(zip(got["ids"], got["documents"]))["r1"] == CHUNKS[0].text
    web = next(c for c in r.retrieve("providers keep logs", top_k=5) if c.id == "w1")
    assert (web.page_number, web.source_url, web.source_type) == (None, "https://example.org/rule", SourceType.WEB)


def test_restart_reloads_from_disk_and_gives_identical_results(idx):
    r = build(idx)
    first = r.retrieve("access logs retained days", top_k=3)
    fresh_process()
    emb = HashingEmbedder()
    r2 = HybridRetriever(persist_directory=str(idx), embedder=emb)
    second = r2.retrieve("access logs retained days", top_k=3)
    assert [(c.id, c.score) for c in first] == [(c.id, c.score) for c in second]
    assert emb.doc_calls == 0 and emb.query_calls == 1  # nothing re-embedded; BM25 rebuilt from the stored corpus
    assert r2.count() == 5


def test_manifest_is_written_and_matches_the_corpus(idx):
    build(idx)
    m = json.loads((idx / "auditagent_evidence.manifest.json").read_text(encoding="utf-8"))
    assert m["count"] == 5 and m["embedding_model"] == "hashing-fake" and m["embedding_dim"] == 64
    assert m["revision"] == corpus_revision(CHUNKS) == corpus_revision(list(reversed(CHUNKS)))


# ------------------------------------------------------------------ upsert / replace / rebuild semantics
def test_upsert_replaces_a_changed_chunk_and_keeps_the_rest(idx):
    r = build(idx)
    rev0 = json.loads((idx / "auditagent_evidence.manifest.json").read_text())["revision"]
    r.index_documents([mk("r1", "Access logs must be retained for 365 days.", "Synthetic Rule R-1 Log Retention", 1)])
    assert r.count() == 5
    top = r.retrieve("access logs retained 365 days", top_k=1)[0]
    assert top.id == "r1" and "365" in top.text
    col = r._client.get_collection("auditagent_evidence")
    assert "365" in col.get(ids=["r1"], include=["documents"])["documents"][0]
    assert json.loads((idx / "auditagent_evidence.manifest.json").read_text())["revision"] != rev0
    assert {"r2", "r3", "r4", "w1"} <= set(ids(r.retrieve("personal data access providers", top_k=5)))


def test_reindexing_identical_chunks_is_a_no_op_that_embeds_nothing(idx):
    emb = HashingEmbedder()
    r = build(idx, embedder=emb)
    before, rev = emb.texts_embedded, (idx / "auditagent_evidence.manifest.json").read_bytes()
    r.index_documents(CHUNKS)
    assert emb.texts_embedded == before and (idx / "auditagent_evidence.manifest.json").read_bytes() == rev


def test_duplicate_ids_in_one_call_last_wins_with_warning(idx, caplog):
    r = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder())
    with caplog.at_level(logging.WARNING):
        r.index_documents([mk("x", "first version text"), mk("x", "second version text"), mk("y", "other chunk")])
    assert "duplicate chunk id" in caplog.text and r.count() == 2
    assert next(c for c in r.retrieve("second version", top_k=2) if c.id == "x").text == "second version text"


def test_replace_and_reindex_drop_old_chunks(idx):
    r = build(idx)
    r.reindex([mk("only", "A single replacement chunk about logs.")])
    assert r.count() == 1 and ids(r.retrieve("access logs personal", top_k=5)) == ["only"]
    fresh_process()
    assert HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder()).count() == 1


def test_non_chunk_input_is_rejected(idx):
    with pytest.raises(TypeError):
        HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder()).index_documents(["not a chunk"])


# ------------------------------------------------------------------ stale / partial index handling
def test_missing_manifest_or_corpus_is_reported_as_partial(idx):
    build(idx)
    (idx / "auditagent_evidence.manifest.json").unlink()
    fresh_process()
    with pytest.raises(RuntimeError, match="partially initialized"):
        HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder()).retrieve("logs")
    r = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder())
    r.rebuild()  # the canonical corpus survived, so the manifest can be regenerated
    assert r.count() == 5 and r.retrieve("access logs", top_k=1)[0].id == "r1"
    (idx / "auditagent_evidence.corpus.json").unlink()
    r.refresh()
    with pytest.raises(RuntimeError, match="partially initialized"):
        r.retrieve("logs")


def test_tampered_corpus_is_detected_as_stale_and_rebuild_recovers(idx):
    r = build(idx)
    path = idx / "auditagent_evidence.corpus.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["chunks"][0]["text"] = "Changed behind the manifest's back."
    path.write_text(json.dumps(data), encoding="utf-8")
    r.refresh()
    with pytest.raises(RuntimeError, match="stale"):
        r.retrieve("logs")
    r.rebuild()
    assert "Changed behind" in next(c for c in r.retrieve("changed manifest back", top_k=5) if c.id == data["chunks"][0]["id"]).text


def test_corrupt_files_give_clear_errors(idx):
    build(idx)
    (idx / "auditagent_evidence.corpus.json").write_text("{not json", encoding="utf-8")
    fresh_process()
    with pytest.raises(RuntimeError, match="unreadable or corrupt"):
        HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder()).retrieve("logs")


def test_dense_index_missing_an_item_is_detected_and_rebuild_restores_it(idx):
    r = build(idx)
    r._client.get_collection("auditagent_evidence").delete(ids=["r1"])
    r.refresh()
    with pytest.raises(RuntimeError, match="differ"):
        r.retrieve("access logs")
    r.rebuild()
    assert r.retrieve("access logs retained days", top_k=1)[0].id == "r1"


def test_embedding_failure_during_indexing_changes_nothing(idx):
    r = build(idx, embedder=HashingEmbedder(fail_after=1))
    files = {n: (idx / n).read_bytes() for n in ("auditagent_evidence.corpus.json", "auditagent_evidence.manifest.json")}
    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        r.index_documents([mk("new", "A brand new chunk about retention.")])
    assert {n: (idx / n).read_bytes() for n in files} == files
    assert r.count() == 5 and r.retrieve("access logs", top_k=1)[0].id == "r1"


def test_crash_after_corpus_write_is_detected_and_rebuild_completes_the_intended_index(idx, monkeypatch):
    r = build(idx)
    monkeypatch.setattr(r, "_upsert", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        r.index_documents([mk("new", "A brand new chunk about retention periods.")])
    r.refresh()
    with pytest.raises(RuntimeError, match="rebuild"):
        r.retrieve("retention")
    with pytest.raises(RuntimeError, match="rebuild"):  # indexing on top of a broken index is refused too
        r.index_documents([mk("more", "Another chunk.")])
    monkeypatch.undo()
    r.rebuild()
    assert r.count() == 6 and r.retrieve("brand new retention periods", top_k=1)[0].id == "new"


def test_crash_before_manifest_write_is_detected(idx, monkeypatch):
    r = build(idx)
    monkeypatch.setattr(r, "_write_manifest", lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        r.index_documents([mk("new", "A brand new chunk about retention periods.")])
    r.refresh()
    with pytest.raises(RuntimeError, match="stale"):
        r.retrieve("retention")


def test_embedding_model_or_dimension_change_is_refused_with_instructions(idx):
    build(idx)
    fresh_process()
    other = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder(dim=32, model_name="other-model"))
    with pytest.raises(RuntimeError, match="rebuild"):
        other.retrieve("access logs")
    with pytest.raises(RuntimeError, match="reindex"):
        other.index_documents([mk("n", "New chunk text here.")])
    other.rebuild()  # re-embeds the stored corpus with the new model
    assert other.retrieve("access logs retained", top_k=1)[0].id == "r1"
    same_name_new_dim = HybridRetriever(persist_directory=str(idx), embedder=HashingEmbedder(dim=16, model_name="other-model"))
    with pytest.raises(RuntimeError, match="dims"):
        same_name_new_dim.retrieve("access logs")


def test_index_text_includes_the_section_title_unless_general():
    assert index_text(mk("a", "body", "Section 4.2 Log Retention")) == "Section 4.2 Log Retention\nbody"
    assert index_text(mk("a", "body", "GENERAL")) == "body"