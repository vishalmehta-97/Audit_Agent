"""Retriever tests. This file currently covers the pure building blocks; HybridRetriever tests are added next."""
import pytest

from src.retriever import Bm25Ranker, candidate_count, rrf_fuse, tokenize


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