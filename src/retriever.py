"""Retrieval building blocks (pure, deterministic) + HybridRetriever (added in the next stage).

tokenize()      identical tokenizer for indexing AND querying; keeps legal identifiers such as 14.2(b)
rrf_fuse()      Reciprocal Rank Fusion over ranked ID lists
Bm25Ranker      rank-bm25 wrapper that ranks only documents sharing at least one query token
candidate_count how many candidates to take per retrieval method (top_k * 2, bounded by corpus size)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

CANDIDATE_MULTIPLIER = 2

_ALNUM = r"[^\W_]+"  # letters/digits in any script, no underscore
_TOKEN = re.compile(rf"{_ALNUM}(?:[./\-]{_ALNUM})*(?:\([^\W_]+\))*")
_TRAILING_PAREN = re.compile(r"\([^\W_]+\)$")


def tokenize(text: str) -> List[str]:
    """Lowercase tokens that preserve legal identifiers.

    '14.2(b)(ii)' yields '14.2(b)(ii)', '14.2(b)' and '14.2' so that a query for the clause family
    ('14.2') still finds sub-clauses, while a precise query ('14.2(b)') prefers the exact sub-clause.
    '4.2.1' also yields '4.2'. 'log-retention' also yields 'log' and 'retention'; 'r-1' stays whole.
    No stopword removal. Duplicates are kept (they carry term frequency).
    """
    out: List[str] = []
    for m in _TOKEN.finditer((text or "").lower()):
        tok = m.group(0)
        out.append(tok)
        base = tok
        while True:  # strip trailing (x) groups one at a time
            stripped = _TRAILING_PAREN.sub("", base)
            if stripped == base:
                break
            base = stripped
            out.append(base)
        if base[0].isdigit() and "." in base:  # 4.2.1 -> 4.2 (never a bare '4')
            parts = base.split(".")
            for k in range(len(parts) - 1, 1, -1):
                out.append(".".join(parts[:k]))
        if re.search(r"[\-/]", base):  # alphabetic parts of hyphenated/slashed words
            out.extend(p for p in re.split(r"[\-/]", base) if len(p) >= 2 and p.isalpha())
    return out


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], k: int, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion.

    For each unique id: sum of 1 / (k + rank) over every list containing it (1-indexed rank, each list counted
    at most once). Duplicates inside one list are dropped first (first occurrence kept), then ranks are 1..n.
    Result is sorted by descending score; ties are broken by id ascending, so output is fully deterministic.
    """
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(dict.fromkeys(lst), start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused if top_n is None else fused[:max(top_n, 0)]


def candidate_count(top_k: int, corpus_size: int) -> int:
    return max(0, min(top_k * CANDIDATE_MULTIPLIER, corpus_size))


class Bm25Ranker:
    """BM25 (Okapi) over a fixed corpus of pre-tokenized documents.

    Only documents that share at least one token with the query are returned (so 'no match' really means no
    match). The BM25 score is used for ORDERING only: in tiny corpora Okapi idf can be zero or negative, so
    do not treat the number as a probability or compare it across queries. Ties are broken by id ascending.
    """

    def __init__(self, doc_ids: Sequence[str], corpus_tokens: Sequence[Sequence[str]]):
        if len(doc_ids) != len(corpus_tokens):
            raise ValueError("doc_ids and corpus_tokens must have the same length")
        self._ids = list(doc_ids)
        self._token_sets = [set(t) for t in corpus_tokens]
        self._bm25 = BM25Okapi([list(t) for t in corpus_tokens]) if any(corpus_tokens) else None

    def __len__(self) -> int:
        return len(self._ids)

    def rank(self, query_tokens: Sequence[str], limit: int) -> List[Tuple[str, float]]:
        if self._bm25 is None or not query_tokens or limit < 1:
            return []
        q = set(query_tokens)
        scores = self._bm25.get_scores(list(query_tokens))
        hits = [(self._ids[i], float(scores[i])) for i, ts in enumerate(self._token_sets) if ts & q]
        hits.sort(key=lambda h: (-h[1], h[0]))
        return hits[:limit]