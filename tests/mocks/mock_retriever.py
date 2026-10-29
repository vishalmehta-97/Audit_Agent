"""Deterministic in-memory retriever for offline tests. Scores by query-token overlap."""
import re
from typing import Dict, List, Optional

from src.retriever_interface import BaseRegulatoryRetriever
from src.schemas import EvidenceChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN.findall(text.lower()))


class MockRegulatoryRetriever(BaseRegulatoryRetriever):
    def __init__(self, chunks: Optional[List[EvidenceChunk]] = None):
        self._chunks = list(chunks or [])
        self.calls: List[Dict] = []  # records every valid call, for exact call-count tests

    def retrieve(self, query: str, top_k: int = 4) -> List[EvidenceChunk]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.calls.append({"query": query, "top_k": top_k})
        q = _tokens(query)
        scored, seen = [], set()
        for c in self._chunks:
            if c.id in seen:
                continue
            overlap = len(q & _tokens(c.text + " " + c.section_title))
            if overlap > 0:
                seen.add(c.id)
                scored.append(c.model_copy(update={"score": float(overlap)}))
        scored.sort(key=lambda c: (-c.score, c.id))
        return scored[:top_k]