"""Retriever contract shared by HybridRetriever and MockRegulatoryRetriever.

retrieve(): empty/whitespace query -> ValueError; at most top_k unique chunks,
sorted by descending score; no matches -> []; no LLM or web calls.
"""
from abc import ABC, abstractmethod
from typing import List

from src.schemas import EvidenceChunk


class BaseRegulatoryRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> List[EvidenceChunk]:
        ...