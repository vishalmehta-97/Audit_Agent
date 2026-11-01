"""Deterministic fake embedders so retrieval tests need no model download or network."""
import hashlib
import math
from typing import Dict, List, Optional, Sequence

from src.retriever import tokenize


class HashingEmbedder:
    """Bag-of-words hashing embedder: texts sharing tokens get similar vectors. Counts its calls."""

    def __init__(self, dim: int = 64, model_name: str = "hashing-fake", fail_after: Optional[int] = None):
        self.dim, self.model_name, self.fail_after = dim, model_name, fail_after
        self.doc_calls = self.query_calls = self.texts_embedded = 0

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            v[0], norm = 1.0, 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.doc_calls += 1
        if self.fail_after is not None and self.doc_calls > self.fail_after:
            raise RuntimeError("embedding service unavailable")
        self.texts_embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.query_calls += 1
        return self._vec(text)


class MapEmbedder:
    """Returns hand-picked vectors so a test controls the dense ranking exactly."""

    def __init__(self, mapping: Dict[str, List[float]], model_name: str = "map-fake"):
        self.mapping, self.model_name = mapping, model_name

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [list(self.mapping[t]) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return list(self.mapping[text])