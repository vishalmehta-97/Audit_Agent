"""Retrieval: pure building blocks + HybridRetriever (dense Chroma + sparse BM25, fused with RRF).

tokenize()      identical tokenizer for indexing AND querying; keeps legal identifiers such as 14.2(b)
rrf_fuse()      Reciprocal Rank Fusion over ranked ID lists
Bm25Ranker      rank-bm25 wrapper that ranks only documents sharing at least one query token
candidate_count how many candidates to take per retrieval method (top_k * 2, bounded by corpus size)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from src.config import settings
from src.retriever_interface import BaseRegulatoryRetriever
from src.schemas import EvidenceChunk

logger = logging.getLogger(__name__)

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


# =====================================================================================================
# HybridRetriever
# =====================================================================================================
SCHEMA_VERSION = 1
DEFAULT_COLLECTION = "auditagent_evidence"
_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")


class Embedder(Protocol):
    """Anything with these members can embed text (FastEmbed in production, a fake in tests)."""
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


class FastEmbedEmbedder:
    """fastembed.TextEmbedding, loaded lazily. The FIRST use downloads the model (needs internet)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:
                raise RuntimeError(f"Could not load embedding model {self.model_name!r}. The first use downloads it, "
                                   "so check your internet connection and the model name.") from exc
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [[float(x) for x in v] for v in self._load().passage_embed(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return [float(x) for x in next(iter(self._load().query_embed(text)))]


def index_text(chunk: EvidenceChunk) -> str:
    """Text used for BOTH dense and sparse indexing: the section title (unless GENERAL) plus the chunk text."""
    return chunk.text if chunk.section_title in ("", "GENERAL") else f"{chunk.section_title}\n{chunk.text}"


def _chunk_dict(c: EvidenceChunk) -> Dict[str, Any]:
    return c.model_dump(mode="json", exclude={"score"})  # score is per-query, never persisted


def corpus_revision(chunks: Iterable[EvidenceChunk]) -> str:
    """Deterministic fingerprint of a corpus: same chunks (any order) -> same revision."""
    payload = json.dumps([_chunk_dict(c) for c in sorted(chunks, key=lambda c: c.id)],
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chroma_metadata(c: EvidenceChunk) -> Dict[str, Any]:
    md: Dict[str, Any] = {"document_name": c.document_name, "section_title": c.section_title,
                          "source_type": c.source_type.value}
    if c.page_number is not None:  # Chroma metadata cannot hold None: optional fields are omitted
        md["page_number"] = c.page_number
    if c.source_url is not None:
        md["source_url"] = c.source_url
    return md


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class HybridRetriever(BaseRegulatoryRetriever):
    """Dense (ChromaDB cosine) + sparse (BM25) retrieval fused with Reciprocal Rank Fusion.

    Files under persist_directory:
        chroma/                          ChromaDB data (dense index)
        <collection>.corpus.json         CANONICAL chunks (source of truth; BM25 is rebuilt from it at load)
        <collection>.manifest.json       revision/count/model/dim; written LAST, only after a successful index

    Index semantics
      * index_documents(docs)              upsert by chunk id (a duplicate id inside one call: the last one wins);
                                           identical chunks are skipped; refuses to run on an inconsistent index.
      * index_documents(docs, replace=True) / reindex(docs)   replace the whole index.
      * rebuild()                          deterministic recovery: re-embed the canonical corpus file.
      * Write order: embed first (a failure changes nothing) -> corpus file -> Chroma -> manifest. A crash in between
        leaves manifest != corpus, which is detected and reported with instructions to call rebuild().
    Retrieval returns the nearest neighbours even when the query shares no words with the corpus (dense search has no
    notion of 'no match'); deciding relevance is the grader's job. A never-indexed retriever raises RuntimeError.
    The index is validated when first used; call refresh() if another process may have changed it.
    """

    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = DEFAULT_COLLECTION,
                 embedder: Optional[Embedder] = None, embedding_model_name: Optional[str] = None,
                 rrf_k: Optional[int] = None, batch_size: int = 128):
        if not _COLLECTION_NAME.match(collection_name):
            raise ValueError("collection_name must be 3-512 characters: letters, digits, '.', '_' or '-'")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._dir = Path(persist_directory or settings.CHROMA_PERSIST_DIR)
        self._name = collection_name
        self._embedder = embedder or FastEmbedEmbedder(embedding_model_name or settings.EMBEDDING_MODEL_NAME)
        self._rrf_k = rrf_k
        self._batch = batch_size
        self._corpus_path = self._dir / f"{collection_name}.corpus.json"
        self._manifest_path = self._dir / f"{collection_name}.manifest.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._dir / "chroma"),
                                                 settings=ChromaSettings(anonymized_telemetry=False))
        self._state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ small helpers
    def _model_name(self) -> str:
        return getattr(self._embedder, "model_name", type(self._embedder).__name__)

    def refresh(self) -> None:
        """Forget cached state; the next call re-validates the files on disk."""
        self._state = None

    def count(self) -> int:
        return len(self._load_state()["corpus"])

    def _read_manifest(self) -> Dict[str, Any]:
        try:
            m = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if m.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported manifest version")
            return m
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"The index manifest {self._manifest_path.name} is unreadable. "
                               "Call rebuild() or index the documents again.") from exc

    def _read_corpus(self) -> Dict[str, EvidenceChunk]:
        try:
            data = json.loads(self._corpus_path.read_text(encoding="utf-8"))
            if data.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported corpus version")
            chunks = [EvidenceChunk.model_validate(d) for d in data["chunks"]]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"The stored corpus {self._corpus_path.name} is unreadable or corrupt; "
                               "index the documents again.") from exc
        return {c.id: c for c in sorted(chunks, key=lambda c: c.id)}

    def _write_corpus(self, corpus: Dict[str, EvidenceChunk]) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "chunks": [_chunk_dict(c) for c in corpus.values()]}
        _atomic_write(self._corpus_path, json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _write_manifest(self, revision: str, count: int, model: str, dim: int) -> None:
        _atomic_write(self._manifest_path, json.dumps(
            {"schema_version": SCHEMA_VERSION, "collection": self._name, "revision": revision, "count": count,
             "embedding_model": model, "embedding_dim": dim}, sort_keys=True))

    # ------------------------------------------------------------------ validation / loading
    def _load_state(self) -> Dict[str, Any]:
        if self._state is not None:
            return self._state
        has_m, has_c = self._manifest_path.exists(), self._corpus_path.exists()
        if not has_m and not has_c:
            raise RuntimeError(f"No index found in {self._dir}. Index documents first "
                               "(HybridRetriever.index_documents, or ingest_pdf).")
        if has_m != has_c:
            raise RuntimeError("The index is partially initialized (the manifest or the corpus file is missing). "
                               "Call rebuild() if the corpus file exists; otherwise index the documents again.")
        manifest, corpus = self._read_manifest(), self._read_corpus()
        if manifest.get("revision") != corpus_revision(corpus.values()) or manifest.get("count") != len(corpus):
            raise RuntimeError("The index is stale or was only partly written (manifest and corpus disagree). "
                               "Call rebuild() to regenerate the dense index from the stored corpus.")
        try:
            collection = self._client.get_collection(self._name)
        except Exception as exc:
            raise RuntimeError("The dense index is missing. Call rebuild() to regenerate it from the stored "
                               "corpus.") from exc
        dense_ids = set(collection.get(include=[])["ids"])
        if dense_ids != set(corpus):
            raise RuntimeError(f"The dense index and the stored corpus differ ({len(dense_ids)} vs {len(corpus)} "
                               "chunks). Call rebuild().")
        ranker = Bm25Ranker(list(corpus), [tokenize(index_text(c)) for c in corpus.values()])
        self._state = {"manifest": manifest, "corpus": corpus, "collection": collection, "ranker": ranker}
        return self._state

    # ------------------------------------------------------------------ indexing
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._batch):
            vectors.extend(self._embedder.embed_documents(texts[i:i + self._batch]))
        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            raise RuntimeError("The embedder returned the wrong number of vectors or empty vectors.")
        if len({len(v) for v in vectors}) != 1:
            raise RuntimeError("The embedder returned vectors of different sizes.")
        return vectors

    def _upsert(self, collection, chunks: List[EvidenceChunk], vectors: List[List[float]]) -> None:
        for i in range(0, len(chunks), self._batch):
            part = chunks[i:i + self._batch]
            collection.upsert(ids=[c.id for c in part], embeddings=vectors[i:i + self._batch],
                              documents=[c.text for c in part], metadatas=[_chroma_metadata(c) for c in part])

    def _fresh_collection(self):
        try:
            self._client.delete_collection(self._name)
        except Exception:  # does not exist yet: nothing to delete
            pass
        return self._client.create_collection(self._name, configuration={"hnsw": {"space": "cosine"}})

    def index_documents(self, documents: List[EvidenceChunk], replace: bool = False) -> None:
        unique: Dict[str, EvidenceChunk] = {}
        duplicates = 0
        for c in documents:
            if not isinstance(c, EvidenceChunk):
                raise TypeError("index_documents expects EvidenceChunk objects")
            duplicates += c.id in unique
            unique[c.id] = c  # last one wins
        if duplicates:
            logger.warning("index_documents: %d duplicate chunk id(s) in one call; the last occurrence wins.", duplicates)
        if not unique:
            logger.warning("index_documents called with no documents; nothing was changed.")
            return

        existing: Dict[str, EvidenceChunk] = {}
        old_manifest: Optional[Dict[str, Any]] = None
        if not replace and (self._manifest_path.exists() or self._corpus_path.exists()):
            state = self._load_state()  # raises if inconsistent: never index on top of a broken index
            existing, old_manifest = dict(state["corpus"]), state["manifest"]
        changed = [c for c in unique.values()
                   if replace or c.id not in existing or _chunk_dict(existing[c.id]) != _chunk_dict(c)]
        if not changed:
            logger.info("index_documents: every chunk is already indexed unchanged.")
            return

        vectors = self._embed_texts([index_text(c) for c in changed])  # a failure here changes nothing
        model, dim = self._model_name(), len(vectors[0])
        if old_manifest and (old_manifest["embedding_model"] != model or old_manifest["embedding_dim"] != dim):
            raise RuntimeError(f"The index was built with embedding model {old_manifest['embedding_model']!r} "
                               f"({old_manifest['embedding_dim']} dims) but the current embedder is {model!r} "
                               f"({dim} dims). Use reindex(...) or rebuild() to rebuild with the new model.")
        merged = dict(sorted({**existing, **{c.id: c for c in changed}}.items()))
        self._state = None
        self._write_corpus(merged)                                   # 1. durable intent
        collection = self._fresh_collection() if replace else self._client.get_or_create_collection(
            self._name, configuration={"hnsw": {"space": "cosine"}})
        self._upsert(collection, changed, vectors)                   # 2. dense index
        self._write_manifest(corpus_revision(merged.values()), len(merged), model, dim)  # 3. commit marker
        logger.info("Indexed %d chunk(s); corpus now has %d.", len(changed), len(merged))

    def reindex(self, documents: List[EvidenceChunk]) -> None:
        """Replace the whole index with these documents."""
        self.index_documents(documents, replace=True)

    def rebuild(self) -> None:
        """Deterministic recovery: regenerate the dense index and manifest from the stored canonical corpus."""
        if not self._corpus_path.exists():
            raise RuntimeError(f"No stored corpus in {self._dir}; nothing to rebuild from. Index documents again.")
        corpus = self._read_corpus()
        if not corpus:
            raise RuntimeError("The stored corpus is empty; nothing to rebuild.")
        chunks = list(corpus.values())
        vectors = self._embed_texts([index_text(c) for c in chunks])
        self._state = None
        self._upsert(self._fresh_collection(), chunks, vectors)
        self._write_manifest(corpus_revision(corpus.values()), len(corpus), self._model_name(), len(vectors[0]))
        logger.info("Rebuilt the index from the stored corpus (%d chunks).", len(corpus))

    # ------------------------------------------------------------------ retrieval
    def _check_embedder(self, manifest: Dict[str, Any]) -> None:
        if manifest["embedding_model"] != self._model_name():
            raise RuntimeError(f"The index was built with embedding model {manifest['embedding_model']!r} but the "
                               f"current embedder is {self._model_name()!r}. Call rebuild() to re-embed.")

    def _dense_rank(self, query: str, n: int) -> List[str]:
        state = self._load_state()
        self._check_embedder(state["manifest"])
        q = self._embedder.embed_query(query)
        if len(q) != state["manifest"]["embedding_dim"]:
            raise RuntimeError(f"Query embedding has {len(q)} dims but the index uses "
                               f"{state['manifest']['embedding_dim']}. Call rebuild().")
        res = state["collection"].query(query_embeddings=[q], n_results=n, include=["distances"])
        pairs = sorted(zip(res["distances"][0], res["ids"][0]))  # (distance, id): deterministic ties
        return [cid for _, cid in pairs]

    def _sparse_rank(self, query: str, n: int) -> List[str]:
        return [cid for cid, _ in self._load_state()["ranker"].rank(tokenize(query), n)]

    def retrieve(self, query: str, top_k: int = 4) -> List[EvidenceChunk]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        state = self._load_state()
        n = candidate_count(top_k, len(state["corpus"]))
        fused = rrf_fuse([self._dense_rank(query, n), self._sparse_rank(query, n)],
                         self._rrf_k if self._rrf_k is not None else settings.RRF_K, top_n=top_k)
        return [state["corpus"][cid].model_copy(update={"score": score}) for cid, score in fused]