"""PDF ingestion: page extraction, conservative heading detection, character chunking, stable IDs.

Public API
    extract_pdf_pages(pdf_path)                      -> [{"page_number": int (1-indexed), "text": str}]
    chunk_pages(pages, document_name, content_id)    -> [EvidenceChunk]   (pure; no PDF needed)
    chunk_pdf(pdf_path)                              -> [EvidenceChunk]
    ingest_pdf(pdf_path, retriever)                  -> [EvidenceChunk]   (chunks, then retriever.index_documents)

Policies (documented, applied consistently)
  * Missing file            -> FileNotFoundError (message contains the path).
  * Corrupt / encrypted PDF -> PdfIngestionError is RAISED (never a silent empty result) and logged.
  * Pages with < MIN_PAGE_CHARS non-whitespace characters are skipped (and logged); page numbers stay the
    real 1-indexed numbers. A PDF with no extractable text returns [] and logs a warning.
  * CHUNK_SIZE / CHUNK_OVERLAP are CHARACTERS (not words).
  * Limitation: a heading that sits alone on a skipped near-empty page is not carried to the next page.
  * Chunk id = <slug of file name>-<first 8 hex of SHA-256 of the file bytes>-p<page>-c<index on that page>.
    Same bytes -> same ids; different files that share a basename get different ids.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Union

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.schemas import EvidenceChunk, SourceType

logger = logging.getLogger(__name__)

MIN_PAGE_CHARS = 20
DEFAULT_SECTION = "GENERAL"
MAX_HEADING_CHARS = 80


class PdfIngestionError(RuntimeError):
    """A PDF could not be read (corrupt, encrypted, or otherwise unusable)."""


class SupportsIndexing(Protocol):
    def index_documents(self, documents: List[EvidenceChunk], *args, **kwargs) -> None: ...


# --------------------------------------------------------------------------- headings
_HEADING = re.compile(r"^(section|clause|chapter)\s+(\d+(?:\.\d+)*(?:\([A-Za-z0-9]+\))*)(.*)$", re.IGNORECASE)


def parse_heading(line: str) -> Optional[str]:
    """Return a normalized heading for lines like 'Section 4.2 Log Retention', else None.

    Conservative on purpose: the line must be short, the identifier must end cleanly, any title must start
    with an uppercase letter or digit, and a line that reads like a sentence (ends with . , ;) is rejected,
    so body text such as 'Section 4.2 requires logs.' is NOT a heading.
    """
    s = " ".join(line.split())
    if not s or len(s) > MAX_HEADING_CHARS:
        return None
    m = _HEADING.match(s)
    if not m:
        return None
    rest = m.group(3)
    if rest and rest[0].isalnum():          # e.g. 'Section 4.2a...' -> identifier does not end cleanly
        return None
    title = rest.strip(" \t.:;-\u2013\u2014")
    if title:
        if not (title[0].isupper() or title[0].isdigit()):
            return None
        if rest.rstrip()[-1] in ".,;":
            return None
    return s.rstrip(" .:;")


def _segments(text: str, carried: str):
    """Split one page into (heading, text) segments. Heading-only segments (no body) are dropped; the
    heading still carries forward. Returns (segments, heading_in_effect_at_page_end)."""
    segs: List[tuple] = []
    heading, head_line, body = carried, None, []

    def flush():
        b = "\n".join(body).strip()
        if b:
            segs.append((heading, (head_line + "\n" if head_line else "") + b))

    for line in text.splitlines():
        h = parse_heading(line)
        if h:
            flush()
            heading, head_line, body = h, line.strip(), []
        else:
            body.append(line)
    flush()
    return segs, heading


# --------------------------------------------------------------------------- ids / chunking
def slugify_name(name: str, max_len: int = 40) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:max_len] or "doc"


def make_chunk_id(document_name: str, content_id: str, page_number: int, index: int) -> str:
    return f"{slugify_name(document_name)}-{content_id}-p{page_number}-c{index}"


def chunk_pages(pages: List[Dict], document_name: str, content_id: str,
                chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[EvidenceChunk]:
    """Chunk already-extracted pages. Size/overlap are characters; defaults come from settings at call time."""
    size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if size < 1 or overlap < 0 or overlap >= size:  # the splitter itself only rejects overlap > size
        raise ValueError(f"need chunk_size >= 1 and 0 <= chunk_overlap < chunk_size (got {size}, {overlap})")
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, length_function=len)
    chunks: List[EvidenceChunk] = []
    heading = DEFAULT_SECTION
    for page in pages:
        page_no, idx = page["page_number"], 0
        segs, heading = _segments(page["text"], heading)
        for seg_heading, seg_text in segs:
            for piece in splitter.split_text(seg_text):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(EvidenceChunk(
                    id=make_chunk_id(document_name, content_id, page_no, idx), document_name=document_name,
                    page_number=page_no, section_title=seg_heading, text=piece,
                    source_type=SourceType.INTERNAL, score=0.0, source_url=None))
                idx += 1
    return chunks


# --------------------------------------------------------------------------- PDF reading
def extract_pdf_pages(pdf_path: Union[str, Path]) -> List[Dict]:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found or not a file: {path.resolve()}")
    try:
        doc = pymupdf.open(str(path))
    except (RuntimeError, ValueError) as exc:
        logger.warning("Could not open PDF %s: %s", path, exc)
        raise PdfIngestionError(f"Could not open {path.name}: the file is corrupt or not a valid PDF.") from exc
    with doc:
        if doc.needs_pass:
            logger.warning("PDF %s is password-protected; refusing to ingest.", path)
            raise PdfIngestionError(f"{path.name} is password-protected; remove the password first.")
        if doc.page_count == 0:
            logger.warning("PDF %s has no pages; nothing to ingest.", path)
            return []
        pages, skipped = [], []
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if len(re.sub(r"\s+", "", text)) < MIN_PAGE_CHARS:
                skipped.append(i)
            else:
                pages.append({"page_number": i, "text": text})
    if not pages:
        logger.warning("PDF %s has no extractable text (scanned images?); skipped pages: %s", path, skipped)
    elif skipped:
        logger.info("PDF %s: skipped near-empty pages %s", path, skipped)
    return pages


def chunk_pdf(pdf_path: Union[str, Path], chunk_size: Optional[int] = None,
              chunk_overlap: Optional[int] = None) -> List[EvidenceChunk]:
    path = Path(pdf_path)
    pages = extract_pdf_pages(path)
    if not pages:
        return []
    content_id = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return chunk_pages(pages, path.name, content_id, chunk_size, chunk_overlap)


def ingest_pdf(pdf_path: Union[str, Path], retriever: SupportsIndexing) -> List[EvidenceChunk]:
    """Chunk a PDF and hand the chunks to retriever.index_documents. Returns the chunks (possibly [])."""
    chunks = chunk_pdf(pdf_path)
    if not chunks:
        logger.warning("Nothing to index from %s.", pdf_path)
        return []
    retriever.index_documents(chunks)
    return chunks


# --------------------------------------------------------------------------- command line
def _collect_pdfs(paths: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        found.extend(sorted(p.rglob("*.pdf")) if p.is_dir() else [p])
    return found


def main(argv: Optional[Sequence[str]] = None, embedder=None) -> int:
    """python -m src.ingestion <pdf-or-folder>... [--persist-dir DIR] [--replace] | --rebuild

    Reads every PDF, then indexes all good chunks in one call. Files that fail are reported and make the exit
    code 1 (the good files are still indexed, unless --replace is used: then nothing is replaced).
    Exit codes: 0 ok, 1 problem (see output), 2 bad usage.
    """
    ap = argparse.ArgumentParser(prog="python -m src.ingestion", description="Ingest PDFs into the AuditAgent index.")
    ap.add_argument("paths", nargs="*", help="PDF files or folders containing PDFs")
    ap.add_argument("--persist-dir", default=None, help="index folder (default: CHROMA_PERSIST_DIR)")
    ap.add_argument("--replace", action="store_true", help="replace the whole index with these documents")
    ap.add_argument("--rebuild", action="store_true", help="regenerate the dense index from the stored corpus")
    args = ap.parse_args(argv)

    from src.retriever import EmbeddingModelError, HybridRetriever, RetrievalIndexError  # lazy: avoids import cycles
    try:
        retriever = HybridRetriever(persist_directory=args.persist_dir, embedder=embedder)
        if args.rebuild:
            retriever.rebuild()
            print(f"Rebuilt the index from the stored corpus ({retriever.count()} chunks).")
            return 0
        pdfs = _collect_pdfs(args.paths)
        if not pdfs:
            print("Give at least one PDF file or folder (or use --rebuild).")
            return 2
        chunks: List[EvidenceChunk] = []
        failures = 0
        for pdf in pdfs:
            try:
                got = chunk_pdf(pdf)
            except (FileNotFoundError, PdfIngestionError) as exc:
                failures += 1
                print(f"FAILED  {pdf}: {exc}")
                continue
            print(f"read    {pdf.name}: {len(got)} chunk(s)")
            chunks.extend(got)
        if args.replace and failures:
            print("Not replacing the index because some files failed.")
            return 1
        if chunks:
            retriever.index_documents(chunks, replace=args.replace)
            print(f"Submitted {len(chunks)} chunk(s); the index now holds {retriever.count()}.")
        else:
            print("No chunks to index.")
        return 1 if failures or not chunks else 0
    except (RetrievalIndexError, EmbeddingModelError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())