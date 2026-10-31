"""Offline ingestion tests: PDFs are generated per test; nothing touches the network."""
import logging

import pytest

import src.ingestion as ing
from src.ingestion import (PdfIngestionError, chunk_pages, chunk_pdf, extract_pdf_pages, ingest_pdf,
                           make_chunk_id, parse_heading, slugify_name)
from src.schemas import SourceType

P1 = "Section 4.2 Log Retention\nAccess logs must be retained for a minimum of 90 days."
P3 = "Clause 7(b) Encryption\nPersonal data must be encrypted at rest using an approved algorithm."


def words(n):
    return " ".join(f"word{i:04d}" for i in range(n))


def overlap_len(a, b):
    return max((k for k in range(1, min(len(a), len(b)) + 1) if a[-k:] == b[:k]), default=0)


# ------------------------------------------------------------------ extraction
def test_pages_extracted_with_one_indexed_numbers_and_short_pages_skipped(make_pdf):
    pdf = make_pdf("rules.pdf", [P1, "tiny", P3])
    pages = extract_pdf_pages(pdf)
    assert [p["page_number"] for p in pages] == [1, 3]  # page 2 skipped, numbering stays real
    assert "retained for a minimum of 90 days" in pages[0]["text"] and "encrypted at rest" in pages[1]["text"]


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError) as e:
        extract_pdf_pages(missing)
    assert "nope.pdf" in str(e.value)
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(tmp_path)  # a directory is not a PDF


def test_blank_pdf_returns_empty_with_warning(make_pdf, caplog):
    pdf = make_pdf("blank.pdf", ["", "   "])
    with caplog.at_level(logging.WARNING):
        assert extract_pdf_pages(pdf) == [] and chunk_pdf(pdf) == []
    assert "no extractable text" in caplog.text


def test_zero_page_pdf_returns_empty_with_warning(tmp_path, caplog):
    p = tmp_path / "zero.pdf"
    p.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                  b"2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n")
    with caplog.at_level(logging.WARNING):
        assert extract_pdf_pages(p) == []
    assert "no pages" in caplog.text


def test_corrupt_pdf_raises_domain_error_and_logs(tmp_path, caplog):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is not a pdf at all")
    with caplog.at_level(logging.WARNING), pytest.raises(PdfIngestionError) as e:
        extract_pdf_pages(p)
    assert "corrupt" in str(e.value) and "Could not open PDF" in caplog.text


def test_password_protected_pdf_raises_domain_error(make_pdf, caplog):
    pdf = make_pdf("locked.pdf", [P1], password="secret")
    with caplog.at_level(logging.WARNING), pytest.raises(PdfIngestionError) as e:
        chunk_pdf(pdf)
    assert "password" in str(e.value) and "password-protected" in caplog.text


# ------------------------------------------------------------------ ids
def test_reingesting_same_file_gives_identical_ids(make_pdf):
    pdf = make_pdf("rules.pdf", [P1, "", P3])
    a, b = chunk_pdf(pdf), chunk_pdf(pdf)
    assert [c.id for c in a] == [c.id for c in b] and len({c.id for c in a}) == len(a) > 0


def test_ids_differ_for_same_basename_different_content(make_pdf):
    a = chunk_pdf(make_pdf("rules.pdf", [P1], subdir="one"))
    b = chunk_pdf(make_pdf("rules.pdf", [P3], subdir="two"))
    assert not {c.id for c in a} & {c.id for c in b}


def test_id_format_and_sanitized_name():
    assert slugify_name("My Rules (v2)!.pdf") == "my-rules-v2"
    assert slugify_name("###.pdf") == "doc"
    assert make_chunk_id("My Rules.pdf", "abcd1234", 3, 0) == "my-rules-abcd1234-p3-c0"


# ------------------------------------------------------------------ chunking
def test_chunk_size_and_overlap_are_characters():
    pages = [{"page_number": 1, "text": words(400)}]
    chunks = chunk_pages(pages, "d.pdf", "00000000", chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 10 and all(len(c.text) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert 0 < overlap_len(a.text, b.text) <= 40
    assert all(f"word{i:04d}" in " ".join(c.text for c in chunks) for i in (0, 199, 399))  # nothing lost


def test_defaults_come_from_settings_at_call_time(monkeypatch):
    monkeypatch.setattr(ing.settings, "CHUNK_SIZE", 120)
    monkeypatch.setattr(ing.settings, "CHUNK_OVERLAP", 20)
    chunks = chunk_pages([{"page_number": 1, "text": words(100)}], "d.pdf", "0")
    assert max(len(c.text) for c in chunks) <= 120 and len(chunks) > 5


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([{"page_number": 1, "text": words(50)}], "d.pdf", "0", chunk_size=100, chunk_overlap=100)


def test_metadata_preserved_across_split_chunks():
    text = "Section 9 Penalties\n" + words(300)
    chunks = chunk_pages([{"page_number": 5, "text": text}], "Rules.pdf", "ab12cd34", chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 5
    for c in chunks:
        assert (c.document_name, c.page_number, c.section_title) == ("Rules.pdf", 5, "Section 9 Penalties")
        assert c.source_type == SourceType.INTERNAL and c.source_url is None and c.score == 0.0
    assert chunks[0].text.startswith("Section 9 Penalties")


# ------------------------------------------------------------------ sections
def test_heading_carries_across_pages_and_defaults_to_general():
    pages = [{"page_number": 1, "text": "Preamble text that has no heading at all here."},
             {"page_number": 2, "text": "Section 4.2 Log Retention\nLogs must be kept 90 days at least."},
             {"page_number": 3, "text": "Continuation of the retention rule on a later page of text."},
             {"page_number": 4, "text": "Clause 7(b) Encryption\nData must be encrypted at rest always."}]
    got = [(c.page_number, c.section_title) for c in chunk_pages(pages, "d.pdf", "0")]
    assert got == [(1, "GENERAL"), (2, "Section 4.2 Log Retention"), (3, "Section 4.2 Log Retention"),
                   (4, "Clause 7(b) Encryption")]


def test_two_headings_on_one_page_split_into_separate_sections():
    text = "Section 1 Scope\nThis applies to all records.\nSection 2 Duties\nProviders must keep logs."
    got = [(c.section_title, c.text.split("\n")[0]) for c in chunk_pages([{"page_number": 1, "text": text}], "d", "0")]
    assert got == [("Section 1 Scope", "Section 1 Scope"), ("Section 2 Duties", "Section 2 Duties")]


def test_heading_only_trailing_segment_dropped_but_carried():
    pages = [{"page_number": 1, "text": "Some opening body text of reasonable length.\nSection 5 Penalties"},
             {"page_number": 2, "text": "Fines apply for each late report filed."}]
    got = [(c.page_number, c.section_title) for c in chunk_pages(pages, "d", "0")]
    assert got == [(1, "GENERAL"), (2, "Section 5 Penalties")]


@pytest.mark.parametrize("line,expected", [
    ("Section 4.2 Log Retention", "Section 4.2 Log Retention"),
    ("SECTION 14.2(b) Notification", "SECTION 14.2(b) Notification"),
    ("Clause 7.", "Clause 7"),
    ("Chapter 3: Data Protection", "Chapter 3: Data Protection"),
    ("Section 4.2 requires logs to be kept", None),        # lowercase continuation = body text
    ("Section 4.2 The controller shall keep logs.", None),  # reads like a sentence
    ("Section 4.2a Something", None),                       # identifier does not end cleanly
    ("See Section 4.2 for details", None),                  # does not start the line
    ("Section Overview", None),                             # no numeric identifier
    ("Article 5 Records", None),                            # only Section/Clause/Chapter
    ("Section 4 " + "X" * 100, None),                       # too long to be a heading
    ("", None),
])
def test_heading_detection_is_conservative(line, expected):
    assert parse_heading(line) == expected


# ------------------------------------------------------------------ end to end
def test_chunk_pdf_end_to_end(make_pdf):
    pdf = make_pdf("Rule Book.pdf", [P1, "tiny", P3])
    chunks = chunk_pdf(pdf)
    assert [(c.page_number, c.section_title) for c in chunks] == [(1, "Section 4.2 Log Retention"), (3, "Clause 7(b) Encryption")]
    assert all(c.document_name == "Rule Book.pdf" and c.id.startswith("rule-book-") for c in chunks)
    assert all(c.source_type == SourceType.INTERNAL and c.source_url is None for c in chunks)


def test_ingest_pdf_indexes_chunks_once_and_skips_empty(make_pdf):
    class Spy:
        def __init__(self): self.calls = []
        def index_documents(self, documents, *a, **k): self.calls.append(list(documents))
    spy = Spy()
    chunks = ingest_pdf(make_pdf("r.pdf", [P1, P3]), spy)
    assert len(spy.calls) == 1 and spy.calls[0] == chunks and len(chunks) == 2
    assert ingest_pdf(make_pdf("blank.pdf", [""]), spy) == [] and len(spy.calls) == 1


def test_ingest_pdf_propagates_errors_instead_of_hiding_them(tmp_path):
    class Spy:
        def index_documents(self, documents, *a, **k): raise AssertionError("must not be called")
    with pytest.raises(FileNotFoundError):
        ingest_pdf(tmp_path / "missing.pdf", Spy())