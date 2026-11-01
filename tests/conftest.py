"""Shared pytest helpers. PDFs are generated on the fly with PyMuPDF: no binary fixtures are committed."""
from pathlib import Path
from typing import List, Optional

import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path):
    """make_pdf(name, pages, password=None, subdir=None) -> Path. One page per string."""
    def _make(name: str, pages: List[str], password: Optional[str] = None, subdir: Optional[str] = None) -> Path:
        folder = tmp_path / subdir if subdir else tmp_path
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            if text:
                left = page.insert_textbox(pymupdf.Rect(50, 50, 545, 792), text, fontsize=10)
                assert left >= 0, "test text overflowed the page"
        kwargs = {}
        if password:
            kwargs = dict(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password + "-owner")
        doc.save(str(path), **kwargs)
        doc.close()
        return path
    return _make