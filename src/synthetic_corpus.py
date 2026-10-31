"""Generates a small SYNTHETIC rulebook PDF for demos and tests. Not real law; not legal advice.

    python -m src.synthetic_corpus [output.pdf]     (default: data/synthetic_corpus/synthetic_rulebook.pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import pymupdf

SYNTHETIC_NOTICE = "[Synthetic test text - not real law]"
SYNTHETIC_PAGES = [
    ("Section 1 Log Retention", "Synthetic Rule R-1: Access logs must be retained for a minimum of 90 days."),
    ("Section 2 Data Location", "Synthetic Rule R-2: Personal data must be stored on servers located within Region Alpha."),
    ("Section 3 Encryption", "Synthetic Rule R-3: Personal data must be encrypted at rest using an approved algorithm."),
    ("Section 4 Access Review", "Synthetic Rule R-4: Access rights must be reviewed periodically."),
]
DEFAULT_PATH = Path("data/synthetic_corpus/synthetic_rulebook.pdf")


def write_synthetic_pdf(path: Union[str, Path] = DEFAULT_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for heading, body in SYNTHETIC_PAGES:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 792), f"{heading}\n{body}\n\n{SYNTHETIC_NOTICE}", fontsize=11)
    doc.save(str(out))
    doc.close()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    print(f"Wrote {write_synthetic_pdf(args[0] if args else DEFAULT_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())