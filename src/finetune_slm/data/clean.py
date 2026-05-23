"""Text cleaning for doctrine PDFs.

Pipeline per PDF
----------------
1. Per-page text extraction (pypdf preferred, pdfplumber fallback).
2. Detect repeated headers / footers — any non-trivial line that appears
   on >= ``HEADER_FOOTER_THRESHOLD`` of pages is treated as boilerplate
   and stripped from every page.
3. Join pages with a paragraph break.
4. Apply global text cleaning:
   - normalize line endings to ``\n``
   - replace common PDF ligatures (fi / fl / ff / ffi / ffl)
   - strip control chars except ``\n`` and ``\t``
   - repair hyphenated line-end breaks (``wo-\\nrd`` → ``word``) when the
     next line continues lowercase
   - drop standalone page-number lines (``^\\s*\\d+\\s*$``)
   - collapse runs of spaces / tabs
   - strip trailing whitespace on each line
   - collapse runs of >=3 newlines down to exactly two (preserves
     paragraph breaks)

Tables, captions, and footnotes are **intentionally preserved**: doctrine
documents lean heavily on numbered tables and footnote citations, and
removing them would degrade the training signal.

Bumping ``CLEANING_VERSION`` re-marks chunk records so downstream stages
can spot stale data.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stderr
from pathlib import Path

CLEANING_VERSION = "v1"

HEADER_FOOTER_THRESHOLD = 0.30  # fraction of pages a line must appear on

# Common PDF private-use ligature codepoints → their ASCII expansion.
LIGATURES: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

# Translation table that maps C0/C1 control chars (except \t and \n) to None.
_CONTROL_TRANSLATE = {c: None for c in range(0x20) if c not in (0x09, 0x0A)}
_CONTROL_TRANSLATE[0x7F] = None
CONTROL_TRANSLATE = str.maketrans(_CONTROL_TRANSLATE)

RE_LINE_ENDINGS = re.compile(r"\r\n?")
# Hyphenated line break: a word character, a hyphen, optional spaces, a newline,
# then a lowercase letter — that's the rejoin signal.
RE_HYPHENATION = re.compile(r"(\w)-[ \t]*\n[ \t]*([a-z])")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
RE_TRAILING_WS = re.compile(r"[ \t]+\n")
RE_MANY_NEWLINES = re.compile(r"\n{3,}")
RE_PAGE_NUMBER_LINE = re.compile(r"^\s*\d+\s*$")


# ────────── extraction ──────────


def _extract_pages_pypdf(path: Path) -> list[str]:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    out: list[str] = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
    return out


def _extract_pages_pdfplumber(path: Path) -> list[str]:
    import pdfplumber

    out: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                out.append("")
    return out


def extract_raw_pages(path: Path) -> tuple[list[str], str]:
    """Extract per-page text without any cleaning.

    Returns ``(pages, method)`` where ``method`` is ``"pypdf"`` or
    ``"pdfplumber"``. pypdf is tried first; if it yields no text at all
    (sometimes happens for image-heavy or oddly-encoded PDFs), we fall
    back to pdfplumber.
    """
    try:
        with redirect_stderr(io.StringIO()):
            pages = _extract_pages_pypdf(path)
        if any(p.strip() for p in pages):
            return pages, "pypdf"
    except Exception:
        pass
    with redirect_stderr(io.StringIO()):
        pages = _extract_pages_pdfplumber(path)
    return pages, "pdfplumber"


# ────────── header / footer detection ──────────


def detect_repeated_headers_footers(
    pages: list[str], threshold: float = HEADER_FOOTER_THRESHOLD
) -> set[str]:
    """Return the set of stripped line strings that appear on >= ``threshold``
    of pages.

    Only meaningful non-trivial lines are counted (>= 2 chars after
    stripping). Lines are deduplicated per page before counting, so a
    line that recurs once per page is the right signal — we count *which
    pages it appears on*, not total occurrences.
    """
    if not pages:
        return set()
    n_pages = len(pages)
    min_pages = max(2, int(round(n_pages * threshold)))
    line_to_pages: dict[str, set[int]] = {}
    for i, page in enumerate(pages):
        seen_on_this_page: set[str] = set()
        for raw_line in page.splitlines():
            line = raw_line.strip()
            if len(line) < 2:
                continue
            if line in seen_on_this_page:
                continue
            seen_on_this_page.add(line)
            line_to_pages.setdefault(line, set()).add(i)
    return {line for line, pgs in line_to_pages.items() if len(pgs) >= min_pages}


def _strip_lines_in_page(page: str, banned: set[str]) -> str:
    if not banned:
        return page
    kept = [line for line in page.splitlines() if line.strip() not in banned]
    return "\n".join(kept)


# ────────── global cleaning ──────────


def apply_text_cleaning(text: str) -> str:
    """Apply all the post-extraction, post-strip cleaning normalizations.

    Order matters — see module docstring.
    """
    if not text:
        return ""

    # 1. Normalize line endings.
    text = RE_LINE_ENDINGS.sub("\n", text)
    # 2. Ligatures.
    for src, dst in LIGATURES.items():
        if src in text:
            text = text.replace(src, dst)
    # 3. Strip control chars (keep \t, \n).
    text = text.translate(CONTROL_TRANSLATE)
    # 4. Repair hyphenated line breaks BEFORE we collapse newlines.
    text = RE_HYPHENATION.sub(r"\1\2", text)
    # 5. Drop standalone page-number lines.
    kept_lines = [line for line in text.split("\n") if not RE_PAGE_NUMBER_LINE.match(line)]
    text = "\n".join(kept_lines)
    # 6. Collapse multi-space / multi-tab runs.
    text = RE_MULTI_SPACE.sub(" ", text)
    # 7. Strip trailing whitespace on each line.
    text = RE_TRAILING_WS.sub("\n", text)
    # 8. Collapse runs of 3+ newlines to exactly 2 (preserve paragraph breaks).
    text = RE_MANY_NEWLINES.sub("\n\n", text)
    return text.strip() + "\n"


def clean_pages(pages: list[str]) -> str:
    """Header/footer detection + per-page strip + global cleaning."""
    banned = detect_repeated_headers_footers(pages)
    stripped = [_strip_lines_in_page(p, banned) for p in pages]
    joined = "\n\n".join(stripped)
    return apply_text_cleaning(joined)


def clean_pdf_text(path: Path) -> tuple[str, str]:
    """Convenience: extract + clean. Returns ``(cleaned_text, method)``."""
    pages, method = extract_raw_pages(path)
    return clean_pages(pages), method
