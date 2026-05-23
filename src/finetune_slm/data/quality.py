"""Heuristic quality filter for doctrine chunks.

Five detectors flag chunks that are poor Q&A-generation material:

- ``is_toc`` — Table of Contents / dot-leader-dense lists.
- ``is_bibliography`` — Reference / Works Cited / academic citation lists.
- ``is_low_content`` — fragmented chunks (few full sentences, glossaries,
  equation dumps, garbled PDF extracts).
- ``is_page_ref_only`` — chunks dominated by section / "see page X" refs.
- ``is_cover_or_metadata`` — cover pages, distribution statements,
  by-order-of front matter.

Each detector returns ``(failed: bool, reason: str | None,
score_component: float)``. ``score_component`` is in [0, 1]: 0 = junk,
1 = clean prose. Aggregator ``evaluate_chunk`` averages the five soft
scores and ANDs the failure flags.

Conservative bias: every detector except the unambiguous "<3 sentence
terminators" floor requires **two or more independent signals** to flag.
Better to ship a few mediocre chunks than to drop genuine doctrine.

All functions are pure functions of the input text. Module-level state
is limited to compiled regex constants and the cover-keyword tuple.
"""

from __future__ import annotations

import re
from typing import Optional

QUALITY_FILTER_VERSION = "v1"

# ───────── compiled regex patterns ─────────

# Dotted leaders typical of TOC entries ("Chapter 1 ............ 5")
# and Unicode ellipsis runs.
RE_DOT_LEADER = re.compile(r"\.{5,}|…{3,}")

# Years: in parentheses (academic citation style) and bare 4-digit years.
RE_YEAR_PAREN = re.compile(r"\(\d{4}\)")
RE_YEAR_BARE = re.compile(r"\b(?:19|20)\d{2}\b")

# Bibliography signals
RE_AUTHOR_INITIAL = re.compile(r"\b[A-Z][a-z]+,\s+[A-Z]\.")  # "Smith, J."
RE_ET_AL = re.compile(r"\bet\s+al\.", re.IGNORECASE)
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_BIB_HEADER = re.compile(
    r"\b(References|Bibliography|Works\s+Cited|Citations)\b"
)

# Roman-numeral-section-prefixed page refs ("III-7", "A-1-2") and
# explicit "see page / refer to" pointers.
RE_SECTION_REF = re.compile(r"\b[IVXLCM]{1,4}-\d+(?:-\d+)?\b")
RE_PAGE_REF = re.compile(
    r"\b(?:see\s+page|see\s+chapter|see\s+section|see\s+figure|"
    r"see\s+table|refer\s+to)\b",
    re.IGNORECASE,
)

# Sentence terminators: ., !, ? followed by whitespace or end-of-string.
RE_SENT_END = re.compile(r"[.!?](?:\s|$)")

# Cover-page / front-matter signal keywords. Lowercase comparison.
COVER_KEYWORDS: tuple[str, ...] = (
    "distribution statement",
    "for official use only",
    "by order of",
    "headquarters,",
    "department of the army",
    "department of the air force",
    "department of the navy",
    "department of defense",
    "joint chiefs of staff",
    "this publication is available",
    "this publication supersedes",
    "approved for public release",
    "distribution is unlimited",
    "active component, regular army",
)


# ───────── individual detectors ─────────


def is_toc(text: str) -> tuple[bool, Optional[str], float]:
    """Detect Table-of-Contents and dot-leader-dense list chunks.

    Threshold ≥10 dot-leader runs flagged 99 chunks (≈2.4%) on the
    current 4,051-chunk corpus, all of which spot-check as
    legitimate TOC / acronym-list content. Under that threshold we
    return a soft penalty so borderline chunks still drag the
    aggregate score down without being rejected.
    """
    if not text or not text.strip():
        return True, "toc: empty", 0.0
    n_dots = len(RE_DOT_LEADER.findall(text))
    if n_dots >= 10:
        return True, f"toc: {n_dots} dot-leader runs", 0.0
    score = max(0.0, 1.0 - n_dots / 12.0)
    return False, None, score


def is_bibliography(text: str) -> tuple[bool, Optional[str], float]:
    """Detect reference / works-cited / bibliography chunks.

    Combines five signals (header keyword in first 200 chars, year-in-
    parens count, bare-year count, author-comma-initial count, "et al."
    count, URL count) into an aggregate citation-strength score. Two
    or more independent strong signals required to fail. Single-signal
    matches only depress the soft score.
    """
    if not text or not text.strip():
        return False, None, 1.0
    head = text[:200]
    body = text[:3000]
    has_header = bool(RE_BIB_HEADER.search(head))
    n_year_paren = len(RE_YEAR_PAREN.findall(body))
    n_year = len(RE_YEAR_BARE.findall(body))
    n_author = len(RE_AUTHOR_INITIAL.findall(body))
    n_etal = len(RE_ET_AL.findall(body))
    n_url = len(RE_URL.findall(body))

    # Weighted aggregate: weight the "academic" signals (parens, authors,
    # et al., URLs) higher than bare years (which appear in lots of
    # ordinary doctrine prose: "in 2019, the policy was…").
    strength = 2 * n_year_paren + 2 * n_author + 2 * n_etal + 2 * n_url + n_year

    if has_header and strength >= 4:
        return True, "bibliography: header + citation density", 0.0
    if n_url >= 3 and n_year >= 4:
        return True, f"bibliography: {n_url} URLs + {n_year} years", 0.0
    if n_author >= 5 and n_year >= 4:
        return True, f"bibliography: {n_author} author-initial citations", 0.0
    if strength >= 18:
        return True, f"bibliography: citation strength {strength}", 0.0

    score = max(0.0, 1.0 - strength / 30.0)
    return False, None, score


def is_low_content(text: str) -> tuple[bool, Optional[str], float]:
    """Detect fragmented, listy, or garbled chunks.

    Primary trigger (<3 sentence terminators) is unambiguous: anything
    that fails this on a 500-1500 token chunk is broken extract or pure
    list. Secondary trigger requires *both* many short lines and few
    sentences to fire — guards against single-paragraph chunks that
    happen to have short lines.
    """
    if not text or not text.strip():
        return True, "low_content: empty", 0.0
    n_sent = len(RE_SENT_END.findall(text))
    if n_sent < 3:
        return True, f"low_content: {n_sent} sentence terminators", 0.0

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    n_lines = len(lines)
    if n_lines >= 8:
        n_short = sum(1 for line in lines if len(line) < 40)
        if n_short / n_lines > 0.75 and n_sent < 10:
            return (
                True,
                f"low_content: {n_short}/{n_lines} short lines + {n_sent} sentences",
                0.0,
            )

    score = min(1.0, n_sent / 15.0)
    return False, None, score


def is_page_ref_only(text: str) -> tuple[bool, Optional[str], float]:
    """Detect chunks dominated by section / page references with little prose.

    Doctrine cross-references each other heavily, so a chunk with 5–10
    section refs is normal. We only flag when ref density is extreme
    AND sentence count is low — both signals required.
    """
    if not text or not text.strip():
        return False, None, 1.0
    n_section = len(RE_SECTION_REF.findall(text))
    n_page = len(RE_PAGE_REF.findall(text))
    n_sent = len(RE_SENT_END.findall(text))

    if n_section >= 15 and n_sent < 5:
        return True, f"page_ref: {n_section} section refs + {n_sent} sentences", 0.0

    score = max(0.0, 1.0 - (n_section + n_page) / 30.0)
    return False, None, score


def is_cover_or_metadata(text: str) -> tuple[bool, Optional[str], float]:
    """Detect cover pages, distribution statements, front-matter authorization.

    Catches chunks like the start of every Army Publication that opens
    with "DISTRIBUTION RESTRICTION: Approved for public release;
    distribution is unlimited. Headquarters, Department of the Army".
    Requires either keyword density + low prose, OR a high all-caps-line
    ratio + low prose.
    """
    if not text or not text.strip():
        return False, None, 1.0
    lower = text.lower()
    keyword_hits = sum(1 for kw in COVER_KEYWORDS if kw in lower)
    n_sent = len(RE_SENT_END.findall(text))

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if keyword_hits >= 3 and n_sent < 10:
        return (
            True,
            f"cover_metadata: {keyword_hits} cover keywords + {n_sent} sentences",
            0.0,
        )

    if len(lines) >= 8 and n_sent < 8:
        allcaps_lines = [
            line
            for line in lines
            if line == line.upper()
            and any(c.isalpha() for c in line)
            and len(line) >= 6
        ]
        ratio = len(allcaps_lines) / len(lines)
        if ratio > 0.4:
            return (
                True,
                f"cover_metadata: {ratio:.0%} all-caps lines + {n_sent} sentences",
                0.0,
            )

    score = max(0.0, 1.0 - keyword_hits / 8.0)
    return False, None, score


# ───────── aggregator ─────────


def evaluate_chunk(text: str) -> dict:
    """Run all detectors and aggregate the result.

    Returns dict with:
      - ``quality_passed`` (bool): True iff every detector returned
        ``failed=False``.
      - ``quality_score`` (float): average of the five score components,
        always in [0, 1].
      - ``quality_reasons`` (list[str]): reason strings from every
        detector that returned ``failed=True``; empty when passed.
    """
    detectors = (
        is_toc,
        is_bibliography,
        is_low_content,
        is_page_ref_only,
        is_cover_or_metadata,
    )
    results = [d(text) for d in detectors]
    reasons = [reason for failed, reason, _ in results if failed]
    score_sum = sum(score for _, _, score in results)
    return {
        "quality_passed": not reasons,
        "quality_score": round(score_sum / len(results), 4),
        "quality_reasons": reasons,
    }
