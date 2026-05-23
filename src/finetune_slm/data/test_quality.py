"""Unit tests for ``src/finetune_slm/data/quality.py``.

Run from the project root::

    uv run pytest src/finetune_slm/data/test_quality.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put src/ on sys.path so `from finetune_slm.data.quality import ...` resolves
# regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune_slm.data.quality import (  # noqa: E402
    evaluate_chunk,
    is_bibliography,
    is_cover_or_metadata,
    is_low_content,
    is_page_ref_only,
    is_toc,
)


# ───────── individual detectors ─────────


def test_is_toc_flags_synthetic_toc():
    text = (
        "Table of Contents\n"
        "\n"
        "Chapter 1 — Introduction ........................................ 1\n"
        "Chapter 2 — Overview ............................................ 5\n"
        "Section A ........................................................ I-1\n"
        "Section B ........................................................ I-3\n"
        "Section C ........................................................ I-5\n"
        "Section D ........................................................ I-7\n"
        "Section E ........................................................ II-1\n"
        "Section F ........................................................ II-3\n"
        "Section G ........................................................ II-5\n"
        "Section H ........................................................ II-7\n"
        "Section I ........................................................ III-1\n"
        "Section J ........................................................ III-5\n"
    )
    failed, reason, _ = is_toc(text)
    assert failed, f"TOC not flagged; reason={reason!r}"


def test_is_bibliography_flags_synthetic_bibliography():
    text = (
        "References\n"
        "\n"
        "Smith, J. (2020). Drone warfare in the modern era. Journal of Defense, 15(3), 22-45.\n"
        "Johnson, A. and Brown, M. (2019). Counter-UAS systems. Defense Press.\n"
        "Williams, R., Davis, C., et al. (2021). Autonomous systems and ethics. CRS Report.\n"
        "Brown, M. (2018). UAV swarming algorithms. IEEE Conference Proceedings.\n"
        "Anderson, P. (2017). Joint targeting cycle. U.S. Army Pubs.\n"
    )
    failed, reason, _ = is_bibliography(text)
    assert failed, f"Bibliography not flagged; reason={reason!r}"


def test_is_low_content_flags_fragmented_glossary():
    text = (
        "Figure 1 - chart\n"
        "Figure 2 - graph\n"
        "Figure 3 - histogram\n"
        "DOD department of defense\n"
        "GCD greatest common divisor\n"
        "MANET mobile ad hoc network\n"
        "UAV unmanned aerial vehicle\n"
        "FM frequency modulation\n"
        "5G fifth generation\n"
    )
    failed, reason, _ = is_low_content(text)
    assert failed, f"Glossary not flagged; reason={reason!r}"


def test_is_cover_or_metadata_flags_cover_page():
    text = (
        "DISTRIBUTION RESTRICTION: Approved for public release; "
        "distribution is unlimited.\n"
        "HEADQUARTERS, DEPARTMENT OF THE ARMY\n"
        "BY ORDER OF THE SECRETARY OF THE ARMY\n"
        "OCTOBER 2020\n"
        "ATP 3-04.7\n"
    )
    failed, reason, _ = is_cover_or_metadata(text)
    assert failed, f"Cover page not flagged; reason={reason!r}"


# ───────── aggregator ─────────


def test_normal_prose_passes_with_high_score():
    text = (
        "Joint operations require careful coordination across all participating services. "
        "Commanders must maintain situational awareness throughout the campaign. "
        "Course of action development typically follows a structured process. "
        "First, the staff identifies the mission and operational constraints. "
        "Second, they generate multiple feasible options for the commander. "
        "Third, they analyze each option against established criteria. "
        "Finally, the commander selects the preferred course of action based on the analysis. "
        "This methodology ensures that all relevant factors are considered before final decisions are made. "
        "The process is iterative and may require multiple revisions based on new information. "
        "Lessons learned are captured for use in subsequent planning cycles. "
        "Each phase has specific deliverables and gates that must be met. "
        "These gates serve as decision points for commanders."
    )
    result = evaluate_chunk(text)
    assert result["quality_passed"], (
        f"Normal prose unexpectedly failed; reasons={result['quality_reasons']}"
    )
    assert result["quality_score"] > 0.8, (
        f"Score too low for normal prose: {result['quality_score']}"
    )


def test_empty_chunk_fails():
    result = evaluate_chunk("")
    assert not result["quality_passed"]
    assert result["quality_reasons"], "empty chunk should have at least one reason"


def test_page_ref_only_flags_dense_section_refs():
    text = (
        "See page III-7 for joint air operations planning. "
        "Section II-12 III-7 IV-1 V-2 VI-3 VII-4 VIII-5 IX-6 X-7 XI-8 "
        "II-13 III-8 IV-2 V-3 VI-4 VII-5 VIII-6"
    )
    failed, reason, _ = is_page_ref_only(text)
    assert failed, f"page_ref_only not flagged; reason={reason!r}"


def test_evaluate_chunk_returns_required_shape():
    result = evaluate_chunk("Hello. World. Friends.")
    assert set(result.keys()) == {"quality_passed", "quality_score", "quality_reasons"}
    assert isinstance(result["quality_passed"], bool)
    assert isinstance(result["quality_score"], float)
    assert isinstance(result["quality_reasons"], list)
