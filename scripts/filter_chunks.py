"""Apply the chunk quality filter and emit annotated JSONL + journal.

Loads ``data/interim/chunks.jsonl``, runs each chunk through
``finetune_slm.data.quality.evaluate_chunk``, and writes every chunk
(passing AND failing) to ``data/interim/chunks_filtered.jsonl`` with
three new fields appended: ``quality_passed``, ``quality_score``,
``quality_reasons``. Downstream stages decide whether to filter or
re-classify at consumption time.

Pure local — no API calls. Re-running overwrites the output cleanly.

Run::

    uv run python scripts/filter_chunks.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table  # noqa: E402

from finetune_slm.data.quality import (  # noqa: E402
    QUALITY_FILTER_VERSION,
    evaluate_chunk,
)

console = Console()

JOURNAL_PATH = PROJECT_ROOT / "docs" / "journal" / "04-chunk-quality-filter.md"
SAMPLE_SEED = 42

# ── cost-model placeholders ──
GPT_5_4_INPUT_PRICE_PER_1M_USD = 5.0     # placeholder
GPT_5_4_OUTPUT_PRICE_PER_1M_USD = 15.0   # placeholder
COHERE_RERANK_PRICE_PER_1K = 2.0         # placeholder
TINKER_TRAIN_PRICE_PER_1M_USD = 0.40     # placeholder

QA_PAIRS_PER_CHUNK = 5
QA_PROMPT_OVERHEAD_TOKENS = 400
QA_OUTPUT_TOKENS_PER_PAIR = 250
SFT_TOKENS_PER_PAIR = 280

REASON_EXPLANATIONS = {
    "toc": "long publication with extensive TOC / list / dot-leader content",
    "bibliography": "academic / research document with bibliography sections",
    "low_content": "PDF extraction issues or many fragmented / glossary chunks",
    "page_ref": "doctrine with dense cross-references between sections",
    "cover_metadata": "cover page / distribution statement / front-matter",
}


def categorize_reason(reason: str) -> str:
    """Pluck the category label from a reason string like 'toc: 87 ...'."""
    if ":" in reason:
        return reason.split(":", 1)[0].strip()
    return reason


def fmt_money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 else f"${x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-path", type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "chunks.jsonl",
    )
    ap.add_argument(
        "--out-path", type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "chunks_filtered.jsonl",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_path = args.in_path.resolve()
    out_path = args.out_path.resolve()

    if not in_path.exists():
        console.print(f"[red]input not found: {in_path}[/red]")
        return 1

    # ── load ──
    chunks: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    console.print(
        Panel.fit(
            f"[bold]Filtering {len(chunks):,} chunks[/bold]\n"
            f"  in:  {in_path.relative_to(PROJECT_ROOT)}\n"
            f"  out: {out_path.relative_to(PROJECT_ROOT)}"
            f"{'  (dry-run)' if args.dry_run else ''}"
        )
    )

    # ── evaluate ──
    annotated: list[dict] = []
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Evaluating chunks", total=len(chunks))
        for c in chunks:
            qr = evaluate_chunk(c.get("text", ""))
            annotated.append({**c, **qr})
            progress.advance(task)

    # ── write annotated jsonl ──
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in annotated:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── stats ──
    n_total = len(annotated)
    n_passed = sum(1 for r in annotated if r["quality_passed"])
    n_failed = n_total - n_passed

    reason_cat_counts: Counter = Counter()
    for r in annotated:
        for reason in r["quality_reasons"]:
            reason_cat_counts[categorize_reason(reason)] += 1

    per_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0}
    )
    for r in annotated:
        s = per_source[r["source_filename"]]
        s["total"] += 1
        if r["quality_passed"]:
            s["passed"] += 1
        else:
            s["failed"] += 1

    sorted_sources = sorted(
        per_source.items(),
        key=lambda kv: (
            -kv[1]["failed"] / kv[1]["total"] if kv[1]["total"] else 0,
            -kv[1]["failed"],
        ),
    )

    # Per-source dominant failure reason (used in Top-5 section)
    source_top_reason: dict[str, str] = {}
    src_reason_counter: dict[str, Counter] = defaultdict(Counter)
    for r in annotated:
        if not r["quality_reasons"]:
            continue
        for reason in r["quality_reasons"]:
            src_reason_counter[r["source_filename"]][categorize_reason(reason)] += 1
    for src, ctr in src_reason_counter.items():
        if ctr:
            source_top_reason[src] = ctr.most_common(1)[0][0]

    # ── journal ──
    journal = render_journal(
        annotated=annotated,
        in_path=in_path,
        out_path=out_path,
        n_total=n_total,
        n_passed=n_passed,
        n_failed=n_failed,
        reason_cat_counts=reason_cat_counts,
        per_source=per_source,
        sorted_sources=sorted_sources,
        source_top_reason=source_top_reason,
    )
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.write_text(journal, encoding="utf-8")

    # ── terminal summary ──
    summary = Table(title="Filter summary")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("Total chunks", f"{n_total:,}")
    summary.add_row(
        "Passing",
        f"{n_passed:,}  ({n_passed/n_total*100:.1f}%)" if n_total else "0",
    )
    summary.add_row(
        "Failing",
        f"{n_failed:,}  ({n_failed/n_total*100:.1f}%)" if n_total else "0",
    )
    for cat, n in reason_cat_counts.most_common():
        summary.add_row(f"  reason: {cat}", f"{n:,}")
    console.print(summary)
    console.print(f"[green]wrote[/green] {JOURNAL_PATH.relative_to(PROJECT_ROOT)}")
    if not args.dry_run:
        console.print(
            f"[green]wrote[/green] {out_path.relative_to(PROJECT_ROOT)}  "
            f"({n_total:,} records)"
        )

    return 0


# ────────────────────── journal renderer ──────────────────────


def render_journal(
    *,
    annotated: list[dict],
    in_path: Path,
    out_path: Path,
    n_total: int,
    n_passed: int,
    n_failed: int,
    reason_cat_counts: Counter,
    per_source: dict[str, dict[str, int]],
    sorted_sources: list,
    source_top_reason: dict[str, str],
) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Chunk quality filter — Phase A pre-Q&A")
    a("")
    a("_Generated by `scripts/filter_chunks.py`. Re-running overwrites this file._")
    a("")
    a(f"- Input:  `{in_path.relative_to(PROJECT_ROOT)}`")
    a(f"- Output: `{out_path.relative_to(PROJECT_ROOT)}`  (all chunks "
      "written; downstream filters at consumption time)")
    a("")

    # ── Configuration ──
    a("## Configuration")
    a("")
    a(f"- Filter version: **{QUALITY_FILTER_VERSION}**")
    a("")
    a("### Heuristic descriptions")
    a("")
    a(
        "- **`is_toc`** — Counts `\\.{5,}|…{3,}` (dot-leader / ellipsis "
        "runs). Flags when **≥10 runs** are present in a chunk. Catches "
        "TOC pages, front-matter contents, and acronym lists with "
        "dot-leader formatting. Below the threshold, a soft penalty "
        "`1 − n_runs/12` drags borderline cases' aggregate score down."
    )
    a(
        "- **`is_bibliography`** — Aggregates five citation signals "
        "(header `References|Bibliography|Works Cited|Citations` in "
        "first 200 chars, paren-year `(YYYY)` count, bare-year count, "
        "`Author, X.` comma-initial pattern count, `et al.` count, URL "
        "count) into a citation-strength score. Flags when any of: "
        "`header + strength≥4`, `URLs≥3 AND years≥4`, "
        "`author-initials≥5 AND years≥4`, `strength≥18`."
    )
    a(
        "- **`is_low_content`** — Floor trigger: **fewer than 3 sentence "
        "terminators** (`.!?` + whitespace) catches glossaries, equation "
        "dumps, and garbled extracts. Secondary: ≥8 lines AND >75% lines "
        "<40 chars AND <10 sentences catches list-shaped chunks."
    )
    a(
        "- **`is_page_ref_only`** — Flags chunks with **≥15 Roman-numeral "
        "section refs** (`III-7`-style) AND **<5 sentence terminators**. "
        "Two-signal trigger guards against legitimate doctrinal cross-"
        "references."
    )
    a(
        "- **`is_cover_or_metadata`** — Keyword set (\"distribution "
        "statement\", \"by order of\", \"headquarters,\", \"approved "
        "for public release\", \"this publication supersedes\", etc.). "
        "Flags when **keyword_hits≥3 AND sentences<10**, OR **>40% "
        "all-caps lines AND sentences<8**."
    )
    a("")
    a(
        "Conservative bias is enforced everywhere except the `<3 "
        "sentence terminators` floor: every other detector requires "
        "**two or more independent signals** before flagging."
    )
    a("")

    # ── Filter results ──
    a("## Filter results")
    a("")
    a(f"- Total chunks loaded: **{n_total:,}**")
    if n_total:
        a(f"- Passing: **{n_passed:,}**  ({n_passed/n_total*100:.1f}%)")
        a(f"- Failing: **{n_failed:,}**  ({n_failed/n_total*100:.1f}%)")
    a("")
    a("### Breakdown by reason (chunks can have multiple reasons)")
    a("")
    a("| Reason | Count | % of failures |")
    a("|---|---:|---:|")
    for cat, n in sorted(reason_cat_counts.items(), key=lambda kv: -kv[1]):
        pct = (n / n_failed * 100) if n_failed else 0
        a(f"| `{cat}` | {n:,} | {pct:.1f}% |")
    a("")

    # ── Per-source breakdown ──
    a("## Per-source breakdown")
    a("")
    a("Sorted by failure rate descending; ties broken by failure count.")
    a("")
    a("| Source | Total | Passing | Failing | Failure rate |")
    a("|---|---:|---:|---:|---:|")
    for src, s in sorted_sources:
        rate = s["failed"] / s["total"] * 100 if s["total"] else 0
        a(
            f"| `{src}` | {s['total']} | {s['passed']} | "
            f"{s['failed']} | {rate:.1f}% |"
        )
    a("")

    # ── Top 5 sources by failure rate ──
    a("## Top 5 sources by failure rate")
    a("")
    a("(NOT a problem — orientation only.)")
    a("")
    for src, s in sorted_sources[:5]:
        rate = s["failed"] / s["total"] * 100 if s["total"] else 0
        top_reason = source_top_reason.get(src, "n/a")
        explanation = REASON_EXPLANATIONS.get(top_reason, "mixed signals")
        a(
            f"- **`{src}`** — {s['failed']}/{s['total']} failed "
            f"({rate:.1f}%) — {explanation}."
        )
    a("")

    # ── Sample dropped chunks per category ──
    rng = random.Random(SAMPLE_SEED)

    a("## Sample dropped chunks — sanity check")
    a("")
    a(
        "Three full chunks per rejection category. Read these and "
        "confirm \"yes, droppable\"."
    )
    a("")
    for cat in ("toc", "bibliography", "low_content", "page_ref", "cover_metadata"):
        matches = [
            r
            for r in annotated
            if any(categorize_reason(reason) == cat for reason in r["quality_reasons"])
        ]
        a(f"### Category: `{cat}` — {len(matches)} chunks total")
        a("")
        if not matches:
            a("_No chunks flagged with this reason._")
            a("")
            continue
        sample = rng.sample(matches, min(3, len(matches)))
        for r in sample:
            a(f"#### `{r['chunk_id']}`")
            a(f"- **source:** `{r['source_filename']}`")
            a(f"- **quality_score:** {r['quality_score']:.3f}")
            a(f"- **quality_reasons:** `{r['quality_reasons']}`")
            a("")
            a("```text")
            a(r["text"].replace("```", "ʼʼʼ"))
            a("```")
            a("")

    # ── Borderline ──
    borderline = [r for r in annotated if 0.4 <= r["quality_score"] <= 0.6]
    a("## Sample borderline chunks — human review")
    a("")
    a(
        f"Three chunks with quality_score in [0.4, 0.6]. {len(borderline)} "
        "chunks fall in this band; sampling 3 with the seed for stability."
    )
    a("")
    if borderline:
        sample = rng.sample(borderline, min(3, len(borderline)))
        for r in sample:
            a(f"### `{r['chunk_id']}`")
            a(f"- **source:** `{r['source_filename']}`")
            a(f"- **quality_score:** {r['quality_score']:.3f}")
            a(f"- **quality_passed:** {r['quality_passed']}")
            a(f"- **quality_reasons:** `{r['quality_reasons']}`")
            a("")
            a("```text")
            a(r["text"].replace("```", "ʼʼʼ"))
            a("```")
            a("")
    else:
        a("_No chunks in the borderline band._")
        a("")

    # ── Kept ──
    a("## Sample kept chunks — confirm no false drops")
    a("")
    a(
        "Three random passing chunks. Confirms we didn't accidentally "
        "throw out good doctrine."
    )
    a("")
    passing = [r for r in annotated if r["quality_passed"]]
    if passing:
        sample = rng.sample(passing, min(3, len(passing)))
        for r in sample:
            a(f"### `{r['chunk_id']}`")
            a(f"- **source:** `{r['source_filename']}`")
            a(f"- **quality_score:** {r['quality_score']:.3f}")
            a("")
            a("```text")
            a(r["text"].replace("```", "ʼʼʼ"))
            a("```")
            a("")

    # ── Updated cost projection ──
    sum_passing_tokens = sum(
        r["token_count"] for r in annotated if r["quality_passed"]
    )
    qa_input = sum_passing_tokens + n_passed * QA_PROMPT_OVERHEAD_TOKENS
    qa_output = n_passed * QA_PAIRS_PER_CHUNK * QA_OUTPUT_TOKENS_PER_PAIR
    qa_input_cost = qa_input / 1_000_000 * GPT_5_4_INPUT_PRICE_PER_1M_USD
    qa_output_cost = qa_output / 1_000_000 * GPT_5_4_OUTPUT_PRICE_PER_1M_USD
    qa_total = qa_input_cost + qa_output_cost

    n_pairs = n_passed * QA_PAIRS_PER_CHUNK
    cohere_cost = n_pairs / 1000 * COHERE_RERANK_PRICE_PER_1K

    sft_tokens = n_pairs * SFT_TOKENS_PER_PAIR
    sft_cost = sft_tokens / 1_000_000 * TINKER_TRAIN_PRICE_PER_1M_USD

    a("## Updated cost projection")
    a("")
    a(
        "All rates are **placeholders**. Calibrate the GPT-5.4 rate with "
        "one real call before committing the full Q&A run in Prompt #4."
    )
    a("")
    a(f"- Passing chunks: **{n_passed:,}**  "
      f"({n_passed/n_total*100:.1f}% of {n_total:,})")
    a(f"- Passing chunk tokens: **{sum_passing_tokens:,}**")
    a("")
    a("### (a) Q&A generation via GPT-5.4 (placeholder $5 in / $15 out per 1M)")
    a("")
    a(f"- Calls: **{n_passed:,}**  (1/chunk → {QA_PAIRS_PER_CHUNK} pairs/call)")
    a(f"- Input tokens: **{qa_input:,}**  (chunk + ~{QA_PROMPT_OVERHEAD_TOKENS} overhead)")
    a(
        f"- Output tokens: **{qa_output:,}**  "
        f"({QA_PAIRS_PER_CHUNK} × ~{QA_OUTPUT_TOKENS_PER_PAIR} per pair)"
    )
    a(f"- Input cost: **{fmt_money(qa_input_cost)}**")
    a(f"- Output cost: **{fmt_money(qa_output_cost)}**")
    a(f"- **Total Q&A gen: {fmt_money(qa_total)}**")
    a("")
    a("### (b) Cohere rerank pair-quality filter (placeholder $2 / 1k searches)")
    a("")
    a(f"- Searches (1 per pair): **{n_pairs:,}**")
    a(f"- Estimated cost: **{fmt_money(cohere_cost)}**")
    a("")
    a("### (c) 1-epoch SFT on resulting Q&A dataset (Tinker placeholder $0.40 / 1M)")
    a("")
    a(f"- Q&A pairs: **{n_pairs:,}**")
    a(
        f"- Training tokens (1 epoch, context-free SFT, "
        f"~{SFT_TOKENS_PER_PAIR}/pair): **{sft_tokens:,}**"
    )
    a(f"- **Estimated 1-epoch SFT cost: {fmt_money(sft_cost)}**")
    a("")

    # ── Known issues ──
    a("## Known issues / things to revisit")
    a("")
    a(
        "- **Bibliography detection is calibrated for academic-style "
        "citations.** Doctrine documents that cite via footnote (e.g. "
        "NATO publications) won't trip the detector. If a chunk full of "
        "footnote-style refs slips into passing, broaden the patterns."
    )
    a(
        "- **The FAA 7110.65 acronym-list chunks are caught as `toc`** "
        "because they use dot-leader formatting. Strictly they're "
        "glossaries — useful for acronym-expansion Q&A but listy enough "
        "that Q&A generation will produce shallow questions. Dropping is "
        "the right default; revisit if final corpus is too thin."
    )
    a(
        "- **`dtic_5g_uav_swarm_2023` has garbled PDF extraction** "
        "(letter encoding artifacts — 'DOD department of defense' lands "
        "as continuous non-prose). The `<3 sentence terminators` floor "
        "catches these correctly. Better fix: re-extract with "
        "pdfplumber-only or drop the source."
    )
    a(
        "- **Cover-page detection assumes US-style boilerplate.** "
        "NATO / UK / EASA / ICAO documents use different masthead "
        "language. If a non-US doc's chunk 0 still passes when it "
        "shouldn't, extend `COVER_KEYWORDS`."
    )
    a(
        "- **Conservative bias is intentional.** If a future iteration "
        "of this filter looks too aggressive against a specific source, "
        "raise the threshold for that detector rather than disabling it "
        "wholesale."
    )
    a("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
