"""Phase A corpus prep: discover → clean → chunk → attach manifest → JSONL.

Run::

    uv run python scripts/prepare_corpus.py

Defaults mirror the constants in ADR 0001 / ADR 0002:

- target chunk size: 1200 tokens
- hard cap:          1500 tokens
- overlap:           150 tokens
- tokenizer:         openai/gpt-oss-20b (HF AutoTokenizer)

Output: ``data/interim/chunks.jsonl``, one chunk per line. A summary
journal entry is written to ``docs/journal/03-cleaning-and-chunking.md``.

This stage NEVER generates Q&A pairs — that lives in a separate step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Make `from finetune_slm...` work without depending on how setuptools is configured.
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

from finetune_slm.data.chunk import CHUNKING_VERSION, chunk_text  # noqa: E402
from finetune_slm.data.clean import (  # noqa: E402
    CLEANING_VERSION,
    HEADER_FOOTER_THRESHOLD,
    clean_pages,
    extract_raw_pages,
)
from finetune_slm.data.manifest_loader import (  # noqa: E402
    discover_training_pdfs,
    list_non_training_files,
    load_paired_manifest,
)

warnings.filterwarnings("ignore")
console = Console()

# ── cost model placeholders (revisit before any real spend) ──
SONNET_INPUT_PRICE_PER_1M_USD = 3.0
SONNET_OUTPUT_PRICE_PER_1M_USD = 15.0
TINKER_TRAIN_PRICE_PER_1M_USD = 0.40

QA_PAIRS_PER_CHUNK = 5
QA_PROMPT_OVERHEAD_TOKENS = 400        # system prompt + scaffolding per call
QA_OUTPUT_TOKENS_PER_PAIR = 250        # rough Q + A length per pair
SFT_TOKENS_PER_PAIR = 280              # context-free SFT (no source in prompt)

JOURNAL_PATH = PROJECT_ROOT / "docs" / "journal" / "03-cleaning-and-chunking.md"
SAMPLE_SEED = 0
N_SAMPLES = 3


# ──────────────────────── helpers ────────────────────────


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def derive_source_slug(pdf_path: Path, manifest: dict, raw_dir: Path) -> str:
    """Stable, filesystem-safe slug. Prefer manifest.source_id; fall back to path."""
    sid = manifest.get("source_id")
    if sid:
        return str(sid).strip()
    rel = pdf_path.relative_to(raw_dir)
    base = "_".join(list(rel.parent.parts) + [pdf_path.stem])
    base = "".join(c if (c.isalnum() or c in "_-") else "_" for c in base)
    return base.lower()[:120]


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b", token=os.environ.get("HF_TOKEN")
    )


def percentile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def fmt_money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 else f"${x:.4f}"


# ──────────────────────── main ────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    ap.add_argument(
        "--out-path", type=Path, default=PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
    )
    ap.add_argument("--target-tokens", type=int, default=1200)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--overlap-tokens", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_dir = args.raw_dir.resolve()
    out_path = args.out_path.resolve()

    if not raw_dir.exists():
        console.print(f"[red]raw dir not found: {raw_dir}[/red]")
        return 1

    pdfs = discover_training_pdfs(raw_dir)
    if not pdfs:
        console.print(f"[red]no PDFs in {raw_dir}[/red]")
        return 1

    skipped_other = list_non_training_files(raw_dir)

    console.print(
        Panel.fit(
            f"[bold]Cleaning + chunking {len(pdfs)} PDFs[/bold]\n"
            f"  raw:  {raw_dir.relative_to(PROJECT_ROOT)}\n"
            f"  out:  {out_path.relative_to(PROJECT_ROOT)}"
            f"{'  (dry-run)' if args.dry_run else ''}"
        )
    )

    tokenizer = load_tokenizer()
    console.print("[bold green]Tokenizer:[/bold green] openai/gpt-oss-20b\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_source: list[dict] = []
    all_chunks_meta: list[dict] = []
    known_issues: list[tuple[str, str]] = []

    out_f = None if args.dry_run else out_path.open("w", encoding="utf-8")
    try:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        )
        with progress:
            task = progress.add_task("Cleaning + chunking", total=len(pdfs))
            for pdf in pdfs:
                rel = str(pdf.relative_to(raw_dir))
                try:
                    pages, method = extract_raw_pages(pdf)
                    raw_text = "\n\n".join(pages)
                    raw_tokens = count_tokens(tokenizer, raw_text)

                    cleaned_text = clean_pages(pages)
                    if not cleaned_text.strip():
                        known_issues.append((rel, "cleaning produced empty text"))
                        per_source.append(
                            {
                                "source_filename": rel,
                                "source_slug": pdf.parent.name,
                                "raw_tokens": raw_tokens,
                                "cleaned_tokens": 0,
                                "n_chunks": 0,
                                "mean_tokens_per_chunk": 0,
                                "extraction_method": method,
                                "n_pages": len(pages),
                            }
                        )
                        progress.advance(task)
                        continue

                    cleaned_tokens = count_tokens(tokenizer, cleaned_text)
                    manifest = load_paired_manifest(pdf)
                    slug = derive_source_slug(pdf, manifest, raw_dir)

                    chunks = chunk_text(
                        cleaned_text,
                        tokenizer,
                        target_tokens=args.target_tokens,
                        max_tokens=args.max_tokens,
                        overlap_tokens=args.overlap_tokens,
                    )

                    for c in chunks:
                        record = {
                            "chunk_id": f"{slug}__{c['chunk_index']:04d}",
                            "source_filename": rel,
                            "source_manifest": manifest,
                            "chunk_index": c["chunk_index"],
                            "total_chunks_in_source": len(chunks),
                            "token_count": c["token_count"],
                            "char_count": c["char_count"],
                            "text": c["text"],
                            "char_offset_start": c["char_offset_start"],
                            "char_offset_end": c["char_offset_end"],
                            "extraction_method": method,
                            "cleaning_version": CLEANING_VERSION,
                            "chunking_version": CHUNKING_VERSION,
                        }
                        if out_f is not None:
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        all_chunks_meta.append(
                            {
                                "chunk_id": record["chunk_id"],
                                "source_filename": rel,
                                "source_slug": slug,
                                "chunk_index": record["chunk_index"],
                                "total_chunks_in_source": record["total_chunks_in_source"],
                                "token_count": record["token_count"],
                                "char_count": record["char_count"],
                                "text": record["text"],
                                "char_offset_start": record["char_offset_start"],
                                "char_offset_end": record["char_offset_end"],
                                "extraction_method": record["extraction_method"],
                                "manifest_source_id": manifest.get("source_id"),
                                "manifest_license": manifest.get("license"),
                            }
                        )

                    chunk_tokens_this_pdf = [c["token_count"] for c in chunks]
                    per_source.append(
                        {
                            "source_filename": rel,
                            "source_slug": slug,
                            "raw_tokens": raw_tokens,
                            "cleaned_tokens": cleaned_tokens,
                            "n_chunks": len(chunks),
                            "mean_tokens_per_chunk": (
                                statistics.mean(chunk_tokens_this_pdf)
                                if chunk_tokens_this_pdf
                                else 0
                            ),
                            "extraction_method": method,
                            "n_pages": len(pages),
                        }
                    )
                except Exception as e:
                    known_issues.append((rel, f"{type(e).__name__}: {e}"))
                progress.advance(task)
    finally:
        if out_f is not None:
            out_f.close()

    # ── aggregate ──
    n_pdfs_attempted = len(pdfs)
    n_pdfs_with_chunks = sum(1 for s in per_source if s["n_chunks"] > 0)
    total_raw = sum(s["raw_tokens"] for s in per_source)
    total_cleaned = sum(s["cleaned_tokens"] for s in per_source)
    n_chunks = sum(s["n_chunks"] for s in per_source)
    retention = (total_cleaned / total_raw * 100) if total_raw else 0.0

    chunk_token_counts = sorted(c["token_count"] for c in all_chunks_meta)
    if chunk_token_counts:
        cstats = {
            "min": min(chunk_token_counts),
            "mean": statistics.mean(chunk_token_counts),
            "p50": percentile(chunk_token_counts, 0.50),
            "p90": percentile(chunk_token_counts, 0.90),
            "p99": percentile(chunk_token_counts, 0.99),
            "max": max(chunk_token_counts),
        }
    else:
        cstats = {k: 0 for k in ("min", "mean", "p50", "p90", "p99", "max")}

    # ── cost projection ──
    sum_chunk_tokens = sum(chunk_token_counts)
    qa_input_tokens = sum_chunk_tokens + n_chunks * QA_PROMPT_OVERHEAD_TOKENS
    qa_output_tokens = n_chunks * QA_PAIRS_PER_CHUNK * QA_OUTPUT_TOKENS_PER_PAIR
    qa_input_cost = qa_input_tokens / 1_000_000 * SONNET_INPUT_PRICE_PER_1M_USD
    qa_output_cost = qa_output_tokens / 1_000_000 * SONNET_OUTPUT_PRICE_PER_1M_USD
    qa_total_cost = qa_input_cost + qa_output_cost

    sft_pairs = n_chunks * QA_PAIRS_PER_CHUNK
    sft_train_tokens = sft_pairs * SFT_TOKENS_PER_PAIR
    sft_cost = sft_train_tokens / 1_000_000 * TINKER_TRAIN_PRICE_PER_1M_USD

    # ── sample chunks (seeded) ──
    rng = random.Random(SAMPLE_SEED)
    sample_chunks = (
        rng.sample(all_chunks_meta, min(N_SAMPLES, len(all_chunks_meta)))
        if all_chunks_meta
        else []
    )

    # ── write journal ──
    md = render_journal(
        config={
            "tokenizer": "openai/gpt-oss-20b (HF AutoTokenizer)",
            "target_tokens": args.target_tokens,
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
            "header_footer_threshold": HEADER_FOOTER_THRESHOLD,
            "cleaning_version": CLEANING_VERSION,
            "chunking_version": CHUNKING_VERSION,
            "raw_dir": str(raw_dir.relative_to(PROJECT_ROOT)),
            "out_path": str(out_path.relative_to(PROJECT_ROOT)),
            "dry_run": args.dry_run,
        },
        n_pdfs_attempted=n_pdfs_attempted,
        n_pdfs_with_chunks=n_pdfs_with_chunks,
        total_raw=total_raw,
        total_cleaned=total_cleaned,
        retention=retention,
        n_chunks=n_chunks,
        cstats=cstats,
        per_source=per_source,
        sample_chunks=sample_chunks,
        skipped_other=skipped_other,
        known_issues=known_issues,
        cost_proj={
            "qa_input_tokens": qa_input_tokens,
            "qa_output_tokens": qa_output_tokens,
            "qa_input_cost": qa_input_cost,
            "qa_output_cost": qa_output_cost,
            "qa_total_cost": qa_total_cost,
            "sft_pairs": sft_pairs,
            "sft_train_tokens": sft_train_tokens,
            "sft_cost": sft_cost,
        },
    )
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.write_text(md, encoding="utf-8")

    # ── terminal summary ──
    summary = Table(title="Pipeline summary")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("PDFs processed", f"{n_pdfs_with_chunks} / {n_pdfs_attempted}")
    summary.add_row("Total raw tokens", f"{total_raw:,}")
    summary.add_row("Total cleaned tokens", f"{total_cleaned:,}")
    summary.add_row("Retention", f"{retention:.1f}%")
    summary.add_row("Total chunks", f"{n_chunks:,}")
    summary.add_row("Mean tokens/chunk", f"{cstats['mean']:.0f}")
    summary.add_row("p99 tokens/chunk", f"{int(cstats['p99']):,}")
    summary.add_row("Q&A gen est. cost", fmt_money(qa_total_cost))
    summary.add_row("1-epoch SFT est. cost", fmt_money(sft_cost))
    summary.add_row("Known issues", f"{len(known_issues)}")
    console.print(summary)

    console.print(f"[green]wrote[/green] {JOURNAL_PATH.relative_to(PROJECT_ROOT)}")
    if not args.dry_run:
        console.print(
            f"[green]wrote[/green] {out_path.relative_to(PROJECT_ROOT)}  "
            f"({n_chunks:,} chunks)"
        )

    return 0


# ──────────────────────── journal renderer ────────────────────────


def render_journal(
    *,
    config: dict,
    n_pdfs_attempted: int,
    n_pdfs_with_chunks: int,
    total_raw: int,
    total_cleaned: int,
    retention: float,
    n_chunks: int,
    cstats: dict,
    per_source: list[dict],
    sample_chunks: list[dict],
    skipped_other: list[Path],
    known_issues: list[tuple[str, str]],
    cost_proj: dict,
) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Cleaning and chunking — Phase A")
    a("")
    a("_Generated by `scripts/prepare_corpus.py`. Re-running overwrites this file._")
    a("")
    a(f"- Project root: `{PROJECT_ROOT}`")
    a(f"- Output JSONL: `{config['out_path']}`")
    a("")

    a("## 1. Pipeline configuration")
    a("")
    a("| Parameter | Value |")
    a("|---|---|")
    for k in (
        "tokenizer",
        "target_tokens",
        "max_tokens",
        "overlap_tokens",
        "header_footer_threshold",
        "cleaning_version",
        "chunking_version",
        "raw_dir",
        "out_path",
        "dry_run",
    ):
        a(f"| `{k}` | `{config[k]}` |")
    a("")

    a("## 2. Source files processed")
    a("")
    a(f"- PDFs attempted: **{n_pdfs_attempted:,}**")
    a(f"- PDFs with chunks emitted: **{n_pdfs_with_chunks:,}**")
    a(f"- Total raw tokens (post-extraction, pre-clean): **{total_raw:,}**")
    a(f"- Total cleaned tokens: **{total_cleaned:,}**")
    a(f"- Retention after cleaning: **{retention:.1f}%**")
    a("")
    if skipped_other:
        a(f"- Skipped non-PDF / non-manifest files: **{len(skipped_other)}**")
        for f in skipped_other[:20]:
            a(f"  - `{f}`")
        a("")

    a("## 3. Chunking output")
    a("")
    a(f"- Total chunks: **{n_chunks:,}**")
    a("")
    a("| Statistic | Tokens |")
    a("|---|---:|")
    for k, label in (
        ("mean", "mean"),
        ("min", "min"),
        ("p50", "p50 (median)"),
        ("p90", "p90"),
        ("p99", "p99"),
        ("max", "max"),
    ):
        a(f"| {label} | {int(cstats[k]):,} |")
    a("")

    a("## 4. Per-source breakdown")
    a("")
    a("| Source | Raw tokens | Cleaned tokens | Chunks | Mean tokens/chunk |")
    a("|---|---:|---:|---:|---:|")
    for s in sorted(per_source, key=lambda x: -x["cleaned_tokens"]):
        a(
            f"| `{s['source_filename']}` | {s['raw_tokens']:,} | "
            f"{s['cleaned_tokens']:,} | {s['n_chunks']:,} | "
            f"{s['mean_tokens_per_chunk']:.0f} |"
        )
    a("")

    a("## 5. Sample chunks (3 random, seeded)")
    a("")
    a(
        f"`random.seed({SAMPLE_SEED})`, `random.Random.sample`. "
        "Full chunk texts below — sanity-check that cleaning preserved "
        "coherent prose and chunks read as self-contained context."
    )
    a("")
    for i, c in enumerate(sample_chunks, 1):
        a(f"### Sample {i}")
        a("")
        a(f"- **chunk_id:** `{c['chunk_id']}`")
        a(f"- **source_filename:** `{c['source_filename']}`")
        a(
            f"- **chunk_index:** {c['chunk_index']} of "
            f"{c['total_chunks_in_source']}"
        )
        a(f"- **token_count:** {c['token_count']:,}")
        a(f"- **char_count:** {c['char_count']:,}")
        a(
            f"- **char offsets:** {c['char_offset_start']:,} → "
            f"{c['char_offset_end']:,}"
        )
        a(f"- **extraction_method:** `{c['extraction_method']}`")
        if c["manifest_source_id"]:
            a(
                f"- **manifest.source_id:** `{c['manifest_source_id']}`  "
                f"(license `{c['manifest_license']}`)"
            )
        a("")
        a("```text")
        a(c["text"].replace("```", "ʼʼʼ"))
        a("```")
        a("")

    a("## 6. Cost projection (placeholders — verify before real spend)")
    a("")
    a(
        f"Assumes **{QA_PAIRS_PER_CHUNK} Q&A pairs per chunk**, generated via one "
        "Claude Sonnet call per chunk. Pricing **placeholders**: "
        f"`${SONNET_INPUT_PRICE_PER_1M_USD}/1M input`, "
        f"`${SONNET_OUTPUT_PRICE_PER_1M_USD}/1M output` for Sonnet; "
        f"`${TINKER_TRAIN_PRICE_PER_1M_USD}/1M training tokens` for Tinker."
    )
    a("")
    a("### (a) Q&A generation (Claude Sonnet)")
    a("")
    a(f"- Calls (one per chunk): **{n_chunks:,}**")
    a(
        f"- Estimated input tokens (chunk + ~{QA_PROMPT_OVERHEAD_TOKENS} "
        f"overhead/call): **{cost_proj['qa_input_tokens']:,}**"
    )
    a(
        f"- Estimated output tokens "
        f"({QA_PAIRS_PER_CHUNK} pairs × ~{QA_OUTPUT_TOKENS_PER_PAIR} tokens): "
        f"**{cost_proj['qa_output_tokens']:,}**"
    )
    a(f"- Input cost: **{fmt_money(cost_proj['qa_input_cost'])}**")
    a(f"- Output cost: **{fmt_money(cost_proj['qa_output_cost'])}**")
    a(f"- **Total Q&A generation: {fmt_money(cost_proj['qa_total_cost'])}**")
    a("")
    a("### (b) 1-epoch SFT on resulting Q&A dataset (Tinker)")
    a("")
    a(
        f"- Q&A pairs: **{cost_proj['sft_pairs']:,}**  "
        f"({n_chunks:,} chunks × {QA_PAIRS_PER_CHUNK})"
    )
    a(
        f"- Tokens/pair (≈ Q + A, context-free SFT): **{SFT_TOKENS_PER_PAIR}** "
        "(placeholder)"
    )
    a(
        f"- Total training tokens (1 epoch): "
        f"**{cost_proj['sft_train_tokens']:,}**"
    )
    a(
        f"- **Estimated 1-epoch SFT cost: "
        f"{fmt_money(cost_proj['sft_cost'])}**"
    )
    a("")

    a("## 7. Known issues")
    a("")
    if known_issues:
        a("| File | Reason |")
        a("|---|---|")
        for path, err in known_issues:
            err_short = err.replace("\n", " ").strip()
            if len(err_short) > 220:
                err_short = err_short[:220] + "…"
            a(f"| `{path}` | {err_short} |")
    else:
        a("_None._")
    a("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
