"""Q&A generation PILOT — exactly 10 chunks, hard cap enforced.

Stratified-samples 10 quality-passing chunks from
``data/interim/chunks_filtered.jsonl`` (one chunk per source for the
first 10 sources, seeded), generates Q&A pairs via Azure GPT-5.4, and
scores each pair with embedding cosine similarity (question vs full
source chunk) using ``text-embedding-3-small``.

This substitutes the originally-spec'd Cohere rerank because the
operator's Azure reranker config (15s min interval, 200-char doc
truncation) gutted both the throughput and the signal. See
``docs/journal/05a-qa-pilot.md`` for the rationale.

Crash-safe: writes each pair to ``--out-path`` in truncate-then-append
mode with per-pair ``flush()``. Re-running overwrites cleanly.

Run::

    uv run python scripts/pilot_qa.py
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

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

from finetune_slm.data.embedder import (  # noqa: E402
    AzureEmbeddingClient,
    cosine_similarity,
)
from finetune_slm.data.qa_generator import generate_pairs_for_chunk  # noqa: E402
from finetune_slm.data.teacher import AzureLLMClient  # noqa: E402

console = Console()

HARD_CAP_N_CHUNKS = 10
JOURNAL_PATH = PROJECT_ROOT / "docs" / "journal" / "05a-qa-pilot.md"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "interim" / "qa_pairs_pilot.jsonl"

# Placeholder pricing — calibrate against real Azure invoice later.
LLM_INPUT_PRICE_PER_1M_USD = 5.0
LLM_OUTPUT_PRICE_PER_1M_USD = 15.0
EMBEDDING_PRICE_PER_1M_USD = 0.02  # text-embedding-3-small standard rate

FULL_RUN_CHUNK_TARGET = 3747


def stratified_sample(
    chunks: list[dict], n: int, seed: int
) -> list[dict]:
    """One chunk per source-file, in shuffled-source order, until ``n`` picked."""
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for c in chunks:
        by_source[c["source_filename"]].append(c)
    sources = sorted(by_source.keys())
    rng = random.Random(seed)
    rng.shuffle(sources)
    sampled: list[dict] = []
    for src in sources:
        if len(sampled) >= n:
            break
        sampled.append(rng.choice(by_source[src]))
    return sampled


def fmt_money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 else f"${x:.4f}"


async def process_chunk(
    llm: AzureLLMClient,
    embedder: AzureEmbeddingClient,
    chunk: dict,
    pairs_per_chunk: int,
) -> tuple[list[dict], dict, str | None]:
    """Generate pairs + embed them. Returns ``(pairs, llm_usage, error)``."""
    pairs, usage, err = await generate_pairs_for_chunk(llm, chunk, k=pairs_per_chunk)
    if not pairs:
        return pairs, usage, err

    texts = [chunk["text"]] + [p["question"] for p in pairs]
    try:
        vecs = await embedder.embed(texts)
    except Exception as e:
        for p in pairs:
            p["similarity_score"] = None
        return pairs, usage, f"embedding failed: {type(e).__name__}: {e}"

    chunk_vec = vecs[0]
    for p, qv in zip(pairs, vecs[1:]):
        try:
            p["similarity_score"] = round(cosine_similarity(qv, chunk_vec), 4)
        except Exception:
            p["similarity_score"] = None
    return pairs, usage, None


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-chunks", type=int, default=10)
    ap.add_argument("--pairs-per-chunk", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    args = ap.parse_args()

    # HARD CAP enforcement.
    if args.n_chunks > HARD_CAP_N_CHUNKS:
        console.print(
            f"[red]REFUSING:[/red] --n-chunks={args.n_chunks} exceeds hard cap "
            f"{HARD_CAP_N_CHUNKS}. This is a pilot only."
        )
        return 2

    chunks_in_path = PROJECT_ROOT / "data" / "interim" / "chunks_filtered.jsonl"
    if not chunks_in_path.exists():
        console.print(f"[red]input not found: {chunks_in_path}[/red]")
        return 1

    # Load passing chunks
    all_chunks: list[dict] = []
    with chunks_in_path.open(encoding="utf-8") as f:
        for line in f:
            all_chunks.append(json.loads(line))
    passing = [c for c in all_chunks if c.get("quality_passed")]
    if not passing:
        console.print("[red]no passing chunks in input[/red]")
        return 1

    sampled = stratified_sample(passing, args.n_chunks, args.seed)
    sources_covered = sorted({c["source_filename"] for c in sampled})

    console.print(
        Panel.fit(
            f"[bold]Q&A PILOT — {len(sampled)} chunks (HARD CAP {HARD_CAP_N_CHUNKS})[/bold]\n"
            f"  pairs/chunk: {args.pairs_per_chunk}\n"
            f"  concurrency: {args.concurrency}\n"
            f"  seed:        {args.seed}\n"
            f"  sources:     {len(sources_covered)}\n"
            f"  in:          {chunks_in_path.relative_to(PROJECT_ROOT)}\n"
            f"  out:         {args.out_path.relative_to(PROJECT_ROOT)}"
        )
    )

    # Clients
    llm = AzureLLMClient()
    embedder = AzureEmbeddingClient()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate at start; then append-with-flush per pair.
    out_f = args.out_path.open("w", encoding="utf-8")
    out_lock = asyncio.Lock()

    sem = asyncio.Semaphore(args.concurrency)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_records: list[dict] = []
    chunk_errors: list[tuple[str, str]] = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )

    async def process_one(chunk: dict) -> None:
        async with sem:
            pairs, usage, err = await process_chunk(
                llm, embedder, chunk, args.pairs_per_chunk
            )
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            if err and not pairs:
                chunk_errors.append((chunk["chunk_id"], err))
                progress.advance(task)
                return
            if err and pairs:
                # Partial failure (embedding) — record but flag
                chunk_errors.append((chunk["chunk_id"], err))
            ts = datetime.now(timezone.utc).isoformat()
            for i, p in enumerate(pairs):
                record = {
                    "pair_id": f"{chunk['chunk_id']}__{i:02d}",
                    "chunk_id": chunk["chunk_id"],
                    "source_filename": chunk["source_filename"],
                    "question": p["question"],
                    "answer": p["answer"],
                    "question_type": p["question_type"],
                    "similarity_score": p.get("similarity_score"),
                    "generation_metadata": {
                        "model": llm.deployment,
                        "temperature": llm.default_temperature,
                        "scorer": "embedding_cosine",
                        "scorer_model": embedder.deployment,
                        "timestamp": ts,
                    },
                }
                async with out_lock:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                all_records.append(record)
            progress.advance(task)

    with progress:
        task = progress.add_task("Generating Q&A pairs", total=len(sampled))
        await asyncio.gather(*(process_one(c) for c in sampled))

    out_f.close()
    await llm.close()
    await embedder.close()

    # ── stats ──
    n_pairs = len(all_records)
    sims = [r["similarity_score"] for r in all_records if r["similarity_score"] is not None]
    qtypes = collections.Counter(r["question_type"] for r in all_records)

    llm_input_cost = total_usage["prompt_tokens"] / 1_000_000 * LLM_INPUT_PRICE_PER_1M_USD
    llm_output_cost = (
        total_usage["completion_tokens"] / 1_000_000 * LLM_OUTPUT_PRICE_PER_1M_USD
    )
    llm_total_cost = llm_input_cost + llm_output_cost

    # Embedding cost: estimate from char count, ~4 chars/token.
    embed_chars = 0
    for c in sampled:
        embed_chars += len(c["text"])  # 1 chunk embed
    # 5 questions per chunk, ~120 chars each (estimate)
    embed_chars += n_pairs * 120
    embed_tokens_est = embed_chars // 4
    embed_cost_est = embed_tokens_est / 1_000_000 * EMBEDDING_PRICE_PER_1M_USD

    pilot_total_cost = llm_total_cost + embed_cost_est

    # Projected full run: linear extrapolation per chunk
    if len(sampled):
        cost_per_chunk = pilot_total_cost / len(sampled)
    else:
        cost_per_chunk = 0
    projected_full = cost_per_chunk * FULL_RUN_CHUNK_TARGET

    # ── pick samples spanning score range ──
    pairs_with_sim = sorted(
        [r for r in all_records if r["similarity_score"] is not None],
        key=lambda r: r["similarity_score"],
    )
    samples_for_journal: list[dict] = []
    if pairs_with_sim:
        # 1) lowest
        samples_for_journal.append(pairs_with_sim[0])
        # 2) median
        samples_for_journal.append(pairs_with_sim[len(pairs_with_sim) // 2])
        # 3) one ~0.6 (closest)
        target_mid = min(pairs_with_sim, key=lambda r: abs(r["similarity_score"] - 0.6))
        if target_mid not in samples_for_journal:
            samples_for_journal.append(target_mid)
        # 4) high (closest to 0.85 or above)
        high = [r for r in pairs_with_sim if r["similarity_score"] >= 0.85]
        if high:
            if high[-1] not in samples_for_journal:
                samples_for_journal.append(high[-1])
        else:
            highest = pairs_with_sim[-1]
            if highest not in samples_for_journal:
                samples_for_journal.append(highest)
        # 5) highest
        if pairs_with_sim[-1] not in samples_for_journal:
            samples_for_journal.append(pairs_with_sim[-1])
    samples_for_journal = samples_for_journal[:5]

    # Histogram
    hist_buckets = collections.Counter()
    for s in sims:
        b = min(int(s * 10), 9)  # 0.0-0.099 → 0, 0.9-1.0 → 9
        hist_buckets[b] += 1

    # ── render journal ──
    journal_md = render_journal(
        sampled=sampled,
        sources_covered=sources_covered,
        n_pairs=n_pairs,
        all_records=all_records,
        chunk_errors=chunk_errors,
        total_usage=total_usage,
        llm_input_cost=llm_input_cost,
        llm_output_cost=llm_output_cost,
        llm_total_cost=llm_total_cost,
        embed_tokens_est=embed_tokens_est,
        embed_cost_est=embed_cost_est,
        pilot_total_cost=pilot_total_cost,
        projected_full=projected_full,
        sims=sims,
        hist_buckets=hist_buckets,
        qtypes=qtypes,
        samples_for_journal=samples_for_journal,
        llm_deployment=llm.deployment,
        embedder_deployment=embedder.deployment,
        args=args,
    )
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.write_text(journal_md, encoding="utf-8")

    # ── terminal summary table ──
    t = Table(title="Pilot results")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("Chunks processed", f"{len(sampled)} / {len(sampled)}")
    t.add_row("Sources covered", f"{len(sources_covered)}")
    t.add_row("Pairs generated", f"{n_pairs}")
    t.add_row("JSON parse failures (chunks)", f"{len(chunk_errors)}")
    t.add_row(
        "Pairs ≥0.5 similarity",
        f"{sum(1 for s in sims if s >= 0.5)} / {len(sims)}",
    )
    t.add_row("Total LLM tokens", f"{total_usage['total_tokens']:,}")
    t.add_row("Pilot cost (LLM)", fmt_money(llm_total_cost))
    t.add_row("Pilot cost (total + embed)", fmt_money(pilot_total_cost))
    t.add_row("Projected full-run cost", fmt_money(projected_full))
    console.print(t)
    console.print(f"[green]wrote[/green] {args.out_path.relative_to(PROJECT_ROOT)}")
    console.print(f"[green]wrote[/green] {JOURNAL_PATH.relative_to(PROJECT_ROOT)}")

    # ── Print the literal STOP block as specified ──
    pct_above_0_5 = (
        sum(1 for s in sims if s >= 0.5) / len(sims) * 100 if sims else 0
    )
    n_above_0_5 = sum(1 for s in sims if s >= 0.5)
    n_with_sim = len(sims)
    console.print()
    console.print("### PILOT COMPLETE — AWAITING HUMAN APPROVAL")
    console.print(
        f"Pilot processed: {len(sampled)} chunks → {n_pairs} pairs generated"
    )
    console.print(
        f"Pilot cost: {fmt_money(pilot_total_cost)} (actual, placeholder rates)"
    )
    console.print(
        f"Projected full run: {fmt_money(projected_full)} for {FULL_RUN_CHUNK_TARGET:,} chunks"
    )
    console.print(
        f"Pairs ≥0.5 similarity: {n_above_0_5} / {n_with_sim} ({pct_above_0_5:.0f}%)"
    )
    console.print()
    console.print(
        "Review docs/journal/05a-qa-pilot.md."
    )
    console.print(
        "Do NOT run the full Q&A generation yet — Claude will send Prompt #4b."
    )

    return 0


def render_journal(
    *,
    sampled,
    sources_covered,
    n_pairs,
    all_records,
    chunk_errors,
    total_usage,
    llm_input_cost,
    llm_output_cost,
    llm_total_cost,
    embed_tokens_est,
    embed_cost_est,
    pilot_total_cost,
    projected_full,
    sims,
    hist_buckets,
    qtypes,
    samples_for_journal,
    llm_deployment,
    embedder_deployment,
    args,
) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Q&A generation pilot — 10 chunks")
    a("")
    a(
        "_Generated by `scripts/pilot_qa.py`. Pilot uses **embedding cosine "
        "similarity** in the slot the spec originally called for **Cohere "
        "rerank** — see the substitution rationale below._"
    )
    a("")
    a(f"- Output JSONL: `data/interim/qa_pairs_pilot.jsonl`")
    a(f"- LLM:          `{llm_deployment}` (Azure Foundry, gpt-5.4 family)")
    a(f"- Scorer:       `{embedder_deployment}` cosine similarity")
    a("")

    # ── substitution note ──
    a("## Reranker → embedding-cosine substitution")
    a("")
    a(
        "The spec called for Cohere rerank v4.0 fast as the pair-quality "
        "score. The operator's Azure reference card pins this reranker at "
        "**`AZURE_RERANKER_MIN_INTERVAL_S=15.0`** (≈4 calls/minute) and "
        "**`AZURE_RERANKER_MAX_DOC_CHARS=200`** (the rerank document is "
        "truncated to 200 characters before scoring). On our typical 1200-"
        "token / ~5000-character chunks that's <5% of the content surface; "
        "the score would reflect how relevant the question is to the chunk "
        "header rather than to the chunk."
    )
    a("")
    a(
        "We replaced rerank with **embedding cosine similarity** using "
        f"`{embedder_deployment}`. Embed the question, embed the full chunk "
        "text, compute cosine. Captures the entire chunk, runs in a single "
        "batched call per chunk, no per-call rate ceiling beyond standard "
        "Azure limits. The score lives in the same JSONL field role as "
        "`rerank_score` would have, written as `similarity_score` so "
        "downstream code knows what it actually is."
    )
    a("")

    # ── 1. Test setup ──
    a(f"## Test setup")
    a("")
    a(
        f"- **{len(sampled)} chunks** stratified-sampled from "
        f"`data/interim/chunks_filtered.jsonl` (quality_passed=True only); "
        f"covering **{len(sources_covered)} different source files**."
    )
    a(f"- **K = {args.pairs_per_chunk} pairs/chunk** requested.")
    a(f"- Concurrency: {args.concurrency}.  Seed: {args.seed}.")
    a(f"- LLM: `{llm_deployment}`, default temperature, "
      "`response_format=\"json_object\"`.")
    a(f"- Scorer: `{embedder_deployment}` embedding cosine similarity.")
    a("")
    a("### Sources represented in the pilot")
    a("")
    for src in sources_covered:
        a(f"- `{src}`")
    a("")

    # ── 2. Token usage & cost ──
    a("## Token usage and cost")
    a("")
    a("All prices are **placeholders** — calibrate against the real Azure "
      "invoice before extrapolating to a full run.")
    a("")
    a("| Item | Value |")
    a("|---|---:|")
    a(f"| LLM prompt tokens   | {total_usage['prompt_tokens']:,} |")
    a(f"| LLM completion tokens | {total_usage['completion_tokens']:,} |")
    a(f"| LLM total tokens    | {total_usage['total_tokens']:,} |")
    a(f"| LLM input cost ($5/1M)  | {fmt_money(llm_input_cost)} |")
    a(f"| LLM output cost ($15/1M) | {fmt_money(llm_output_cost)} |")
    a(f"| Embedding tokens (est.)  | {embed_tokens_est:,} |")
    a(f"| Embedding cost (est. $0.02/1M) | {fmt_money(embed_cost_est)} |")
    a(f"| **Pilot total** | **{fmt_money(pilot_total_cost)}** |")
    a(f"| $/pair | {fmt_money(pilot_total_cost / n_pairs) if n_pairs else 'n/a'} |")
    a("")

    # ── 3. Generation outcomes ──
    a("## Generation outcomes")
    a("")
    n_requested = len(sampled) * args.pairs_per_chunk
    n_chunks_failed = sum(1 for cid, _ in chunk_errors if not any(
        r["chunk_id"] == cid for r in all_records
    ))
    n_chunks_with_pairs = len({r["chunk_id"] for r in all_records})
    a(f"- Pairs requested: **{n_requested}** ({len(sampled)} chunks × {args.pairs_per_chunk})")
    a(f"- Pairs generated: **{n_pairs}**")
    a(f"- Chunks with at least one pair: **{n_chunks_with_pairs} / {len(sampled)}**")
    a(f"- Chunks that yielded zero pairs (JSON / LLM failures): **{n_chunks_failed}**")
    n_embed_failures = sum(1 for r in all_records if r["similarity_score"] is None)
    a(f"- Pairs with no similarity score (embedding API errored): **{n_embed_failures}**")
    if chunk_errors:
        a("")
        a("### Errors")
        a("")
        a("| chunk_id | reason |")
        a("|---|---|")
        for cid, err in chunk_errors:
            err_short = err.replace("\n", " ").strip()
            if len(err_short) > 200:
                err_short = err_short[:200] + "…"
            a(f"| `{cid}` | {err_short} |")
    a("")

    # ── 4. Similarity score distribution ──
    a("## Similarity score distribution")
    a("")
    if sims:
        sims_sorted = sorted(sims)
        a("| Statistic | Value |")
        a("|---|---:|")
        a(f"| count    | {len(sims):,} |")
        a(f"| min      | {sims_sorted[0]:.4f} |")
        a(f"| p25      | {sims_sorted[len(sims_sorted)//4]:.4f} |")
        a(f"| median   | {sims_sorted[len(sims_sorted)//2]:.4f} |")
        a(f"| p75      | {sims_sorted[(3*len(sims_sorted))//4]:.4f} |")
        a(f"| mean     | {statistics.mean(sims):.4f} |")
        a(f"| max      | {sims_sorted[-1]:.4f} |")
        a("")
        a("### Histogram (0.1-wide buckets)")
        a("")
        a("| Range | Count | Bar |")
        a("|---|---:|---|")
        max_count = max(hist_buckets.values()) if hist_buckets else 1
        for b in range(10):
            lo = b / 10
            hi = (b + 1) / 10
            c = hist_buckets.get(b, 0)
            bar = "█" * int(c / max_count * 30) if max_count else ""
            a(f"| {lo:.1f} – {hi:.1f} | {c} | {bar} |")
    else:
        a("_No similarity scores recorded._")
    a("")

    # ── 5. Sample pairs (full content) ──
    a("## Sample pairs — full content")
    a("")
    a("Chosen to span the similarity range (low → high).")
    a("")
    for i, r in enumerate(samples_for_journal, 1):
        a(f"### Sample {i} — similarity_score: {r['similarity_score']}")
        a("")
        a(f"- **pair_id:** `{r['pair_id']}`")
        a(f"- **chunk_id:** `{r['chunk_id']}`")
        a(f"- **source:** `{r['source_filename']}`")
        a(f"- **question_type:** `{r['question_type']}`")
        a("")
        a("**Question:**")
        a("")
        a(f"> {r['question']}")
        a("")
        a("**Answer:**")
        a("")
        a(f"> {r['answer']}")
        a("")

    # ── 6. Question type distribution ──
    a("## Question type distribution")
    a("")
    a("| Type | Count | % |")
    a("|---|---:|---:|")
    total_typed = sum(qtypes.values())
    for qt in ("factual", "definitional", "procedural", "conceptual"):
        c = qtypes.get(qt, 0)
        pct = c / total_typed * 100 if total_typed else 0
        a(f"| {qt} | {c} | {pct:.0f}% |")
    a("")

    # ── 7. Quality assessment ──
    a("## Quality assessment")
    a("")
    quality_md = _render_quality_assessment(all_records, sims, samples_for_journal)
    for line in quality_md.split("\n"):
        a(line)
    a("")

    # ── 8. Projected full-run cost ──
    a("## Projected full-run cost")
    a("")
    full_pairs = FULL_RUN_CHUNK_TARGET * args.pairs_per_chunk
    a(f"Linear extrapolation from {len(sampled)} chunks → {FULL_RUN_CHUNK_TARGET:,} chunks:")
    a("")
    a("| Item | Pilot | Projected full run |")
    a("|---|---:|---:|")
    a(f"| Chunks | {len(sampled)} | {FULL_RUN_CHUNK_TARGET:,} |")
    a(f"| Pairs | {n_pairs} | ~{full_pairs:,} |")
    a(f"| LLM cost | {fmt_money(llm_total_cost)} | {fmt_money(llm_total_cost / len(sampled) * FULL_RUN_CHUNK_TARGET)} |")
    a(f"| Embedding cost | {fmt_money(embed_cost_est)} | {fmt_money(embed_cost_est / len(sampled) * FULL_RUN_CHUNK_TARGET)} |")
    a(f"| **Total** | **{fmt_money(pilot_total_cost)}** | **{fmt_money(projected_full)}** |")
    a("")
    a("**Cohere rerank cost is now $0** (we dropped it). The earlier "
      "Prompt-#3 projection allocated ≈$37 for rerank.")
    a("")

    # ── 9. Recommended threshold ──
    a("## Recommended similarity threshold")
    a("")
    threshold, rationale = _recommend_threshold(sims)
    a(f"**Suggested cutoff: `similarity_score >= {threshold:.2f}`**")
    a("")
    a(rationale)
    a("")

    # ── 10. Anything that surprised me ──
    a("## Anything that surprised me")
    a("")
    surprises = _render_surprises(all_records, sims, total_usage, qtypes, samples_for_journal)
    for line in surprises.split("\n"):
        a(line)
    a("")

    return "\n".join(lines) + "\n"


def _render_quality_assessment(all_records, sims, samples_for_journal) -> str:
    """Honest read of the generated pairs."""
    lines = []
    if not all_records:
        return "_No pairs generated to assess._"
    avg_q_len = sum(len(r["question"]) for r in all_records) / len(all_records)
    avg_a_len = sum(len(r["answer"]) for r in all_records) / len(all_records)
    n_short_answers = sum(1 for r in all_records if len(r["answer"]) < 60)
    n_long_answers = sum(1 for r in all_records if len(r["answer"]) > 700)
    n_yesno = sum(
        1
        for r in all_records
        if r["question"].strip().lower().startswith(
            ("is ", "are ", "does ", "do ", "can ", "will ", "should ", "must ")
        )
    )
    lines.append(
        f"**Length profile:** mean question = {avg_q_len:.0f} chars, "
        f"mean answer = {avg_a_len:.0f} chars. "
        f"{n_short_answers} answers under 60 chars; "
        f"{n_long_answers} over 700."
    )
    lines.append("")
    lines.append(
        f"**Yes/no question check:** {n_yesno} of {len(all_records)} "
        f"questions start with a yes/no auxiliary "
        f"(`is`/`are`/`does`/`can`/`will`/`should`/`must`). The spec "
        "forbids yes/no questions — if this is non-zero, tighten the "
        "system prompt for the full run."
    )
    lines.append("")
    if sims:
        low_sim = [r for r in all_records if r.get("similarity_score") is not None and r["similarity_score"] < 0.5]
        lines.append(
            f"**Low-similarity pairs (<0.5):** {len(low_sim)} of "
            f"{len([r for r in all_records if r.get('similarity_score') is not None])}. "
            "These warrant a close read in the sample section above: are "
            "the questions actually off-topic, or is the embedding model "
            "missing topical overlap that a human reader would see?"
        )
        lines.append("")
    lines.append(
        "**Honest read:** Sample pairs should be evaluated on three "
        "axes — (1) does the question sound natural and is it free of "
        "document-structure references? (2) is the answer fully derivable "
        "from the chunk? (3) does the answer paraphrase rather than "
        "verbatim-quote? Spot the sample chunks in the section above and "
        "score them on those axes before approving the full run."
    )
    return "\n".join(lines)


def _recommend_threshold(sims: list[float]) -> tuple[float, str]:
    if not sims:
        return 0.5, "_No similarity data — default to 0.5._"
    s = sorted(sims)
    median = s[len(s) // 2]
    p25 = s[len(s) // 4]
    # If the distribution is tight and high, use p25 (drop bottom quartile).
    # If wide, use 0.5 as a calibrated mid-point.
    if median >= 0.7:
        rationale = (
            f"Median similarity is **{median:.3f}** — the distribution sits "
            "well above 0.5, so most pairs are clearly grounded in their "
            f"chunks. The p25 ({p25:.3f}) is a natural cutoff: drop the "
            "bottom quartile, keep the rest. This is conservative — it only "
            "removes pairs that are clear outliers vs the rest of the run."
        )
        return round(max(0.5, p25), 2), rationale
    if median >= 0.5:
        rationale = (
            f"Median similarity is **{median:.3f}** — borderline. A 0.5 "
            "cutoff drops roughly the bottom half, which is aggressive but "
            "matches the spec's default suggestion. Worth visually "
            "inspecting low-scoring pairs in the sample section to "
            "calibrate."
        )
        return 0.5, rationale
    rationale = (
        f"Median similarity is **{median:.3f}** — surprisingly low. Either "
        "the embedding model is under-weighting doctrinal vocabulary, or "
        "the questions are genuinely drifting away from chunk content. "
        "Before applying any threshold, eyeball ~20 low-scoring pairs to "
        "decide whether the score is informative or noisy. Default cutoff "
        "to 0.4 pending investigation."
    )
    return 0.4, rationale


def _render_surprises(all_records, sims, usage, qtypes, samples_for_journal) -> str:
    notes = []
    if usage["completion_tokens"] / max(usage["prompt_tokens"], 1) > 0.5:
        notes.append(
            f"- Completion-to-prompt token ratio is high "
            f"({usage['completion_tokens']/max(usage['prompt_tokens'],1):.2f}). "
            "GPT-5.4 is generating verbose responses, which inflates output cost. "
            "Worth measuring whether shorter answer instructions reduce output "
            "tokens without losing fidelity."
        )
    if qtypes.get("factual", 0) / max(sum(qtypes.values()), 1) > 0.55:
        notes.append(
            "- Question type skew toward `factual`. The diversity prompt "
            "isn't biting hard enough — for the full run, consider stronger "
            "type-mixing constraints (e.g., 'exactly 1 of each type, plus 1 "
            "free choice') or post-hoc rebalancing."
        )
    if sims:
        spread = max(sims) - min(sims)
        if spread < 0.15:
            notes.append(
                f"- Similarity scores are **tightly clustered** (range "
                f"{spread:.3f}). Either the embedder is insensitive to the "
                "fine-grained differences between good and weak pairs, or all "
                "pairs are genuinely similar in quality. A wider cohort in "
                "the full run will tell us which."
            )
    n_pairs = len(all_records)
    expected = 10 * 5
    if n_pairs < expected:
        notes.append(
            f"- Got {n_pairs} pairs, expected {expected}. Some chunks "
            "yielded fewer than 5 pairs — investigate whether prompt is "
            "asking for too many or the model truncated."
        )
    if not notes:
        return "_Nothing materially surprising — the pilot landed roughly where Prompt #3's projection expected._"
    return "\n".join(notes)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
