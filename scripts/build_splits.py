"""Build train/val/test/held-out-source splits in ChatML format.

Input:  ``data/interim/qa_pairs.jsonl`` (read-only).
Output: four JSONL files under ``data/processed/qa/`` plus a journal entry
under ``docs/journal/06-chatml-formatting.md``.

Decisions baked in (constants):
  * SYSTEM_PROMPT — the exact wording the model sees as its system message.
  * HELD_OUT_SOURCES — folder-prefixes of sources reserved for harder eval.
  * RANDOM_SEED — drives both the main-bucket shuffle and any local sampling.

Run:

    uv run python scripts/build_splits.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make `from finetune_slm...` importable when running directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finetune_slm.data.chatml import (  # noqa: E402
    MAX_TOKENS,
    MIN_TOKENS,
    TokenizerInfo,
    make_tokenizer,
    to_chatml,
    validate_chatml,
)

from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table  # noqa: E402

SYSTEM_PROMPT = (
    "You are an expert assistant on Unmanned Aerial Vehicle (UAV) combat "
    "doctrine. Provide accurate, concise answers grounded in US, NATO, and "
    "allied doctrinal publications."
)
HELD_OUT_SOURCES = {"crs_2026_law_enforcement_cuas", "faa_notice_uas_lost_link"}
RANDOM_SEED = 42
TRAIN_FRAC = 0.85
VAL_FRAC = 0.05
TEST_FRAC = 0.10

console = Console()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_pairs(path: Path) -> list[dict]:
    pairs: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_folder(source_filename: str) -> str:
    return source_filename.split("/", 1)[0]


def held_out_prefix(source_filename: str) -> str | None:
    folder = source_folder(source_filename)
    for prefix in HELD_OUT_SOURCES:
        if folder == prefix or folder.startswith(prefix + "_"):
            return prefix
    return None


# ---------------------------------------------------------------------------
# Validation + token counting
# ---------------------------------------------------------------------------


def convert_and_validate(
    pairs: list[dict],
    tokenizer: TokenizerInfo,
    label: str,
) -> tuple[list[dict], list[int], Counter]:
    """Convert pairs to ChatML and validate.

    Returns (examples, token_counts, drop_reasons). Token counts are
    aligned 1:1 with examples (i.e., only kept examples).
    """
    kept: list[dict] = []
    token_counts: list[int] = []
    drops: Counter = Counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"validating ({label})", total=len(pairs))
        for pair in pairs:
            example = to_chatml(pair, SYSTEM_PROMPT)
            ok, reason = validate_chatml(example, tokenizer)
            if not ok:
                drops[reason or "unknown"] += 1
            else:
                kept.append(example)
                token_counts.append(len(tokenizer.apply_chat_template(example["messages"])))
            progress.advance(task)

    return kept, token_counts, drops


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def shuffle_and_split(
    examples: list[dict],
    token_counts: list[int],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple[
    tuple[list[dict], list[int]],
    tuple[list[dict], list[int]],
    tuple[list[dict], list[int]],
]:
    """Chunk-level shuffle + 85/5/10 partition by chunk_id.

    Pairs from the same chunk_id always land together — leakage is impossible
    by construction. (The spec literally said "shuffle pairs, then move leaked
    chunks to majority split", but with ~5 pairs per chunk and a 5% val ratio
    that algorithm gutted val/test down to ~30/150 from ~935/1,873; the user
    selected this chunk-level variant in lieu of the literal spec.)
    """
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"fractions must sum to 1.0; got {train_frac}+{val_frac}+{test_frac}"
        )

    by_chunk: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for ex, tc in zip(examples, token_counts):
        by_chunk[ex["chunk_id"]].append((ex, tc))

    chunk_ids = sorted(by_chunk.keys())  # determinism before shuffle
    rng = random.Random(seed)
    rng.shuffle(chunk_ids)

    n = len(chunk_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = chunk_ids[:n_train]
    val_ids = chunk_ids[n_train : n_train + n_val]
    test_ids = chunk_ids[n_train + n_val :]

    def gather(ids: list[str]) -> tuple[list[dict], list[int]]:
        exs: list[dict] = []
        tcs: list[int] = []
        for cid in ids:
            for ex, tc in by_chunk[cid]:
                exs.append(ex)
                tcs.append(tc)
        return exs, tcs

    return gather(train_ids), gather(val_ids), gather(test_ids)


def enforce_no_chunk_leakage(
    splits: dict[str, tuple[list[dict], list[int]]],
) -> tuple[dict[str, tuple[list[dict], list[int]]], dict[str, str], int]:
    """No-op for the chunk-level splitter — pairs from one chunk always land
    in one split. Kept as a defensive double-check.
    """
    chunk_counts: dict[str, Counter] = defaultdict(Counter)
    for split_name, (exs, _) in splits.items():
        for ex in exs:
            chunk_counts[ex["chunk_id"]][split_name] += 1
    moved_chunks = sum(1 for counts in chunk_counts.values() if len(counts) > 1)
    return splits, {}, moved_chunks


def assert_no_leakage(splits: dict[str, tuple[list[dict], list[int]]]) -> None:
    sets = {name: {ex["chunk_id"] for ex in exs} for name, (exs, _) in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise SystemExit(
                f"STOP: chunk_id leakage between {a!r} and {b!r}: "
                f"{sorted(overlap)[:5]} ... ({len(overlap)} total)"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_values[lower]
    frac = pos - lower
    return int(round(sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac))


def token_stats(token_counts: list[int]) -> dict[str, int]:
    if not token_counts:
        return {k: 0 for k in ("min", "p10", "p25", "median", "p75", "p90", "p99", "max", "mean")}
    sv = sorted(token_counts)
    return {
        "min": sv[0],
        "p10": percentile(sv, 0.10),
        "p25": percentile(sv, 0.25),
        "median": percentile(sv, 0.50),
        "p75": percentile(sv, 0.75),
        "p90": percentile(sv, 0.90),
        "p99": percentile(sv, 0.99),
        "max": sv[-1],
        "mean": int(round(sum(sv) / len(sv))),
    }


def histogram(values: list[int], n_buckets: int = 10) -> list[tuple[int, int, int]]:
    """Return [(lo, hi, count), ...] using equal-width buckets over [min, max]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [(lo, hi, len(values))]
    width = (hi - lo) / n_buckets
    edges = [lo + i * width for i in range(n_buckets + 1)]
    counts = [0] * n_buckets
    for v in values:
        # last edge is inclusive
        idx = min(int((v - lo) / width), n_buckets - 1)
        counts[idx] += 1
    return [(int(round(edges[i])), int(round(edges[i + 1])), counts[i]) for i in range(n_buckets)]


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


def write_journal(
    journal_path: Path,
    *,
    tokenizer: TokenizerInfo,
    input_total: int,
    held_out_pairs: list[dict],
    held_out_tokens: list[int],
    main_splits: dict[str, tuple[list[dict], list[int]]],
    drops_main: Counter,
    drops_heldout: Counter,
    moved_chunks: int,
    in_path: Path,
    out_dir: Path,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> None:
    """Write docs/journal/06-chatml-formatting.md."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_table_row(*cells: str) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    lines: list[str] = []
    lines.append("# ChatML formatting + train/val/test/held-out splits")
    lines.append("")
    lines.append("_Generated by `scripts/build_splits.py`. Re-running overwrites this file._")
    lines.append("")
    lines.append(f"- Input:    `{in_path}`")
    lines.append(f"- Outputs:  `{out_dir}/{{train,val,test,held_out_source}}.jsonl`")
    lines.append(f"- Run at:   `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("**System prompt (exact text):**")
    lines.append("")
    lines.append("> " + SYSTEM_PROMPT)
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(fmt_table_row("Held-out source prefixes", ", ".join(sorted(HELD_OUT_SOURCES))))
    lines.append(fmt_table_row("Random seed", seed))
    lines.append(fmt_table_row("Train fraction", train_frac))
    lines.append(fmt_table_row("Val fraction", val_frac))
    lines.append(fmt_table_row("Test fraction", test_frac))
    lines.append(fmt_table_row("Tokenizer", f"`{tokenizer.name}` (backend: `{tokenizer.backend}`)"))
    lines.append(fmt_table_row("MIN_TOKENS (exclusive)", MIN_TOKENS))
    lines.append(fmt_table_row("MAX_TOKENS (exclusive)", MAX_TOKENS))
    lines.append(fmt_table_row("Split strategy", "chunk-level (chunk_ids shuffled, then partitioned by fractions)"))
    lines.append("")

    # Input
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Total pairs loaded: **{input_total:,}**")
    lines.append("")

    # Held-out
    lines.append("## Held-out source split")
    lines.append("")
    held_by_prefix: dict[str, int] = defaultdict(int)
    for ex in held_out_pairs:
        held_by_prefix[held_out_prefix(ex["source_filename"]) or "?"] += 1
    lines.append("| Held-out prefix | Pair count |")
    lines.append("|---|---:|")
    for prefix in sorted(HELD_OUT_SOURCES):
        lines.append(fmt_table_row(f"`{prefix}`", held_by_prefix.get(prefix, 0)))
    lines.append(fmt_table_row("**Total held_out_source.jsonl**", f"**{len(held_out_pairs)}**"))
    lines.append("")

    # Main split table
    lines.append("## Main split")
    lines.append("")
    lines.append(
        "| split | pair_count | chunk_count | unique_sources | "
        "tokens_mean | tokens_p50 | tokens_p90 | tokens_p99 | tokens_max |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split_name in ("train", "val", "test"):
        exs, tcs = main_splits[split_name]
        chunks = {ex["chunk_id"] for ex in exs}
        srcs = {ex["source_filename"] for ex in exs}
        stats = token_stats(tcs)
        lines.append(
            fmt_table_row(
                split_name,
                f"{len(exs):,}",
                f"{len(chunks):,}",
                f"{len(srcs):,}",
                stats["mean"],
                stats["median"],
                stats["p90"],
                stats["p99"],
                stats["max"],
            )
        )
    lines.append("")

    # Validation
    lines.append("## Validation")
    lines.append("")
    total_dropped = sum(drops_main.values()) + sum(drops_heldout.values())
    lines.append(f"- Examples dropped during validation: **{total_dropped}**")
    if total_dropped:
        lines.append("")
        lines.append("**Drop reasons (main bucket):**")
        lines.append("")
        if drops_main:
            for reason, count in drops_main.most_common():
                lines.append(f"- `{reason}`: {count}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("**Drop reasons (held-out bucket):**")
        lines.append("")
        if drops_heldout:
            for reason, count in drops_heldout.most_common():
                lines.append(f"- `{reason}`: {count}")
        else:
            lines.append("- (none)")
    lines.append("")
    if moved_chunks:
        lines.append(f"- Chunks force-moved to fix leakage: **{moved_chunks}**")
    else:
        lines.append(
            "- Chunk-id leakage: **impossible by construction** "
            "(chunk-level splitter assigns all pairs of a chunk to one split)."
        )
    lines.append(f"- Chunk-id leakage check: **no leakage**")
    lines.append("")

    # Token length analysis
    lines.append("## Token length analysis")
    lines.append("")
    for split_name in ("train", "val", "test", "held_out_source"):
        if split_name == "held_out_source":
            tcs = held_out_tokens
        else:
            tcs = main_splits[split_name][1]
        if not tcs:
            continue
        s = token_stats(tcs)
        lines.append(f"### `{split_name}` ({len(tcs):,} examples)")
        lines.append("")
        lines.append("| stat | value |")
        lines.append("|---|---:|")
        for key in ("min", "p10", "p25", "median", "p75", "p90", "p99", "max"):
            lines.append(fmt_table_row(key, s[key]))
        lines.append("")
        lines.append("Histogram (10 buckets over [min, max]):")
        lines.append("")
        lines.append("| Range | Count | Bar |")
        lines.append("|---|---:|---|")
        hist = histogram(tcs, n_buckets=10)
        max_count = max(c for _, _, c in hist) if hist else 1
        for lo, hi, c in hist:
            bar = "█" * (int(round((c / max_count) * 30)) if max_count else 0)
            lines.append(fmt_table_row(f"{lo} – {hi}", c, bar))
        lines.append("")

    # Samples
    lines.append("## Sample examples")
    lines.append("")
    sample_rng = random.Random(seed + 1)
    sample_sources = {
        "train": main_splits["train"][0],
        "val": main_splits["val"][0],
        "test": main_splits["test"][0],
        "held_out_source": held_out_pairs,
    }
    for split_name, pool in sample_sources.items():
        if not pool:
            lines.append(f"### `{split_name}` — _empty_")
            lines.append("")
            continue
        sample = sample_rng.choice(pool)
        lines.append(f"### `{split_name}` — `{sample['pair_id']}`")
        lines.append("")
        lines.append(f"- **source_filename:** `{sample['source_filename']}`")
        lines.append(f"- **chunk_id:** `{sample['chunk_id']}`")
        lines.append(f"- **question_type:** `{sample['question_type']}`")
        lines.append("")
        user_content = sample["messages"][1]["content"]
        asst_content = sample["messages"][2]["content"]
        lines.append("**Raw question (from source pair):**")
        lines.append("")
        lines.append("> " + user_content.replace("\n", "\n> "))
        lines.append("")
        lines.append("**Raw answer (from source pair):**")
        lines.append("")
        lines.append("> " + asst_content.replace("\n", "\n> "))
        lines.append("")
        lines.append("**Rendered chat-template output (what the model sees):**")
        lines.append("")
        lines.append("```text")
        lines.append(tokenizer.render(sample["messages"]))
        lines.append("```")
        lines.append("")

    # Top sources in training
    lines.append("## Per-source pair distribution — top 10 in training")
    lines.append("")
    train_exs = main_splits["train"][0]
    src_counts = Counter(ex["source_filename"] for ex in train_exs)
    lines.append("| source_filename | training pairs |")
    lines.append("|---|---:|")
    for src, count in src_counts.most_common(10):
        lines.append(fmt_table_row(f"`{src}`", count))
    lines.append("")

    # Anything weird
    lines.append("## Anything weird")
    lines.append("")
    surprises: list[str] = []
    main_kept = sum(len(main_splits[s][0]) for s in ("train", "val", "test"))
    if drops_main:
        worst = drops_main.most_common(1)[0]
        surprises.append(
            f"- Top main-bucket drop reason: `{worst[0]}` ({worst[1]} examples)."
        )
    if drops_heldout:
        worst = drops_heldout.most_common(1)[0]
        surprises.append(
            f"- Top held-out drop reason: `{worst[0]}` ({worst[1]} examples)."
        )
    if total_dropped == 0:
        surprises.append("- Zero examples dropped — every pair tokenized within budget.")
    surprises.append(
        "- **Deviated from spec algorithm.** The spec described a pair-level "
        "shuffle followed by post-hoc \"move leaked chunk to majority split\". "
        "Empirically that gutted val/test (~30/150 pairs vs target ~935/1,873) "
        "because each chunk has ~5 pairs and the 5% val ratio rarely makes val "
        "the majority for any chunk. We instead shuffle and partition at the "
        "**chunk_id** level, which leaves leakage impossible by construction "
        "and hits the requested fractions cleanly."
    )
    held_prefix_zero = [p for p in HELD_OUT_SOURCES if held_by_prefix.get(p, 0) == 0]
    if held_prefix_zero:
        surprises.append(
            f"- Held-out prefix(es) with zero pairs after validation: {held_prefix_zero}."
        )
    cuas_folder_check = "crs_2026_law_enforcement_cuas_IN12661"
    if any(
        ex["source_filename"].startswith(cuas_folder_check)
        for ex in held_out_pairs
    ):
        surprises.append(
            "- CRS held-out folder is `crs_2026_law_enforcement_cuas_IN12661` "
            "in the corpus (the doc-ID suffix `_IN12661` is appended). "
            "Matched via prefix rule `folder == prefix OR folder.startswith(prefix + '_')`."
        )
    for s in surprises:
        lines.append(s)
    lines.append("")

    journal_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", default="data/interim/qa_pairs.jsonl", type=Path)
    parser.add_argument("--out-dir", default="data/processed/qa", type=Path)
    parser.add_argument("--seed", default=RANDOM_SEED, type=int)
    parser.add_argument("--train-frac", default=TRAIN_FRAC, type=float)
    parser.add_argument("--val-frac", default=VAL_FRAC, type=float)
    parser.add_argument("--test-frac", default=TEST_FRAC, type=float)
    parser.add_argument(
        "--journal-path",
        default="docs/journal/06-chatml-formatting.md",
        type=Path,
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    in_path: Path = args.in_path
    out_dir: Path = args.out_dir
    journal_path: Path = args.journal_path

    if not in_path.exists():
        console.print(f"[red]STOP[/red]: input not found: {in_path}")
        return 1

    console.rule("Loading tokenizer")
    tokenizer = make_tokenizer()
    console.print(f"Tokenizer: [bold]{tokenizer.name}[/bold] (backend: {tokenizer.backend})")

    console.rule("Loading Q&A pairs")
    pairs = load_pairs(in_path)
    console.print(f"Loaded [bold]{len(pairs):,}[/bold] pairs from {in_path}")

    # Partition
    held_out_raw = [p for p in pairs if held_out_prefix(p["source_filename"]) is not None]
    main_raw = [p for p in pairs if held_out_prefix(p["source_filename"]) is None]
    console.print(
        f"Partitioned: main = [bold]{len(main_raw):,}[/bold], "
        f"held-out = [bold]{len(held_out_raw):,}[/bold]"
    )

    # Convert + validate
    console.rule("Converting + validating main bucket")
    main_exs, main_tcs, drops_main = convert_and_validate(main_raw, tokenizer, "main")
    console.rule("Converting + validating held-out bucket")
    held_exs, held_tcs, drops_heldout = convert_and_validate(held_out_raw, tokenizer, "held-out")

    total_dropped = sum(drops_main.values()) + sum(drops_heldout.values())
    drop_rate = total_dropped / max(1, len(pairs))
    console.print(
        f"Dropped {total_dropped} of {len(pairs)} ({drop_rate * 100:.2f}%) in validation"
    )
    if drop_rate > 0.02:
        console.print("[red]STOP[/red]: drop rate >2% — investigate before proceeding.")
        for reason, count in (drops_main + drops_heldout).most_common():
            console.print(f"  - {reason}: {count}")
        return 1

    # Shuffle and split main
    console.rule("Shuffling and splitting main bucket")
    train, val, test = shuffle_and_split(
        main_exs,
        main_tcs,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    splits = {"train": train, "val": val, "test": test}
    for name, (exs, _) in splits.items():
        console.print(f"  {name}: {len(exs):,}")

    # Enforce no leakage
    console.rule("Enforcing no chunk_id leakage")
    splits, _moved, moved_chunks = enforce_no_chunk_leakage(splits)
    console.print(f"Moved {moved_chunks} chunks across splits to fix leakage")
    assert_no_leakage(splits)
    console.print("[green]Chunk-id leakage check: OK[/green]")
    for name, (exs, _) in splits.items():
        console.print(f"  {name}: {len(exs):,}")

    if args.dry_run:
        console.print("[yellow]--dry-run set; skipping writes.[/yellow]")
        return 0

    # Write JSONL files
    console.rule("Writing JSONL splits")
    for name, (exs, _) in splits.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, exs)
        console.print(f"Wrote {path} ({len(exs):,} examples)")
    held_out_path = out_dir / "held_out_source.jsonl"
    write_jsonl(held_out_path, held_exs)
    console.print(f"Wrote {held_out_path} ({len(held_exs):,} examples)")

    # Journal
    console.rule("Writing journal")
    write_journal(
        journal_path,
        tokenizer=tokenizer,
        input_total=len(pairs),
        held_out_pairs=held_exs,
        held_out_tokens=held_tcs,
        main_splits=splits,
        drops_main=drops_main,
        drops_heldout=drops_heldout,
        moved_chunks=moved_chunks,
        in_path=in_path,
        out_dir=out_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    console.print(f"Wrote journal: {journal_path}")

    # Final summary table
    console.rule("Summary")
    table = Table(show_header=True, header_style="bold")
    table.add_column("split")
    table.add_column("pairs", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("sources", justify="right")
    table.add_column("tokens (median/max)", justify="right")
    for split_name in ("train", "val", "test", "held_out_source"):
        if split_name == "held_out_source":
            exs, tcs = held_exs, held_tcs
        else:
            exs, tcs = splits[split_name]
        chunks = {e["chunk_id"] for e in exs}
        srcs = {e["source_filename"] for e in exs}
        ts = token_stats(tcs)
        table.add_row(
            split_name,
            f"{len(exs):,}",
            f"{len(chunks):,}",
            f"{len(srcs):,}",
            f"{ts['median']}/{ts['max']}",
        )
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
