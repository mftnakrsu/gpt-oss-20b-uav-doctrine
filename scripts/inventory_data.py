"""Deep data inventory for data/raw/.

Walks data/raw/, extracts text from every supported file type, tokenizes
with the GPT-OSS-20B tokenizer (or tiktoken o200k_base fallback), and
writes a full Markdown report to docs/journal/01-data-inventory.md.

Safely re-runnable: never modifies anything inside data/raw/.

Usage:
    uv run python scripts/inventory_data.py
"""

from __future__ import annotations

import hashlib
import io
import os
import statistics
import sys
import textwrap
import warnings
from collections import defaultdict
from contextlib import redirect_stderr
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Silence noisy third-party warnings during extraction.
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
JOURNAL_DIR = PROJECT_ROOT / "docs" / "journal"
JOURNAL_PATH = JOURNAL_DIR / "01-data-inventory.md"

# Placeholder Tinker train price. Update once verified on the dashboard.
TINKER_PRICE_USD_PER_1M_TOKENS = 0.40

TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".jsonl"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
HTML_EXTS = {".html", ".htm"}

console = Console()


# ───────────────────────── helpers ─────────────────────────


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:,.1f} {unit}" if unit != "B" else f"{int(f):,} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * p
    f_idx = int(k)
    c_idx = min(f_idx + 1, len(sorted_values) - 1)
    if f_idx == c_idx:
        return float(sorted_values[f_idx])
    return float(sorted_values[f_idx]) + (k - f_idx) * (
        sorted_values[c_idx] - sorted_values[f_idx]
    )


def ascii_histogram(values: list[int], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"[{lo:,} ── {len(values)} file(s)]"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        b_lo = int(lo + i * step)
        b_hi = int(lo + (i + 1) * step)
        bar = "█" * max(1, int(c / max_count * width)) if c else ""
        lines.append(f"  {b_lo:>10,} – {b_hi:>10,}  | {bar:<{width}} {c}")
    return "\n".join(lines)


# ───────────────────────── extraction ─────────────────────────


def _extract_pdf_pypdf(path: Path) -> str:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts).strip()


def _extract_pdf_pdfplumber(path: Path) -> str:
    import pdfplumber

    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
    return "\n".join(parts).strip()


def extract_text(path: Path) -> tuple[str, Optional[str]]:
    """Return ``(text, error)``. ``text`` is "" on total failure."""
    ext = path.suffix.lower()
    try:
        if ext in TEXT_EXTS:
            return path.read_text(encoding="utf-8", errors="replace"), None

        if ext in PDF_EXTS:
            err: Optional[str] = None
            try:
                with redirect_stderr(io.StringIO()):
                    text = _extract_pdf_pypdf(path)
                if text:
                    return text, None
                err = "pypdf returned empty"
            except Exception as e:
                err = f"pypdf {type(e).__name__}: {e}"
            try:
                with redirect_stderr(io.StringIO()):
                    text = _extract_pdf_pdfplumber(path)
                if text:
                    return text, None
                return "", f"both extractors empty; pypdf: {err}"
            except Exception as e:
                return "", f"pypdf {err}; pdfplumber {type(e).__name__}: {e}"

        if ext in DOCX_EXTS:
            import docx

            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs), None

        if ext in HTML_EXTS:
            from bs4 import BeautifulSoup

            raw = path.read_text(encoding="utf-8", errors="replace")
            return BeautifulSoup(raw, "lxml").get_text(separator=" ", strip=True), None

        if ext == "":
            # No extension — try to read as text if it looks textual.
            raw = path.read_bytes()
            if len(raw) > 50_000_000:
                return "", "no extension; file too large for text guess"
            if b"\x00" in raw[:4096]:
                return "", "no extension; binary content detected"
            return raw.decode("utf-8", errors="replace"), None

        return "", f"unsupported extension {ext}"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


# ───────────────────────── tokenizer ─────────────────────────


class _TiktokenWrapper:
    def __init__(self, enc):
        self.enc = enc

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, disallowed_special=())


def load_tokenizer() -> tuple[object, str, bool]:
    """Return ``(tokenizer, label, is_fallback)``."""
    hf_token = os.environ.get("HF_TOKEN")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", token=hf_token)
        return tok, "openai/gpt-oss-20b (HuggingFace AutoTokenizer)", False
    except Exception as e:
        console.print(
            f"[yellow]HF tokenizer load failed ({type(e).__name__}); "
            "falling back to tiktoken o200k_base.[/yellow]"
        )
        console.print(f"  [dim]reason: {str(e)[:160]}[/dim]")
        import tiktoken

        return (
            _TiktokenWrapper(tiktoken.get_encoding("o200k_base")),
            "tiktoken o200k_base (fallback — HF gate / network / token miss)",
            True,
        )


def count_tokens(tokenizer, text: str, is_fallback: bool) -> int:
    if not text:
        return 0
    if is_fallback:
        return len(tokenizer.encode(text))
    return len(tokenizer.encode(text, add_special_tokens=False))


# ───────────────────────── report rendering ─────────────────────────


def render_markdown(
    *,
    files: list[Path],
    total_size: int,
    ext_stats: dict[str, dict[str, int]],
    sample_names: list[str],
    samples: dict[str, dict],
    tokenizer_label: str,
    is_fallback: bool,
    per_file: list[dict],
    token_counts: list[int],
    stats: dict[str, float],
    histogram: str,
    seq_buckets: dict[int, int],
    epoch_costs: dict[int, float],
    tiny_files: list[dict],
    duplicate_groups: dict[str, list[str]],
    extraction_failures: list[tuple[str, str]],
) -> str:
    total_tokens = sum(token_counts)
    lines: list[str] = []
    a = lines.append

    a("# Data inventory — `data/raw/`")
    a("")
    a("_Generated by `scripts/inventory_data.py`. Re-running overwrites this file._")
    a("")
    a(f"- Project root: `{PROJECT_ROOT}`")
    a(f"- Corpus root:  `{RAW_DIR.relative_to(PROJECT_ROOT)}`")
    a(f"- Total files:  **{len(files):,}**")
    a(f"- Total size:   **{human_size(total_size)}**")
    a(f"- Total tokens: **{total_tokens:,}**  ({tokenizer_label})")
    a("")

    # 1. File census
    a("## 1. File census")
    a("")
    a(f"Walked `data/raw/` recursively. **{len(files):,} files**, total **{human_size(total_size)}**.")
    a("")
    a("| Extension | Files | Total size | Avg size |")
    a("|---|---:|---:|---:|")
    for ext, st in sorted(ext_stats.items(), key=lambda kv: -kv[1]["size"]):
        avg = st["size"] / st["count"] if st["count"] else 0
        a(f"| `{ext}` | {st['count']:,} | {human_size(st['size'])} | {human_size(int(avg))} |")
    a("")
    a(f"### Sample filenames (up to 30, relative to `data/raw/`)")
    a("")
    for name in sample_names:
        a(f"- `{name}`")
    a("")

    # 2. Content extraction
    a("## 2. Content extraction — per file type")
    a("")
    a("For each distinct extension, the first 800 characters extracted from one representative file:")
    a("")
    for ext in sorted(samples.keys()):
        s = samples[ext]
        a(f"### `{ext}` — `{s['path']}`")
        if s["error"]:
            a("")
            a(f"_extraction error: {s['error']}_")
            a("")
        preview = s["preview"].replace("```", "ʼʼʼ").strip()
        if preview:
            a("")
            a("```text")
            a(preview if len(preview) <= 800 else preview[:800])
            a("```")
        else:
            a("")
            a("_(no text extracted)_")
        a("")

    # 3. Token counting
    a("## 3. Token counting")
    a("")
    a(f"- Tokenizer used: **{tokenizer_label}**")
    a(f"- Fallback engaged: **{'yes' if is_fallback else 'no'}**")
    a(f"- Total corpus tokens: **{total_tokens:,}**")
    a("")
    a("### Per-file token statistics")
    a("")
    a("| Statistic | Tokens |")
    a("|---|---:|")
    a(f"| min    | {int(stats['min']):,} |")
    a(f"| p25    | {int(stats['p25']):,} |")
    a(f"| median | {int(stats['median']):,} |")
    a(f"| mean   | {int(stats['mean']):,} |")
    a(f"| p75    | {int(stats['p75']):,} |")
    a(f"| p90    | {int(stats['p90']):,} |")
    a(f"| p99    | {int(stats['p99']):,} |")
    a(f"| max    | {int(stats['max']):,} |")
    a("")
    a("### Histogram of tokens-per-file (10 bins)")
    a("")
    a("```text")
    a(histogram)
    a("```")
    a("")
    a("### Top 10 files by token count")
    a("")
    a("| Tokens | File |")
    a("|---:|---|")
    sorted_by_tokens = sorted(per_file, key=lambda d: -d["tokens"])
    for d in sorted_by_tokens[:10]:
        a(f"| {d['tokens']:,} | `{d['path']}` |")
    a("")
    a("### Bottom 5 files by token count")
    a("")
    a("| Tokens | File |")
    a("|---:|---|")
    for d in sorted_by_tokens[-5:][::-1]:
        a(f"| {d['tokens']:,} | `{d['path']}` |")
    a("")

    # 4. Sequence length analysis
    a("## 4. Sequence length analysis")
    a("")
    a("Files (after extraction) exceeding each token threshold:")
    a("")
    a("| Threshold | Files above |")
    a("|---:|---:|")
    for t, c in seq_buckets.items():
        a(f"| {t:>6,} | {c:,} |")
    a("")
    a("> This informs whether we chunk source text or rely on sequence extension at train time.")
    a("")

    # 5. Cost estimation
    a("## 5. Training cost estimation")
    a("")
    a(
        f"Assumed Tinker training price for GPT-OSS-20B: "
        f"**~${TINKER_PRICE_USD_PER_1M_TOKENS:.2f} per 1M training tokens** "
        "(placeholder — verify on the Tinker dashboard before committing real spend)."
    )
    a("")
    a("| Epochs | Tokens consumed | Estimated cost (USD) |")
    a("|---:|---:|---:|")
    for n, cost in epoch_costs.items():
        a(f"| {n} | {total_tokens * n:,} | ${cost:,.2f} |")
    a("")
    a(
        "> These are **raw-corpus** costs over the extracted text. Actual training cost depends "
        "on how raw text becomes `(prompt, completion)` pairs and how the loss is masked. "
        "We'll refine in a later step."
    )
    a("")

    # 6. Quality flags
    a("## 6. Quality flags")
    a("")
    a(f"- Files with **< 50 tokens** after extraction (likely empty or scrap): **{len(tiny_files)}**")
    if tiny_files:
        a("")
        a("| Tokens | File |")
        a("|---:|---|")
        for d in tiny_files:
            a(f"| {d['tokens']:,} | `{d['path']}` |")
    a("")
    a(f"- Duplicate file groups by SHA-256 of raw bytes: **{len(duplicate_groups)}**")
    if duplicate_groups:
        a("")
        for sha, paths in duplicate_groups.items():
            a(f"  - `{sha[:12]}…` ({len(paths)} files):")
            for p in paths:
                a(f"    - `{p}`")
    a("")
    a(f"- Files where extraction failed entirely: **{len(extraction_failures)}**")
    if extraction_failures:
        a("")
        a("| File | Reason |")
        a("|---|---|")
        for path, err in extraction_failures:
            a(f"| `{path}` | {err} |")
    a("")
    return "\n".join(lines) + "\n"


# ───────────────────────── main ─────────────────────────


def main() -> int:
    if not RAW_DIR.exists():
        console.print(f"[red]data/raw/ not found at {RAW_DIR}[/red]")
        return 1

    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
    if not files:
        console.print("[red]data/raw/ is empty.[/red]")
        return 1

    console.print(
        Panel.fit(
            f"[bold]Inventorying {len(files):,} files under "
            f"{RAW_DIR.relative_to(PROJECT_ROOT)}/[/bold]"
        )
    )

    # 1. File census
    total_size = sum(f.stat().st_size for f in files)
    ext_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "size": 0})
    for f in files:
        ext = f.suffix.lower() or "(no ext)"
        ext_stats[ext]["count"] += 1
        ext_stats[ext]["size"] += f.stat().st_size

    rel_names = [str(f.relative_to(RAW_DIR)) for f in files]
    sample_names = rel_names[:30]

    # Pretty print census
    t = Table(title="File census by extension")
    t.add_column("Ext")
    t.add_column("Count", justify="right")
    t.add_column("Total size", justify="right")
    t.add_column("Avg size", justify="right")
    for ext, st in sorted(ext_stats.items(), key=lambda kv: -kv[1]["size"]):
        avg = st["size"] / st["count"] if st["count"] else 0
        t.add_row(ext, f"{st['count']:,}", human_size(st["size"]), human_size(int(avg)))
    console.print(t)

    # 2. Pick a representative file per extension for the preview section
    ext_to_first_file: dict[str, Path] = {}
    for f in files:
        ext = f.suffix.lower() or "(no ext)"
        if ext not in ext_to_first_file:
            ext_to_first_file[ext] = f
    samples: dict[str, dict] = {}
    for ext, f in ext_to_first_file.items():
        text, err = extract_text(f)
        samples[ext] = {
            "path": str(f.relative_to(RAW_DIR)),
            "preview": text[:800],
            "error": err,
        }

    # 3. Tokenize every file
    tokenizer, tokenizer_label, is_fallback = load_tokenizer()
    console.print(f"[bold green]Tokenizer:[/bold green] {tokenizer_label}")

    per_file: list[dict] = []
    extraction_failures: list[tuple[str, str]] = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Extracting + tokenizing", total=len(files))
        for f in files:
            rel = str(f.relative_to(RAW_DIR))
            text, err = extract_text(f)
            if (not text) and err:
                extraction_failures.append((rel, err))
                per_file.append(
                    {
                        "path": rel,
                        "tokens": 0,
                        "size": f.stat().st_size,
                        "sha": sha256_of(f),
                        "error": err,
                    }
                )
            else:
                tokens = count_tokens(tokenizer, text, is_fallback)
                per_file.append(
                    {
                        "path": rel,
                        "tokens": tokens,
                        "size": f.stat().st_size,
                        "sha": sha256_of(f),
                        "error": None,
                    }
                )
            progress.advance(task)

    token_counts = sorted(d["tokens"] for d in per_file)
    total_tokens = sum(token_counts)
    if token_counts:
        stats = {
            "min": min(token_counts),
            "p25": percentile(token_counts, 0.25),
            "median": percentile(token_counts, 0.50),
            "mean": statistics.mean(token_counts),
            "p75": percentile(token_counts, 0.75),
            "p90": percentile(token_counts, 0.90),
            "p99": percentile(token_counts, 0.99),
            "max": max(token_counts),
        }
    else:
        stats = {k: 0 for k in ("min", "p25", "median", "mean", "p75", "p90", "p99", "max")}
    histogram = ascii_histogram(token_counts, bins=10)

    # 4. Sequence length thresholds
    thresholds = [2_000, 4_000, 8_000, 16_000, 32_000]
    seq_buckets = {t: sum(1 for c in token_counts if c > t) for t in thresholds}

    # 5. Cost estimation
    epoch_costs = {
        n: (total_tokens * n / 1_000_000.0) * TINKER_PRICE_USD_PER_1M_TOKENS
        for n in (1, 2, 3)
    }

    # 6. Quality flags
    tiny_files = sorted(
        [d for d in per_file if 0 < d["tokens"] < 50],
        key=lambda d: d["tokens"],
    )
    sha_groups: dict[str, list[str]] = defaultdict(list)
    for d in per_file:
        if d["sha"]:
            sha_groups[d["sha"]].append(d["path"])
    duplicate_groups = {sha: paths for sha, paths in sha_groups.items() if len(paths) > 1}

    # ── Pretty print key tables to terminal ──
    stats_t = Table(title="Per-file token stats")
    stats_t.add_column("statistic")
    stats_t.add_column("tokens", justify="right")
    for k in ("min", "p25", "median", "mean", "p75", "p90", "p99", "max"):
        stats_t.add_row(k, f"{int(stats[k]):,}")
    console.print(stats_t)

    seq_t = Table(title="Files above each token threshold")
    seq_t.add_column("threshold", justify="right")
    seq_t.add_column("files above", justify="right")
    for thr, c in seq_buckets.items():
        seq_t.add_row(f"{thr:,}", f"{c:,}")
    console.print(seq_t)

    cost_t = Table(title=f"Training cost @ ${TINKER_PRICE_USD_PER_1M_TOKENS:.2f}/1M tokens")
    cost_t.add_column("epochs", justify="right")
    cost_t.add_column("tokens", justify="right")
    cost_t.add_column("USD", justify="right")
    for n, cost in epoch_costs.items():
        cost_t.add_row(str(n), f"{total_tokens * n:,}", f"${cost:,.2f}")
    console.print(cost_t)

    console.print("[bold]Histogram of tokens-per-file[/bold]")
    console.print(histogram)

    # ── Write the Markdown journal ──
    md = render_markdown(
        files=files,
        total_size=total_size,
        ext_stats=dict(ext_stats),
        sample_names=sample_names,
        samples=samples,
        tokenizer_label=tokenizer_label,
        is_fallback=is_fallback,
        per_file=per_file,
        token_counts=token_counts,
        stats=stats,
        histogram=histogram,
        seq_buckets=seq_buckets,
        epoch_costs=epoch_costs,
        tiny_files=tiny_files,
        duplicate_groups=duplicate_groups,
        extraction_failures=extraction_failures,
    )
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.write_text(md, encoding="utf-8")

    # ── Build final summary block ──
    top_issues: list[str] = []
    if extraction_failures:
        top_issues.append(f"{len(extraction_failures)} files failed extraction")
    if tiny_files:
        top_issues.append(f"{len(tiny_files)} files have <50 tokens after extraction")
    if duplicate_groups:
        top_issues.append(f"{len(duplicate_groups)} duplicate-file group(s) by SHA-256")
    over_32k = seq_buckets.get(32_000, 0)
    if over_32k:
        top_issues.append(f"{over_32k} files exceed 32k tokens (will need chunking)")
    if not top_issues:
        top_issues.append("no critical issues flagged")

    summary_lines = [
        "",
        "[bold green]Inventory created[/bold green]",
        f"  • Journal:           {JOURNAL_PATH.relative_to(PROJECT_ROOT)}",
        f"  • Tokenizer used:    {tokenizer_label}",
        f"  • Files:             {len(files):,}  ({human_size(total_size)})",
        f"  • Total tokens:      {total_tokens:,}",
        f"  • 1-epoch est. cost: ${epoch_costs[1]:,.2f}",
        f"  • Top issues:",
    ]
    for issue in top_issues[:3]:
        summary_lines.append(f"      - {issue}")
    console.print(Panel("\n".join(summary_lines), title="Summary", border_style="green"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
