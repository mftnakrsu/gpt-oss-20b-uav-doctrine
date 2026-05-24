"""Full Q&A generation over every quality-passing chunk.

Generates ``--pairs-per-chunk`` Q&A pairs per chunk through Azure GPT-5.4,
optionally annotates each pair with an embedding-cosine debug score (not
used for filtering — kept only as a relative diagnostic signal because of
the question-vs-chunk length asymmetry; see docs/journal/05a-qa-pilot.md).

Operational guardrails:
- **Resumable** via an atomic JSON checkpoint at
  ``--checkpoint-path``. ``--resume`` (default True) reads the
  checkpoint plus paranoia-scans the existing output JSONL to derive
  the "processed" set; never generates a chunk that already has pairs
  on disk.
- **Append + fsync per chunk** to ``--out-path``. Crash-safe at chunk
  granularity.
- **Hard cost cap** ``--max-cost-usd`` aborts the run when the running
  estimate (at placeholder rates) exceeds the cap. Checkpoint is still
  saved on abort.
- **Pause at ``--pause-at-chunks``** with a 10-second countdown. Ctrl+C
  during the countdown aborts cleanly.
- **Dynamic concurrency reduction**: 6 consecutive RateLimitError
  events on the LLM trigger a permanent decrease in effective
  concurrency (acquire-and-hold trick on a semaphore).
- **Ctrl+C** anywhere writes a final checkpoint, flushes the output
  file, and exits 130.

The API key is read from ``AZURE_API_KEY`` and never echoed anywhere.

Run::

    uv run python scripts/full_qa.py
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import signal
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

import openai  # noqa: E402
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

JOURNAL_PATH = PROJECT_ROOT / "docs" / "journal" / "05b-qa-full-run.md"
DEFAULT_IN_PATH = PROJECT_ROOT / "data" / "interim" / "chunks_filtered.jsonl"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "interim" / "qa_pairs.jsonl"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "qa_generation_checkpoint.json"
)

# Placeholder pricing — calibrate against real Azure invoice later.
LLM_INPUT_PRICE_PER_1M_USD = 5.0
LLM_OUTPUT_PRICE_PER_1M_USD = 15.0
EMBEDDING_PRICE_PER_1M_USD = 0.02

CONSECUTIVE_429_THRESHOLD = 6  # > this triggers concurrency reduction
EMBEDDING_CHUNK_SLICE_CHARS = 2000


# ────────── dynamic concurrency ──────────


class DynamicSemaphore:
    """asyncio.Semaphore with a one-way ratchet to reduce effective capacity.

    Implemented by 'acquire and hold' — to reduce capacity by 1, we
    permanently grab one slot. Existing acquires drain normally; new
    acquires see one fewer slot available.
    """

    def __init__(self, initial: int) -> None:
        self._initial = initial
        self._sem = asyncio.Semaphore(initial)
        self._reductions = 0

    @property
    def effective_limit(self) -> int:
        return self._initial - self._reductions

    async def __aenter__(self) -> "DynamicSemaphore":
        await self._sem.acquire()
        return self

    async def __aexit__(self, *a) -> None:
        self._sem.release()

    async def reduce_by_one(self) -> None:
        """Permanently retire one concurrency slot."""
        if self.effective_limit <= 1:
            return
        # Acquire one slot we will never release.
        await self._sem.acquire()
        self._reductions += 1


# ────────── checkpoint I/O ──────────


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON to ``path`` atomically (write tmp → fsync → rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def scan_chunk_ids_in_jsonl(path: Path) -> set[str]:
    """Return the set of chunk_id values present in an existing JSONL.

    Tolerant of partial / malformed lines (e.g. truncated last line from
    a crash) — they're skipped silently.
    """
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("chunk_id")
            if cid:
                seen.add(cid)
    return seen


# ────────── runner ──────────


class FullQARunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # State
        self.processed_chunk_ids: set[str] = set()
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "embedding_tokens": 0,
        }
        self.running_cost_usd = 0.0
        self.pairs_count = 0
        self.json_parse_failures: list[tuple[str, str]] = []
        self.embedding_failures: list[tuple[str, str]] = []
        self.per_source_pairs: dict[str, list[int]] = collections.defaultdict(list)
        # Termination
        self.stop_requested = False
        self.abort_reason: str | None = None
        # Concurrency state
        self._consecutive_429s = 0
        self._concurrency_reductions = 0
        # Resources
        self.llm: AzureLLMClient | None = None
        self.embedder: AzureEmbeddingClient | None = None
        self.dyn_sem: DynamicSemaphore | None = None
        self.out_f = None
        # Locks
        self.state_lock = asyncio.Lock()
        self.out_lock = asyncio.Lock()
        # Timing
        self.started_at_iso = datetime.now(timezone.utc).isoformat()
        self.started_at_perf = time.perf_counter()
        # Chunks-since-last-checkpoint counter
        self._since_checkpoint = 0

    # ---- helpers ----

    def _compute_cost(self) -> float:
        return (
            self.token_usage["prompt_tokens"] / 1_000_000 * LLM_INPUT_PRICE_PER_1M_USD
            + self.token_usage["completion_tokens"]
            / 1_000_000
            * LLM_OUTPUT_PRICE_PER_1M_USD
            + self.token_usage["embedding_tokens"]
            / 1_000_000
            * EMBEDDING_PRICE_PER_1M_USD
        )

    def _save_checkpoint(self) -> None:
        data = {
            "processed_chunk_ids": sorted(self.processed_chunk_ids),
            "running_cost_usd": round(self.running_cost_usd, 6),
            "token_usage": self.token_usage,
            "concurrency_reductions": self._concurrency_reductions,
            "started_at": self.started_at_iso,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.args.checkpoint_path, data)

    def _load_checkpoint(self) -> None:
        if not self.args.resume:
            return
        if not self.args.checkpoint_path.exists():
            return
        try:
            data = json.loads(self.args.checkpoint_path.read_text())
        except json.JSONDecodeError:
            console.print(
                f"[yellow]checkpoint at {self.args.checkpoint_path} is corrupt; "
                "ignoring[/yellow]"
            )
            return
        self.processed_chunk_ids = set(data.get("processed_chunk_ids", []))
        self.running_cost_usd = float(data.get("running_cost_usd", 0.0))
        self.token_usage = data.get("token_usage", self.token_usage)
        self._concurrency_reductions = int(data.get("concurrency_reductions", 0))

    def _paranoia_merge_from_jsonl(self) -> None:
        """Union checkpoint's processed set with chunk_ids in the existing JSONL."""
        existing = scan_chunk_ids_in_jsonl(self.args.out_path)
        if not existing:
            return
        new_from_jsonl = existing - self.processed_chunk_ids
        missing_from_jsonl = self.processed_chunk_ids - existing
        if new_from_jsonl:
            console.print(
                f"[yellow]paranoia merge: adding {len(new_from_jsonl)} chunk_ids "
                "from JSONL not in checkpoint[/yellow]"
            )
        if missing_from_jsonl:
            console.print(
                f"[yellow]warning: {len(missing_from_jsonl)} chunk_ids in "
                "checkpoint have no pairs in JSONL (will not regenerate — "
                "trusting checkpoint)[/yellow]"
            )
        self.processed_chunk_ids |= existing

    # ---- per-chunk processing ----

    async def _embed_for_chunk(self, chunk: dict, pairs: list[dict]) -> None:
        """Compute embedding cosine debug score for each pair. Mutates pairs in place."""
        if not pairs or self.embedder is None:
            for p in pairs:
                p["embedding_cosine_debug"] = None
            return
        try:
            texts = [chunk["text"][:EMBEDDING_CHUNK_SLICE_CHARS]] + [
                p["question"] for p in pairs
            ]
            vecs = await self.embedder.embed(texts)
            chunk_vec = vecs[0]
            for p, qv in zip(pairs, vecs[1:]):
                try:
                    p["embedding_cosine_debug"] = round(
                        cosine_similarity(qv, chunk_vec), 4
                    )
                except Exception:
                    p["embedding_cosine_debug"] = None
            # Track embedding tokens (estimate: ~4 chars per token)
            est_tokens = sum(len(t) for t in texts) // 4
            async with self.state_lock:
                self.token_usage["embedding_tokens"] += est_tokens
        except Exception as e:
            for p in pairs:
                p["embedding_cosine_debug"] = None
            self.embedding_failures.append((chunk["chunk_id"], f"{type(e).__name__}: {e}"))

    async def _process_one(self, chunk: dict) -> None:
        # Early exit if a sibling task already tripped the kill switch.
        if self.stop_requested:
            return
        assert self.dyn_sem is not None
        async with self.dyn_sem:
            if self.stop_requested:
                return

            # ---- LLM call with rate-limit aware accounting ----
            try:
                pairs, usage, err = await generate_pairs_for_chunk(
                    self.llm,
                    chunk,
                    k=self.args.pairs_per_chunk,
                )
                self._consecutive_429s = 0
            except openai.RateLimitError as e:
                self._consecutive_429s += 1
                if (
                    self._consecutive_429s > CONSECUTIVE_429_THRESHOLD
                    and self.dyn_sem.effective_limit > 1
                ):
                    await self.dyn_sem.reduce_by_one()
                    self._concurrency_reductions += 1
                    self._consecutive_429s = 0
                    console.print(
                        f"[yellow]rate-limit pressure persistent — concurrency "
                        f"reduced to {self.dyn_sem.effective_limit}[/yellow]"
                    )
                self.json_parse_failures.append(
                    (chunk["chunk_id"], f"RateLimitError: {e}")
                )
                return
            except Exception as e:
                # Other LLM failures: log + continue, don't mark processed (so a future
                # resume retries this chunk).
                self.json_parse_failures.append(
                    (chunk["chunk_id"], f"LLM call failed: {type(e).__name__}: {e}")
                )
                return

            if not pairs and err:
                self.json_parse_failures.append((chunk["chunk_id"], err))
                # Still mark as processed; otherwise we infinite-loop on a doomed chunk.
                async with self.state_lock:
                    self.processed_chunk_ids.add(chunk["chunk_id"])
                    self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.token_usage["completion_tokens"] += usage.get(
                        "completion_tokens", 0
                    )
                    self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
                    self.running_cost_usd = self._compute_cost()
                    self._since_checkpoint += 1
                return

            # ---- embedding (debug only) ----
            if not self.args.skip_embedding_debug:
                await self._embed_for_chunk(chunk, pairs)
            else:
                for p in pairs:
                    p["embedding_cosine_debug"] = None

            # ---- usage + cost accounting (lock) ----
            async with self.state_lock:
                self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.token_usage["completion_tokens"] += usage.get(
                    "completion_tokens", 0
                )
                self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
                self.running_cost_usd = self._compute_cost()
                if self.running_cost_usd > self.args.max_cost_usd:
                    self.stop_requested = True
                    self.abort_reason = "cost cap exceeded"

            # ---- write pairs (lock + fsync per chunk) ----
            ts = datetime.now(timezone.utc).isoformat()
            records_to_write = []
            for i, p in enumerate(pairs):
                records_to_write.append(
                    {
                        "pair_id": f"{chunk['chunk_id']}__{i:02d}",
                        "chunk_id": chunk["chunk_id"],
                        "source_filename": chunk["source_filename"],
                        "question": p["question"],
                        "answer": p["answer"],
                        "question_type": p["question_type"],
                        "embedding_cosine_debug": p.get("embedding_cosine_debug"),
                        "generation_metadata": {
                            "model": self.llm.deployment,
                            "temperature": self.llm.default_temperature,
                            "timestamp": ts,
                        },
                    }
                )
            async with self.out_lock:
                for rec in records_to_write:
                    self.out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.out_f.flush()
                os.fsync(self.out_f.fileno())

            async with self.state_lock:
                self.processed_chunk_ids.add(chunk["chunk_id"])
                self.pairs_count += len(pairs)
                self.per_source_pairs[chunk["source_filename"]].append(len(pairs))
                self._since_checkpoint += 1
                if self._since_checkpoint >= self.args.checkpoint_interval:
                    self._save_checkpoint()
                    self._since_checkpoint = 0

    # ---- batch driver ----

    async def _process_batch(
        self,
        chunks: list[dict],
        progress: Progress,
        task_id,
    ) -> None:
        async def _wrapped(c: dict) -> None:
            await self._process_one(c)
            progress.advance(task_id)

        # Launch all coroutines; the semaphore inside _process_one caps in-flight.
        tasks = [asyncio.create_task(_wrapped(c)) for c in chunks]
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except asyncio.CancelledError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise

    async def _pause_for_review(self) -> None:
        n = len(self.processed_chunk_ids)
        cost = self.running_cost_usd
        elapsed = time.perf_counter() - self.started_at_perf
        per_chunk = elapsed / n if n else 0
        ext_cost = (cost / n) * 3747 if n else 0
        t = Table(title=f"Sanity check @ {n} chunks")
        t.add_column("metric")
        t.add_column("value", justify="right")
        t.add_row("Chunks processed", f"{n:,}")
        t.add_row("Pairs generated", f"{self.pairs_count:,}")
        t.add_row("Actual cost so far", f"${cost:,.4f}")
        t.add_row("Extrapolated full-run cost", f"${ext_cost:,.2f}")
        t.add_row("Wall time", f"{elapsed:.1f}s")
        t.add_row("Sec / chunk", f"{per_chunk:.2f}")
        t.add_row("JSON parse failures", f"{len(self.json_parse_failures)}")
        t.add_row("Embedding failures", f"{len(self.embedding_failures)}")
        t.add_row("Concurrency reductions", f"{self._concurrency_reductions}")
        console.print(t)
        # 10-second countdown
        for s in range(10, 0, -1):
            print(f"Resuming in {s}s... Ctrl+C to abort", flush=True)
            await asyncio.sleep(1)

    # ---- top-level ----

    def _load_chunks(self) -> list[dict]:
        all_chunks: list[dict] = []
        with self.args.in_path.open(encoding="utf-8") as f:
            for line in f:
                all_chunks.append(json.loads(line))
        passing = [c for c in all_chunks if c.get("quality_passed")]
        return passing

    async def run(self) -> int:
        # ---- setup ----
        self.llm = AzureLLMClient()
        if not self.args.skip_embedding_debug:
            self.embedder = AzureEmbeddingClient()
        self.dyn_sem = DynamicSemaphore(self.args.concurrency)

        self._load_checkpoint()
        self._paranoia_merge_from_jsonl()

        # Open output in append mode — never truncate; existing data stays.
        self.args.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_f = self.args.out_path.open("a", encoding="utf-8")

        chunks = self._load_chunks()
        unprocessed = [c for c in chunks if c["chunk_id"] not in self.processed_chunk_ids]

        console.print(
            Panel.fit(
                f"[bold]Full Q&A run — {self.args.concurrency} concurrency[/bold]\n"
                f"  chunks total:        {len(chunks):,}\n"
                f"  already processed:   {len(self.processed_chunk_ids):,}\n"
                f"  to process this run: {len(unprocessed):,}\n"
                f"  pairs/chunk:         {self.args.pairs_per_chunk}\n"
                f"  cost cap:            ${self.args.max_cost_usd:.2f}\n"
                f"  pause at:            {self.args.pause_at_chunks}\n"
                f"  embedding debug:     "
                f"{'OFF' if self.args.skip_embedding_debug else 'ON'}\n"
                f"  in:  {self.args.in_path.relative_to(PROJECT_ROOT)}\n"
                f"  out: {self.args.out_path.relative_to(PROJECT_ROOT)}\n"
                f"  ckpt: {self.args.checkpoint_path.relative_to(PROJECT_ROOT)}"
            )
        )

        if not unprocessed:
            console.print("[green]all chunks already processed; nothing to do[/green]")
            await self._cleanup()
            return 0

        # ---- pause logic ----
        already_past_pause = (
            len(self.processed_chunk_ids) >= self.args.pause_at_chunks
        )
        if already_past_pause:
            phase1: list[dict] = []
            phase2 = unprocessed
        else:
            n_until = self.args.pause_at_chunks - len(self.processed_chunk_ids)
            phase1 = unprocessed[:n_until]
            phase2 = unprocessed[n_until:]

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        )

        exit_code = 0
        try:
            with progress:
                if phase1:
                    task_id = progress.add_task(
                        "Phase 1: pre-pause", total=len(phase1)
                    )
                    await self._process_batch(phase1, progress, task_id)
                    if self.stop_requested:
                        raise _AbortRun(self.abort_reason or "stop requested")
                    progress.stop()
                    await self._pause_for_review()
                    progress.start()

                if phase2 and not self.stop_requested:
                    task_id = progress.add_task(
                        f"Phase 2: remaining ({len(phase2)})", total=len(phase2)
                    )
                    await self._process_batch(phase2, progress, task_id)
        except _AbortRun as e:
            console.print(f"[red]ABORT:[/red] {e}")
            exit_code = 1
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]interrupt received — saving checkpoint[/yellow]")
            self.stop_requested = True
            self.abort_reason = "user interrupt"
            exit_code = 130
        finally:
            self._save_checkpoint()
            await self._cleanup()

        return exit_code

    async def _cleanup(self) -> None:
        if self.out_f is not None:
            try:
                self.out_f.flush()
                os.fsync(self.out_f.fileno())
            except Exception:
                pass
            self.out_f.close()
            self.out_f = None
        if self.llm is not None:
            await self.llm.close()
        if self.embedder is not None:
            await self.embedder.close()


class _AbortRun(Exception):
    """Signals a graceful, controlled abort (cost cap, etc.)."""


# ────────── journal renderer ──────────


def render_journal(
    runner: FullQARunner,
    args: argparse.Namespace,
    total_chunks_in_filtered: int,
    wall_time_s: float,
    exit_code: int,
) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Full Q&A generation — Phase A")
    a("")
    a(
        "_Generated by `scripts/full_qa.py`. Cohere rerank is **dropped** for "
        "v1; embedding cosine is kept as **debug metadata only**, never used "
        "for filtering._"
    )
    a("")

    # ── config ──
    a("## Run configuration")
    a("")
    a(f"- LLM deployment: `{runner.llm.deployment if runner.llm else '?'}`")
    a(f"- Embedding debug: `{'off' if args.skip_embedding_debug else 'on'}`")
    a(f"- Pairs per chunk: `{args.pairs_per_chunk}`")
    a(f"- Concurrency (initial): `{args.concurrency}`")
    a(f"- Concurrency reductions during run: `{runner._concurrency_reductions}`")
    a(
        f"- Effective concurrency at end: "
        f"`{runner.dyn_sem.effective_limit if runner.dyn_sem else args.concurrency}`"
    )
    a(f"- Resume: `{args.resume}`")
    a(f"- Checkpoint interval: `{args.checkpoint_interval}` chunks")
    a(f"- Cost cap: `${args.max_cost_usd:.2f}`")
    a(f"- Pause at chunks: `{args.pause_at_chunks}`")
    a(f"- In:  `{args.in_path.relative_to(PROJECT_ROOT)}`")
    a(f"- Out: `{args.out_path.relative_to(PROJECT_ROOT)}`")
    a(f"- Checkpoint: `{args.checkpoint_path.relative_to(PROJECT_ROOT)}`")
    a(f"- Started: `{runner.started_at_iso}`")
    a(f"- Exit code: `{exit_code}`  ({'success' if exit_code == 0 else 'aborted'})")
    a("")

    # ── outcomes ──
    n_processed = len(runner.processed_chunk_ids)
    a("## Outcomes")
    a("")
    a(f"- Chunks processed:     **{n_processed:,}** of {total_chunks_in_filtered:,}")
    a(f"- Pairs generated:      **{runner.pairs_count:,}**")
    a(
        f"- JSON parse failures: **{len(runner.json_parse_failures)}** "
        f"chunks (chunk yielded zero pairs)"
    )
    a(f"- Embedding failures:   **{len(runner.embedding_failures)}** chunks")
    a(f"- Total wall time:      **{wall_time_s:.1f}s**  ({wall_time_s/60:.1f} min)")
    if n_processed:
        a(f"- Mean time per chunk:  **{wall_time_s/n_processed:.2f}s**")
        a(f"- Mean pairs per chunk: **{runner.pairs_count/n_processed:.2f}**")
    a(
        f"- Final effective concurrency: "
        f"**{runner.dyn_sem.effective_limit if runner.dyn_sem else args.concurrency}** "
        f"({runner._concurrency_reductions} reduction(s) during run)"
    )
    a("")
    if runner.json_parse_failures:
        a("### Chunks with JSON parse / LLM failures")
        a("")
        a("| chunk_id | reason |")
        a("|---|---|")
        for cid, reason in runner.json_parse_failures[:30]:
            short = reason.replace("\n", " ")[:180]
            a(f"| `{cid}` | {short} |")
        if len(runner.json_parse_failures) > 30:
            a(f"| _… and {len(runner.json_parse_failures) - 30} more_ |  |")
        a("")
    if runner.embedding_failures:
        a("### Chunks with embedding-debug failures")
        a("")
        a("| chunk_id | reason |")
        a("|---|---|")
        for cid, reason in runner.embedding_failures[:30]:
            a(f"| `{cid}` | {reason.replace(chr(10), ' ')[:180]} |")
        if len(runner.embedding_failures) > 30:
            a(f"| _… and {len(runner.embedding_failures) - 30} more_ |  |")
        a("")

    # ── cost ──
    a("## Cost (placeholder rates — calibrate against real Azure invoice)")
    a("")
    a("| Item | Value |")
    a("|---|---:|")
    a(f"| Prompt tokens     | {runner.token_usage['prompt_tokens']:,} |")
    a(f"| Completion tokens | {runner.token_usage['completion_tokens']:,} |")
    a(f"| LLM total tokens  | {runner.token_usage['total_tokens']:,} |")
    a(
        f"| Embedding tokens (est.) | "
        f"{runner.token_usage.get('embedding_tokens', 0):,} |"
    )
    llm_input_cost = (
        runner.token_usage["prompt_tokens"] / 1_000_000 * LLM_INPUT_PRICE_PER_1M_USD
    )
    llm_output_cost = (
        runner.token_usage["completion_tokens"]
        / 1_000_000
        * LLM_OUTPUT_PRICE_PER_1M_USD
    )
    embed_cost = (
        runner.token_usage.get("embedding_tokens", 0)
        / 1_000_000
        * EMBEDDING_PRICE_PER_1M_USD
    )
    total_cost = llm_input_cost + llm_output_cost + embed_cost
    a(f"| LLM input cost ($5/1M)   | ${llm_input_cost:,.4f} |")
    a(f"| LLM output cost ($15/1M) | ${llm_output_cost:,.4f} |")
    a(f"| Embedding cost ($0.02/1M est.) | ${embed_cost:,.4f} |")
    a(f"| **Total** | **${total_cost:,.4f}** |")
    if runner.pairs_count:
        a(f"| $ per pair | ${total_cost/runner.pairs_count:.6f} |")
    a(f"| Cost cap | ${args.max_cost_usd:.2f} |")
    a(f"| Cap hit? | {'YES' if runner.abort_reason == 'cost cap exceeded' else 'no'} |")
    a("")
    a(
        "> **All prices are placeholders.** The user is responsible for "
        "calibrating these against the real Azure invoice line item before "
        "drawing budget conclusions."
    )
    a("")

    # ── question type distribution ──
    a("## Question type distribution")
    a("")
    qtypes = collections.Counter()
    sims: list[float] = []
    pair_records: list[dict] = []
    try:
        with args.out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pair_records.append(r)
                qt = r.get("question_type")
                if qt:
                    qtypes[qt] += 1
                if r.get("embedding_cosine_debug") is not None:
                    sims.append(r["embedding_cosine_debug"])
    except FileNotFoundError:
        pass
    total = sum(qtypes.values())
    a("| Type | Count | % |")
    a("|---|---:|---:|")
    for qt in ("factual", "definitional", "procedural", "conceptual"):
        c = qtypes.get(qt, 0)
        pct = c / total * 100 if total else 0
        a(f"| {qt} | {c:,} | {pct:.1f}% |")
    other = total - sum(qtypes.get(qt, 0) for qt in
                        ("factual", "definitional", "procedural", "conceptual"))
    if other:
        a(f"| _other_ | {other:,} | {other/total*100:.1f}% |")
    a("")

    # ── per-source ──
    a("## Per-source breakdown")
    a("")
    a("| source_filename | chunks | pairs | mean pairs/chunk | min pairs/chunk |")
    a("|---|---:|---:|---:|---:|")
    by_source = collections.defaultdict(list)
    for r in pair_records:
        by_source[r["source_filename"]].append(r["chunk_id"])
    # Pair counts per chunk per source
    chunk_pair_counts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    for r in pair_records:
        chunk_pair_counts[r["source_filename"]][r["chunk_id"]] += 1
    rows = []
    for src, chunks_seen in sorted(chunk_pair_counts.items()):
        counts = list(chunks_seen.values())
        rows.append(
            (
                src,
                len(counts),
                sum(counts),
                statistics.mean(counts) if counts else 0,
                min(counts) if counts else 0,
            )
        )
    for src, n_chunks, n_pairs, mean_pc, min_pc in sorted(
        rows, key=lambda r: -r[2]
    ):
        a(f"| `{src}` | {n_chunks} | {n_pairs} | {mean_pc:.2f} | {min_pc} |")
    a("")

    # ── embedding cosine debug ──
    a("## Embedding cosine — debug metadata only (not used for filtering)")
    a("")
    if sims:
        sims_sorted = sorted(sims)

        def pct(p: float) -> float:
            return sims_sorted[min(int(len(sims_sorted) * p), len(sims_sorted) - 1)]

        a("| Statistic | Value |")
        a("|---|---:|")
        a(f"| count   | {len(sims):,} |")
        a(f"| min     | {sims_sorted[0]:.4f} |")
        a(f"| p10     | {pct(0.10):.4f} |")
        a(f"| p25     | {pct(0.25):.4f} |")
        a(f"| median  | {pct(0.50):.4f} |")
        a(f"| p75     | {pct(0.75):.4f} |")
        a(f"| p90     | {pct(0.90):.4f} |")
        a(f"| max     | {sims_sorted[-1]:.4f} |")
        a(f"| mean    | {statistics.mean(sims):.4f} |")
        a("")
        low_pairs = [r for r in pair_records
                     if r.get("embedding_cosine_debug") is not None
                     and r["embedding_cosine_debug"] < 0.45]
        a(f"- Pairs with cosine < 0.45 (NOT filtered out): **{len(low_pairs):,}** "
          f"({len(low_pairs)/len(sims)*100:.1f}%)")
        if low_pairs:
            import random as _r
            rng = _r.Random(42)
            sample = rng.sample(low_pairs, min(3, len(low_pairs)))
            a("")
            a("### Three random low-cosine examples (for inspection — kept in dataset)")
            a("")
            for i, r in enumerate(sample, 1):
                a(f"#### Example {i} — cosine = {r['embedding_cosine_debug']}")
                a(f"- **pair_id:** `{r['pair_id']}`")
                a(f"- **source:** `{r['source_filename']}`")
                a(f"- **question_type:** `{r['question_type']}`")
                a("")
                a(f"**Q:** {r['question']}")
                a("")
                a(f"**A:** {r['answer']}")
                a("")
        a(
            "> Scores are systematically compressed by the question-vs-chunk "
            "length asymmetry — a ~30-token question embedded against ~500 "
            "tokens of chunk slice averages out topical match. Useful only as "
            "a **relative** ranking signal, not an absolute quality threshold."
        )
    else:
        a("_Embedding debug was skipped or all calls failed._")
    a("")

    # ── anything weird ──
    a("## Anything weird")
    a("")
    weird: list[str] = []
    # Sources with low mean pairs/chunk
    low_sources = [r for r in rows if r[3] < 4]
    if low_sources:
        weird.append(
            f"- **{len(low_sources)}** source(s) had mean pairs/chunk < 4: "
            + ", ".join(f"`{r[0]}` ({r[3]:.2f})" for r in low_sources[:8])
            + ("…" if len(low_sources) > 8 else "")
        )
    if runner._concurrency_reductions:
        weird.append(
            f"- Dynamic concurrency reduction fired "
            f"**{runner._concurrency_reductions}** time(s) — Azure deployment hit "
            "sustained rate-limit pressure. Final effective limit: "
            f"{runner.dyn_sem.effective_limit if runner.dyn_sem else args.concurrency}."
        )
    if runner.json_parse_failures:
        weird.append(
            f"- **{len(runner.json_parse_failures)}** chunks failed to produce "
            "any pairs (JSON parse / LLM error). They are still marked "
            "'processed' in the checkpoint to avoid infinite retry — re-run with "
            "a manual `--resume=False` if you want to retry."
        )
    if runner.embedding_failures:
        weird.append(
            f"- **{len(runner.embedding_failures)}** chunks had embedding-debug "
            "failures. Pairs were still saved (embedding_cosine_debug=null)."
        )
    if runner.abort_reason:
        weird.append(f"- Run aborted: **{runner.abort_reason}**.")
    if not weird:
        weird.append(
            "_Nothing materially surprising. The run extrapolated cleanly "
            "from the pilot's per-chunk numbers._"
        )
    for w in weird:
        a(w)
    a("")

    return "\n".join(lines) + "\n"


# ────────── argparse / main ──────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-path", type=Path, default=DEFAULT_IN_PATH)
    p.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    p.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--pairs-per-chunk", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpoint-interval", type=int, default=25)
    p.add_argument("--max-cost-usd", type=float, default=75.0)
    p.add_argument("--pause-at-chunks", type=int, default=50)
    p.add_argument("--skip-embedding-debug", action="store_true", default=False)
    return p


async def amain(args: argparse.Namespace) -> int:
    if not args.in_path.exists():
        console.print(f"[red]input not found: {args.in_path}[/red]")
        return 1

    runner = FullQARunner(args)

    # Best-effort signal flag — KeyboardInterrupt also works via asyncio.run,
    # but this gives clean intermediate behavior on platforms where the
    # asyncio integration is finicky.
    loop = asyncio.get_running_loop()

    def _set_stop():
        runner.stop_requested = True
        runner.abort_reason = "user interrupt"

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _set_stop)
        except NotImplementedError:
            # add_signal_handler isn't available on Windows; rely on
            # KeyboardInterrupt fallback in run().
            pass

    started = time.perf_counter()
    exit_code = await runner.run()
    wall_time_s = time.perf_counter() - started

    # Render and write journal regardless of exit code.
    total_filtered = 0
    with args.in_path.open(encoding="utf-8") as f:
        for line in f:
            total_filtered += 1
    md = render_journal(runner, args, total_filtered, wall_time_s, exit_code)
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.write_text(md, encoding="utf-8")

    # Terminal summary
    t = Table(title="Full run summary")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("Chunks processed", f"{len(runner.processed_chunk_ids):,}")
    t.add_row("Pairs generated", f"{runner.pairs_count:,}")
    t.add_row("JSON parse failures", f"{len(runner.json_parse_failures)}")
    t.add_row("Embedding failures", f"{len(runner.embedding_failures)}")
    t.add_row("Actual cost", f"${runner.running_cost_usd:,.4f}")
    t.add_row("Cost cap", f"${args.max_cost_usd:.2f}")
    t.add_row("Cap hit", "YES" if runner.abort_reason == "cost cap exceeded" else "no")
    t.add_row("Wall time", f"{wall_time_s:.1f}s ({wall_time_s/60:.1f} min)")
    t.add_row("Concurrency reductions", f"{runner._concurrency_reductions}")
    t.add_row("Exit code", str(exit_code))
    console.print(t)
    console.print(f"[green]wrote[/green] {JOURNAL_PATH.relative_to(PROJECT_ROOT)}")

    return exit_code


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]keyboard interrupt at top level[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
