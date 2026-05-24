"""Phase A judge-based eval — v4 (LoRA SFT) vs baseline (gpt-oss-20b, no adapter).

For each example in:
  data/processed/qa/test.jsonl         (1,875)
  data/processed/qa/held_out_source.jsonl (15)

we:
  1. Generate one completion from v4 and one from baseline (Tinker
     sample_async, temperature=0.3, max_tokens=300, concurrency=5/model).
     Only the **final-channel text** is kept — the gpt-oss baseline emits
     a `<|channel|>analysis|>` CoT block first; including it would unfairly
     load the candidate the judge sees with reasoning trace.
  2. Send (question, reference, v4_gen, baseline_gen) to GPT-5.4 as a
     pair-wise judge using the rubric specified in Prompt #7.

Outputs (all under outputs/phase_a_eval/):
  test_v4_gen.jsonl          one row per (pair_id, v4 generation)
  test_baseline_gen.jsonl    one row per (pair_id, baseline generation)
  test_judged.jsonl          one row per (pair_id, judge scores)
  held_out_v4_gen.jsonl
  held_out_baseline_gen.jsonl
  held_out_judged.jsonl
  aggregates.json            means, win rates, per-source, 95% CI
  cost.json                  cost reconciliation

Idempotency:
  Re-running skips any pair_id already present in the corresponding
  gen/judged file. Aggregates and journal are always regenerated.

Hard cost cap:
  $40 across all Azure judge calls. Aborts the judge phase if exceeded
  (already-saved scores survive). Cap excludes Tinker sampling — Tinker
  cost is tracked but not capped (this project has abundant credits).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tinker  # noqa: E402
from tinker import SamplingParams  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from finetune_slm.data.teacher import AzureLLMClient  # noqa: E402

console = Console()

# =============================================================================
# Constants
# =============================================================================

V4_SAMPLER_URL = (
    "tinker://d79c263a-1e99-57b9-b09e-584fc222f3da:train:0/sampler_weights/final-step-00994"
)
BASE_MODEL = "openai/gpt-oss-20b"
RENDERER_NAME = "gpt_oss_no_sysprompt"

TEST_PATH = ROOT / "data" / "processed" / "qa" / "test.jsonl"
HELD_OUT_PATH = ROOT / "data" / "processed" / "qa" / "held_out_source.jsonl"

OUT_DIR = ROOT / "outputs" / "phase_a_eval"

GEN_TEMP = 0.3
GEN_MAX_TOKENS = 300
GEN_CONCURRENCY = 5

JUDGE_CONCURRENCY = 5
HARD_COST_CAP_USD = 40.0
# GPT-5.4 placeholder prices (matches docs/journal/05b convention).
JUDGE_INPUT_PRICE_PER_TOKEN = 5.0 / 1_000_000
JUDGE_OUTPUT_PRICE_PER_TOKEN = 15.0 / 1_000_000

# Tinker placeholder pricing for accounting only (ADR 0001).
TINKER_PRICE_PER_TOKEN = 0.4 / 1_000_000

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of UAV combat doctrine Q&A. Score each model "
    "answer against a reference answer on three axes, 1-5 each. Be strict; 5 "
    "is reserved for answers that match or exceed the reference."
)

JUDGE_USER_TEMPLATE = """Question: {question}
Reference: {reference}
Model A: {model_a}
Model B: {model_b}

Score each model 1-5 on factual_accuracy, completeness, grounding. Then declare a winner: "A", "B", or "tie".
Return strict JSON: {{"model_a": {{"factual_accuracy": <int 1-5>, "completeness": <int 1-5>, "grounding": <int 1-5>}}, "model_b": {{"factual_accuracy": <int 1-5>, "completeness": <int 1-5>, "grounding": <int 1-5>}}, "winner": "A|B|tie", "rationale": "<one sentence>"}}"""

# =============================================================================
# I/O helpers
# =============================================================================


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_pair_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["pair_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.rename(path)


# =============================================================================
# Renderer helpers
# =============================================================================


_FINAL_BLOCK_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|\Z)",
    re.DOTALL,
)


def extract_final_text(content) -> str:
    """Return the final-channel text only — ignore CoT/thinking parts.

    Two parser cases:
    - **List of typed parts** (clean parse): the gpt_oss renderer returns
      e.g. ``[{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]``.
      Keep only the ``type: "text"`` parts.
    - **Raw string** (parse_response returns stop_reason='malformed'): the
      baseline frequently exhausts its 300-token budget inside the
      ``<|channel|>analysis|>`` CoT block and never opens a ``final``
      channel. In that case the renderer hands back the un-segmented raw
      token text as a string. We regex-extract any final block if present;
      otherwise return empty (a real failure mode the judge should see).
    """
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts).strip()
    if isinstance(content, str):
        m = _FINAL_BLOCK_RE.search(content)
        if m:
            return m.group(1).strip()
        return ""
    return str(content).strip()


# =============================================================================
# Generation
# =============================================================================


async def generate_one(
    sc: tinker.SamplingClient,
    renderer,
    example: dict,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        prompt = renderer.build_generation_prompt(example["messages"][:-1])
        params = SamplingParams(
            max_tokens=GEN_MAX_TOKENS,
            temperature=GEN_TEMP,
            stop=renderer.get_stop_sequences(),
        )
        t0 = time.monotonic()
        try:
            resp = await sc.sample_async(
                prompt=prompt, num_samples=1, sampling_params=params
            )
        except Exception as exc:
            return {
                "pair_id": example["pair_id"],
                "source_filename": example["source_filename"],
                "question_type": example["question_type"],
                "generation": "",
                "error": f"{type(exc).__name__}: {exc}",
                "n_output_tokens": 0,
                "wall_seconds": time.monotonic() - t0,
            }
        tokens = resp.sequences[0].tokens
        try:
            parsed, stop_reason = renderer.parse_response(tokens)
            content = parsed.get("content")
            text = extract_final_text(content)
            has_final_block = isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") == "text" for p in content
            )
            if not has_final_block and isinstance(content, str):
                has_final_block = bool(_FINAL_BLOCK_RE.search(content))
        except Exception as exc:
            text = ""
            stop_reason = None
            has_final_block = False
            parse_err = f"{type(exc).__name__}: {exc}"
        else:
            parse_err = None
        row = {
            "pair_id": example["pair_id"],
            "source_filename": example["source_filename"],
            "question_type": example["question_type"],
            "generation": text,
            "n_output_tokens": len(tokens),
            "stop_reason": stop_reason,
            "has_final_block": has_final_block,
            "wall_seconds": round(time.monotonic() - t0, 3),
        }
        if parse_err:
            row["parse_error"] = parse_err
        return row


async def generate_split(
    sampler_path: str | None,
    label: str,
    examples: list[dict],
    out_path: Path,
) -> dict:
    """Generate for every example; skip pair_ids already in out_path."""
    done = load_existing_pair_ids(out_path)
    todo = [ex for ex in examples if ex["pair_id"] not in done]
    console.print(
        f"[bold]{label}[/bold]: {len(examples)} total, "
        f"{len(done)} already done, {len(todo)} to generate"
    )
    if not todo:
        return {"n_generated_now": 0, "n_total": len(examples), "skipped": len(done)}

    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = get_renderer(RENDERER_NAME, tokenizer, model_name=BASE_MODEL)
    service = tinker.ServiceClient()
    sc = service.create_sampling_client(model_path=sampler_path, base_model=BASE_MODEL)

    sem = asyncio.Semaphore(GEN_CONCURRENCY)
    n_done = 0
    total_output_tokens = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"gen {label}", total=len(todo))
        # Schedule all coroutines; they self-throttle via the semaphore.
        coros = [generate_one(sc, renderer, ex, sem) for ex in todo]
        for coro in asyncio.as_completed(coros):
            row = await coro
            append_jsonl(out_path, row)
            n_done += 1
            total_output_tokens += int(row.get("n_output_tokens", 0))
            progress.advance(task)
    return {
        "n_generated_now": n_done,
        "n_total": len(examples),
        "skipped": len(done),
        "output_tokens_total": total_output_tokens,
    }


# =============================================================================
# Judge
# =============================================================================


class JudgeState:
    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.malformed = 0
        self.skipped_cap = 0
        self.skipped_error = 0

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens * JUDGE_INPUT_PRICE_PER_TOKEN
            + self.completion_tokens * JUDGE_OUTPUT_PRICE_PER_TOKEN
        )

    def would_exceed(self, projected_cost: float) -> bool:
        return self.cost_usd + projected_cost > self.cap_usd


def parse_judge_json(content: str) -> dict | None:
    """Strict JSON parse with light cleanup (strip code-fence wrappers)."""
    text = content.strip()
    if text.startswith("```"):
        # Strip leading ```json\n and trailing ```
        text = text.strip("`")
        # After stripping back-ticks "json\n{...}" may remain
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_judge(parsed: dict) -> bool:
    """Schema check on a judge response. Reject anything not 1-5 / A|B|tie."""
    try:
        for key in ("model_a", "model_b"):
            sub = parsed[key]
            for axis in ("factual_accuracy", "completeness", "grounding"):
                v = sub[axis]
                if not isinstance(v, int) or v < 1 or v > 5:
                    return False
        winner = parsed.get("winner")
        if winner not in ("A", "B", "tie"):
            return False
        return True
    except (KeyError, TypeError):
        return False


async def judge_one(
    azure: AzureLLMClient,
    question: str,
    reference: str,
    v4_gen: str,
    baseline_gen: str,
    sem: asyncio.Semaphore,
    state: JudgeState,
) -> dict:
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question, reference=reference,
        model_a=v4_gen, model_b=baseline_gen,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    async with sem:
        # Coarse pre-call cap check: assume up to ~800 tokens worst case for
        # input + 400 for output → ~$0.01. Don't start a call we can't afford.
        projected = 1200 * JUDGE_INPUT_PRICE_PER_TOKEN + 400 * JUDGE_OUTPUT_PRICE_PER_TOKEN
        if state.would_exceed(projected):
            state.skipped_cap += 1
            return {
                "error": "cost_cap_reached",
                "cost_usd_at_skip": state.cost_usd,
            }
        try:
            content, usage = await azure.chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=400,
            )
        except Exception as exc:
            state.skipped_error += 1
            return {"error": f"{type(exc).__name__}: {exc}"}
        state.calls += 1
        state.prompt_tokens += int(usage.get("prompt_tokens", 0))
        state.completion_tokens += int(usage.get("completion_tokens", 0))

        parsed = parse_judge_json(content)
        if parsed is None or not _validate_judge(parsed):
            state.malformed += 1
            # Retry once at temperature 0.3 (deployment may pin to 1.0; pass
            # through and let teacher fall back if needed).
            try:
                content2, usage2 = await azure.chat_completion(
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=400,
                )
            except Exception as exc:
                return {"error": f"retry_failed: {type(exc).__name__}: {exc}",
                        "raw_first": content}
            state.calls += 1
            state.prompt_tokens += int(usage2.get("prompt_tokens", 0))
            state.completion_tokens += int(usage2.get("completion_tokens", 0))
            parsed = parse_judge_json(content2)
            if parsed is None or not _validate_judge(parsed):
                return {"error": "malformed_after_retry",
                        "raw_first": content, "raw_second": content2}
        return {"judge": parsed,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0))}


async def judge_split(
    label: str,
    examples: list[dict],
    v4_gens: dict[str, dict],
    baseline_gens: dict[str, dict],
    out_path: Path,
    state: JudgeState,
) -> dict:
    done = load_existing_pair_ids(out_path)
    todo: list[dict] = []
    for ex in examples:
        pid = ex["pair_id"]
        if pid in done:
            continue
        if pid not in v4_gens or pid not in baseline_gens:
            continue
        todo.append(ex)
    console.print(
        f"[bold]{label} judge[/bold]: {len(examples)} total, "
        f"{len(done)} already judged, {len(todo)} to judge"
    )
    if not todo:
        return {"n_judged_now": 0, "n_total": len(examples), "skipped": len(done)}

    azure = AzureLLMClient()
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def _one(ex):
        question = ex["messages"][1]["content"]
        reference = ex["messages"][2]["content"]
        v4_gen = v4_gens[ex["pair_id"]].get("generation", "")
        baseline_gen = baseline_gens[ex["pair_id"]].get("generation", "")
        result = await judge_one(
            azure, question, reference, v4_gen, baseline_gen, sem, state
        )
        row = {
            "pair_id": ex["pair_id"],
            "source_filename": ex["source_filename"],
            "question_type": ex["question_type"],
            "question": question,
            "reference": reference,
            "v4_generation": v4_gen,
            "baseline_generation": baseline_gen,
        }
        row.update(result)
        return row

    n_now = 0
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("${task.fields[cost]}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"judge {label}", total=len(todo), cost=f"{state.cost_usd:.3f}"
            )
            coros = [_one(ex) for ex in todo]
            for coro in asyncio.as_completed(coros):
                row = await coro
                append_jsonl(out_path, row)
                n_now += 1
                progress.update(
                    task, advance=1, cost=f"{state.cost_usd:.3f}"
                )
    finally:
        await azure.close()

    return {"n_judged_now": n_now, "n_total": len(examples), "skipped": len(done)}


# =============================================================================
# Aggregation
# =============================================================================


def _gens_by_pair_id(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path) if path.exists() else []
    return {r["pair_id"]: r for r in rows}


def _bootstrap_ci(values: list[int], n_resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(n_resamples):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    lo = samples[int(0.025 * n_resamples)]
    hi = samples[int(0.975 * n_resamples)]
    return (lo, hi)


def _bootstrap_ci_diff(
    a: list[float], b: list[float], n_resamples: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """95% percentile bootstrap CI on mean(a) - mean(b), paired."""
    if not a or len(a) != len(b):
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(a)
    samples = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        ma = sum(a[i] for i in idxs) / n
        mb = sum(b[i] for i in idxs) / n
        samples.append(ma - mb)
    samples.sort()
    lo = samples[int(0.025 * n_resamples)]
    hi = samples[int(0.975 * n_resamples)]
    return (lo, hi)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _subset_metrics(valid: list[dict]) -> dict:
    """Headline metrics over a subset of judged rows."""
    if not valid:
        return {"n": 0}
    v4_acc = [r["judge"]["model_a"]["factual_accuracy"] for r in valid]
    v4_comp = [r["judge"]["model_a"]["completeness"] for r in valid]
    v4_ground = [r["judge"]["model_a"]["grounding"] for r in valid]
    bl_acc = [r["judge"]["model_b"]["factual_accuracy"] for r in valid]
    bl_comp = [r["judge"]["model_b"]["completeness"] for r in valid]
    bl_ground = [r["judge"]["model_b"]["grounding"] for r in valid]
    winners = [r["judge"]["winner"] for r in valid]
    n = len(valid)
    return {
        "n": n,
        "v4_factual_accuracy_mean": _mean(v4_acc),
        "v4_completeness_mean": _mean(v4_comp),
        "v4_grounding_mean": _mean(v4_ground),
        "baseline_factual_accuracy_mean": _mean(bl_acc),
        "baseline_completeness_mean": _mean(bl_comp),
        "baseline_grounding_mean": _mean(bl_ground),
        "v4_win_rate": sum(1 for w in winners if w == "A") / n,
        "baseline_win_rate": sum(1 for w in winners if w == "B") / n,
        "tie_rate": sum(1 for w in winners if w == "tie") / n,
    }


def aggregate_split(judged: list[dict]) -> dict:
    """Compute mean axis scores, win rate, per-type breakdown, per-source top/bottom 5."""
    valid = [r for r in judged if "judge" in r]
    n = len(valid)
    if n == 0:
        return {"n": 0}

    v4_acc = [r["judge"]["model_a"]["factual_accuracy"] for r in valid]
    v4_comp = [r["judge"]["model_a"]["completeness"] for r in valid]
    v4_ground = [r["judge"]["model_a"]["grounding"] for r in valid]
    bl_acc = [r["judge"]["model_b"]["factual_accuracy"] for r in valid]
    bl_comp = [r["judge"]["model_b"]["completeness"] for r in valid]
    bl_ground = [r["judge"]["model_b"]["grounding"] for r in valid]

    winners = [r["judge"]["winner"] for r in valid]
    n_v4_wins = sum(1 for w in winners if w == "A")
    n_bl_wins = sum(1 for w in winners if w == "B")
    n_ties = sum(1 for w in winners if w == "tie")
    win_rate_v4 = n_v4_wins / n
    win_rate_bl = n_bl_wins / n
    tie_rate = n_ties / n

    # 95% CI on v4 win rate (treat each example's win indicator as 0/1)
    win_ind = [1 if w == "A" else 0 for w in winners]
    win_ci = _bootstrap_ci(win_ind, n_resamples=1000)

    # 95% CI on (v4 - baseline) mean per axis
    acc_ci = _bootstrap_ci_diff(v4_acc, bl_acc)
    comp_ci = _bootstrap_ci_diff(v4_comp, bl_comp)
    ground_ci = _bootstrap_ci_diff(v4_ground, bl_ground)

    # Per question_type
    by_type: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "v4_wins": 0, "bl_wins": 0, "ties": 0,
                 "v4_acc": [], "v4_comp": [], "v4_ground": [],
                 "bl_acc": [], "bl_comp": [], "bl_ground": []}
    )
    for r in valid:
        qt = r["question_type"]
        d = by_type[qt]
        d["n"] += 1
        w = r["judge"]["winner"]
        if w == "A":
            d["v4_wins"] += 1
        elif w == "B":
            d["bl_wins"] += 1
        else:
            d["ties"] += 1
        d["v4_acc"].append(r["judge"]["model_a"]["factual_accuracy"])
        d["v4_comp"].append(r["judge"]["model_a"]["completeness"])
        d["v4_ground"].append(r["judge"]["model_a"]["grounding"])
        d["bl_acc"].append(r["judge"]["model_b"]["factual_accuracy"])
        d["bl_comp"].append(r["judge"]["model_b"]["completeness"])
        d["bl_ground"].append(r["judge"]["model_b"]["grounding"])
    qt_summary: dict[str, dict[str, float]] = {}
    for qt, d in by_type.items():
        qt_summary[qt] = {
            "n": d["n"],
            "v4_win_rate": d["v4_wins"] / d["n"],
            "baseline_win_rate": d["bl_wins"] / d["n"],
            "tie_rate": d["ties"] / d["n"],
            "v4_factual_accuracy_mean": _mean(d["v4_acc"]),
            "v4_completeness_mean": _mean(d["v4_comp"]),
            "v4_grounding_mean": _mean(d["v4_ground"]),
            "baseline_factual_accuracy_mean": _mean(d["bl_acc"]),
            "baseline_completeness_mean": _mean(d["bl_comp"]),
            "baseline_grounding_mean": _mean(d["bl_ground"]),
        }

    # Per-source — average v4 advantage = mean(v4 axes) - mean(baseline axes).
    by_src: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "v4_total": [], "bl_total": [], "v4_wins": 0}
    )
    for r in valid:
        s = r["source_filename"]
        v4_tot = (r["judge"]["model_a"]["factual_accuracy"]
                  + r["judge"]["model_a"]["completeness"]
                  + r["judge"]["model_a"]["grounding"])
        bl_tot = (r["judge"]["model_b"]["factual_accuracy"]
                  + r["judge"]["model_b"]["completeness"]
                  + r["judge"]["model_b"]["grounding"])
        d = by_src[s]
        d["n"] += 1
        d["v4_total"].append(v4_tot)
        d["bl_total"].append(bl_tot)
        if r["judge"]["winner"] == "A":
            d["v4_wins"] += 1
    src_rows = []
    for s, d in by_src.items():
        if d["n"] < 3:
            continue
        src_rows.append({
            "source_filename": s,
            "n": d["n"],
            "v4_total_mean": _mean(d["v4_total"]),
            "baseline_total_mean": _mean(d["bl_total"]),
            "delta": _mean(d["v4_total"]) - _mean(d["bl_total"]),
            "v4_win_rate": d["v4_wins"] / d["n"],
        })
    # Sort by delta to surface best/worst
    src_rows.sort(key=lambda r: r["delta"], reverse=True)
    top5 = src_rows[:5]
    bot5 = list(reversed(src_rows[-5:]))

    # Subset: judged rows where baseline produced a non-empty final answer
    # (vs the dominant failure mode where baseline ran out of 300 tokens in
    # CoT). This is the "apples-to-apples doctrine knowledge" comparison —
    # full-set numbers are dominated by baseline's 300-token budget exhaustion.
    nonempty_subset = [r for r in valid if (r.get("baseline_generation") or "").strip()]

    return {
        "n_examples_judged": n,
        "n_examples_missing_judge": len(judged) - n,
        "v4": {
            "factual_accuracy_mean": _mean(v4_acc),
            "completeness_mean": _mean(v4_comp),
            "grounding_mean": _mean(v4_ground),
        },
        "baseline": {
            "factual_accuracy_mean": _mean(bl_acc),
            "completeness_mean": _mean(bl_comp),
            "grounding_mean": _mean(bl_ground),
        },
        "delta_v4_minus_baseline": {
            "factual_accuracy": _mean(v4_acc) - _mean(bl_acc),
            "completeness": _mean(v4_comp) - _mean(bl_comp),
            "grounding": _mean(v4_ground) - _mean(bl_ground),
        },
        "delta_ci_95": {
            "factual_accuracy": acc_ci,
            "completeness": comp_ci,
            "grounding": ground_ci,
        },
        "win_rate": {
            "v4": win_rate_v4,
            "baseline": win_rate_bl,
            "tie": tie_rate,
            "v4_ci_95": win_ci,
        },
        "by_question_type": qt_summary,
        "by_source": {
            "top5_v4_helps_most": top5,
            "bottom5_v4_helps_least": bot5,
        },
        "subset_baseline_produced_answer": _subset_metrics(nonempty_subset),
    }


# =============================================================================
# Main
# =============================================================================


async def stage_generate() -> dict:
    test_examples = read_jsonl(TEST_PATH)
    held_examples = read_jsonl(HELD_OUT_PATH)
    gen_summary: dict = {}
    # Held-out first as a smoke test (15 examples) — any wiring / schema bug
    # surfaces in seconds, not after burning 30 minutes on the test set.
    # Within each split, v4 and baseline run concurrently (separate sampling
    # clients) — each still self-throttles at concurrency=5 per the spec.
    h_v4, h_bl = await asyncio.gather(
        generate_split(V4_SAMPLER_URL, "held/v4", held_examples,
                       OUT_DIR / "held_out_v4_gen.jsonl"),
        generate_split(None, "held/baseline", held_examples,
                       OUT_DIR / "held_out_baseline_gen.jsonl"),
    )
    gen_summary["held_out_v4"] = h_v4
    gen_summary["held_out_baseline"] = h_bl
    t_v4, t_bl = await asyncio.gather(
        generate_split(V4_SAMPLER_URL, "test/v4", test_examples,
                       OUT_DIR / "test_v4_gen.jsonl"),
        generate_split(None, "test/baseline", test_examples,
                       OUT_DIR / "test_baseline_gen.jsonl"),
    )
    gen_summary["test_v4"] = t_v4
    gen_summary["test_baseline"] = t_bl
    return gen_summary


async def stage_judge() -> tuple[dict, JudgeState]:
    test_examples = read_jsonl(TEST_PATH)
    held_examples = read_jsonl(HELD_OUT_PATH)
    test_v4 = _gens_by_pair_id(OUT_DIR / "test_v4_gen.jsonl")
    test_bl = _gens_by_pair_id(OUT_DIR / "test_baseline_gen.jsonl")
    held_v4 = _gens_by_pair_id(OUT_DIR / "held_out_v4_gen.jsonl")
    held_bl = _gens_by_pair_id(OUT_DIR / "held_out_baseline_gen.jsonl")
    state = JudgeState(cap_usd=HARD_COST_CAP_USD)
    summary = {}
    # Held-out (15) first → fast smoke test on judge wiring + JSON parsing
    # before committing to ~$15 on the test set.
    summary["held_out"] = await judge_split(
        "held_out", held_examples, held_v4, held_bl,
        OUT_DIR / "held_out_judged.jsonl", state,
    )
    summary["test"] = await judge_split(
        "test", test_examples, test_v4, test_bl,
        OUT_DIR / "test_judged.jsonl", state,
    )
    return summary, state


def stage_aggregate() -> dict:
    test_judged = read_jsonl(OUT_DIR / "test_judged.jsonl") if (OUT_DIR / "test_judged.jsonl").exists() else []
    held_judged = read_jsonl(OUT_DIR / "held_out_judged.jsonl") if (OUT_DIR / "held_out_judged.jsonl").exists() else []
    aggregates = {
        "test": aggregate_split(test_judged),
        "held_out": aggregate_split(held_judged),
    }
    atomic_write_json(OUT_DIR / "aggregates.json", aggregates)
    return aggregates


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gen_summary = {}
    judge_summary = {}
    judge_state = JudgeState(cap_usd=HARD_COST_CAP_USD)

    if args.do_gen:
        console.rule("Stage 1 — Generation")
        gen_summary = await stage_generate()
        console.print(gen_summary)
    if args.do_judge:
        console.rule("Stage 2 — Judge")
        judge_summary, judge_state = await stage_judge()
        console.print(judge_summary)
        console.print(
            f"judge cost: ${judge_state.cost_usd:.3f} (cap ${HARD_COST_CAP_USD})"
        )
    if args.do_aggregate:
        console.rule("Stage 3 — Aggregate")
        aggregates = stage_aggregate()
        console.print({k: {"test_n": v.get("n_examples_judged"),
                           "held_n": aggregates["held_out"].get("n_examples_judged")}
                       for k, v in [("agg", aggregates["test"])]})

    # Cost ledger — when judge stage did not run this invocation, reconstruct
    # the totals from the persisted judged rows (each row stores prompt_tokens
    # + completion_tokens of its successful judge call).
    if args.do_judge:
        judge_cost_block = {
            "calls": judge_state.calls,
            "prompt_tokens": judge_state.prompt_tokens,
            "completion_tokens": judge_state.completion_tokens,
            "input_cost_usd": judge_state.prompt_tokens * JUDGE_INPUT_PRICE_PER_TOKEN,
            "output_cost_usd": judge_state.completion_tokens * JUDGE_OUTPUT_PRICE_PER_TOKEN,
            "total_cost_usd": judge_state.cost_usd,
            "malformed_first_pass": judge_state.malformed,
            "skipped_cost_cap": judge_state.skipped_cap,
            "skipped_error": judge_state.skipped_error,
            "hard_cap_usd": HARD_COST_CAP_USD,
            "source": "live_run",
        }
    else:
        ptok = ctok = calls = 0
        errored = cost_cap_skipped = malformed_after_retry = 0
        for split_path in ("test_judged.jsonl", "held_out_judged.jsonl"):
            for r in read_jsonl(OUT_DIR / split_path) if (OUT_DIR / split_path).exists() else []:
                if "judge" in r:
                    ptok += int(r.get("prompt_tokens", 0))
                    ctok += int(r.get("completion_tokens", 0))
                    calls += 1
                elif "error" in r:
                    err = r["error"]
                    if "cost_cap" in err:
                        cost_cap_skipped += 1
                    elif "malformed" in err:
                        malformed_after_retry += 1
                    else:
                        errored += 1
        judge_cost_block = {
            "calls": calls,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "input_cost_usd": ptok * JUDGE_INPUT_PRICE_PER_TOKEN,
            "output_cost_usd": ctok * JUDGE_OUTPUT_PRICE_PER_TOKEN,
            "total_cost_usd": ptok * JUDGE_INPUT_PRICE_PER_TOKEN + ctok * JUDGE_OUTPUT_PRICE_PER_TOKEN,
            "hard_cap_usd": HARD_COST_CAP_USD,
            "skipped_error": errored,
            "skipped_cost_cap": cost_cap_skipped,
            "malformed_after_retry": malformed_after_retry,
            "source": "reconstructed_from_judged_files",
        }
    cost_payload = {
        "judge": judge_cost_block,
        "gen_summary": gen_summary,
        "judge_summary": judge_summary,
    }
    cost_path = OUT_DIR / "cost.json"
    if cost_path.exists():
        try:
            prev = json.loads(cost_path.read_text())
            cost_payload["previous_judge_total_cost_usd"] = prev.get("judge", {}).get("total_cost_usd")
        except Exception:
            pass
    atomic_write_json(cost_path, cost_payload)
    console.print(f"cost ledger: {cost_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gen", action="store_true", help="Skip generation stage")
    parser.add_argument("--no-judge", action="store_true", help="Skip judging stage")
    parser.add_argument("--no-aggregate", action="store_true", help="Skip aggregation")
    args = parser.parse_args()
    args.do_gen = not args.no_gen
    args.do_judge = not args.no_judge
    args.do_aggregate = not args.no_aggregate
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
