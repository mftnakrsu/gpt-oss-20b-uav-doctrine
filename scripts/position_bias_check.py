"""Position-bias control for Phase A judge eval.

Background:
  In scripts/eval_phase_a.py (Prompt #7), v4 was *always* placed as Model A
  in the pair-wise judge prompt. Pair-wise LLM judges often favor whoever
  is in position A — so the original 82.6% v4 win rate is not yet
  bias-corrected.

What this does:
  1. Sample 300 examples from outputs/phase_a_eval/test_judged.jsonl,
     stratified ~75 per question_type, seed=4242.
  2. For each, re-judge with positions swapped — baseline as Model A,
     v4 as Model B. Same rubric, same system prompt as Prompt #7.
  3. Save each swap result to position_bias_judged.jsonl with the
     mapping back to MODELS so we can compute agreement + bias-corrected
     win rate.

Hard cap: $5 Azure judge cost. Cost is computed from per-call usage; cap
is checked before each call so we never overshoot.

Idempotent: re-running overwrites position_bias_judged.jsonl from scratch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

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

from finetune_slm.data.teacher import AzureLLMClient  # noqa: E402

console = Console()

# Identical to scripts/eval_phase_a.py — do not change.
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

# Same GPT-5.4 placeholder rates as Prompt #7.
INPUT_PRICE = 5.0 / 1_000_000
OUTPUT_PRICE = 15.0 / 1_000_000


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


def parse_judge_json(content: str) -> dict | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_judge(parsed: dict) -> bool:
    try:
        for key in ("model_a", "model_b"):
            sub = parsed[key]
            for axis in ("factual_accuracy", "completeness", "grounding"):
                v = sub[axis]
                if not isinstance(v, int) or v < 1 or v > 5:
                    return False
        return parsed.get("winner") in ("A", "B", "tie")
    except (KeyError, TypeError):
        return False


def stratified_sample(rows: list[dict], target_per_type: int, seed: int) -> list[dict]:
    """Group rows by question_type; sample target_per_type from each."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    rng = random.Random(seed)
    picked: list[dict] = []
    actual: dict[str, int] = {}
    for qt in sorted(by_type.keys()):
        pool = by_type[qt]
        n = min(target_per_type, len(pool))
        actual[qt] = n
        picked.extend(rng.sample(pool, n))
    rng.shuffle(picked)
    return picked, actual


class BudgetState:
    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.malformed_first = 0
        self.skipped_cap = 0
        self.skipped_error = 0

    @property
    def cost_usd(self) -> float:
        return self.prompt_tokens * INPUT_PRICE + self.completion_tokens * OUTPUT_PRICE


async def judge_swap_one(
    azure: AzureLLMClient,
    ex: dict,
    sem: asyncio.Semaphore,
    state: BudgetState,
) -> dict:
    """Call judge with v4 in B slot, baseline in A slot."""
    # In the swap, positions flip:
    #   Model A = baseline_generation (was B in Prompt #7)
    #   Model B = v4_generation       (was A in Prompt #7)
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=ex["question"],
        reference=ex["reference"],
        model_a=ex["baseline_generation"],
        model_b=ex["v4_generation"],
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    async with sem:
        projected = 1200 * INPUT_PRICE + 400 * OUTPUT_PRICE
        if state.cost_usd + projected > state.cap_usd:
            state.skipped_cap += 1
            return {"error": "cost_cap_reached", "cost_usd_at_skip": state.cost_usd}
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
            state.malformed_first += 1
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
        return {"judge_swapped": parsed,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0))}


def _map_position_to_model(position: str, mode: str) -> str:
    """position ∈ {'A','B','tie'}, mode ∈ {'original','swapped'}.

    original mode: A=v4, B=baseline
    swapped mode:  A=baseline, B=v4
    """
    if position == "tie":
        return "tie"
    if mode == "original":
        return "v4" if position == "A" else "baseline"
    elif mode == "swapped":
        return "baseline" if position == "A" else "v4"
    raise ValueError(mode)


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Idempotent: start fresh.
    if out_path.exists():
        console.print(f"[yellow]overwriting existing {out_path}[/yellow]")
        out_path.unlink()

    console.rule("Phase 1 — Sampling")
    raw_rows = read_jsonl(in_path)
    judged_rows = [r for r in raw_rows if "judge" in r]
    console.print(f"loaded {len(raw_rows)} rows from {in_path}; {len(judged_rows)} with valid judge")

    target = args.sample_size // 4
    picked, actual_per_type = stratified_sample(judged_rows, target, args.seed)
    console.print(f"sampled {len(picked)} (target {args.sample_size}, target per type {target})")
    for qt, n in sorted(actual_per_type.items()):
        console.print(f"  {qt}: {n}")

    console.rule("Phase 2 — Swap re-judging")
    state = BudgetState(cap_usd=args.max_cost_usd)
    azure = AzureLLMClient()
    sem = asyncio.Semaphore(args.concurrency)

    async def _one(ex):
        result = await judge_swap_one(azure, ex, sem, state)
        original_winner_pos = ex["judge"]["winner"]
        swapped_winner_pos = result.get("judge_swapped", {}).get("winner")
        row = {
            "pair_id": ex["pair_id"],
            "source_filename": ex["source_filename"],
            "question_type": ex["question_type"],
            "question": ex["question"],
            "reference": ex["reference"],
            "v4_generation": ex["v4_generation"],
            "baseline_generation": ex["baseline_generation"],
            "original_winner_position": original_winner_pos,
            "original_winner_model": _map_position_to_model(original_winner_pos, "original"),
            "original_scores": {
                "model_a_v4": ex["judge"]["model_a"],
                "model_b_baseline": ex["judge"]["model_b"],
                "rationale": ex["judge"].get("rationale", ""),
            },
        }
        if swapped_winner_pos is not None:
            row["swapped_winner_position"] = swapped_winner_pos
            row["swapped_winner_model"] = _map_position_to_model(swapped_winner_pos, "swapped")
            row["swapped_scores"] = {
                "model_a_baseline": result["judge_swapped"]["model_a"],
                "model_b_v4": result["judge_swapped"]["model_b"],
                "rationale": result["judge_swapped"].get("rationale", ""),
            }
            row["agreement"] = (row["original_winner_model"] == row["swapped_winner_model"])
        else:
            row["error"] = result.get("error", "?")
        return row

    n_done = 0
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("${task.fields[cost]}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "swap-judge", total=len(picked), cost=f"{state.cost_usd:.3f}"
            )
            coros = [_one(ex) for ex in picked]
            for coro in asyncio.as_completed(coros):
                row = await coro
                append_jsonl(out_path, row)
                n_done += 1
                progress.update(task, advance=1, cost=f"{state.cost_usd:.3f}")
    finally:
        await azure.close()

    console.print(f"\nfinished: {n_done} re-judged; cost ${state.cost_usd:.3f} / ${args.max_cost_usd:.2f}")
    console.print(
        f"  calls={state.calls} malformed_first={state.malformed_first} "
        f"skipped_cap={state.skipped_cap} skipped_error={state.skipped_error}"
    )

    # Persist cost ledger alongside the data file.
    cost_path = out_path.parent / "position_bias_cost.json"
    cost_path.write_text(json.dumps({
        "calls": state.calls,
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "input_cost_usd": state.prompt_tokens * INPUT_PRICE,
        "output_cost_usd": state.completion_tokens * OUTPUT_PRICE,
        "total_cost_usd": state.cost_usd,
        "hard_cap_usd": args.max_cost_usd,
        "malformed_first_pass": state.malformed_first,
        "skipped_cost_cap": state.skipped_cap,
        "skipped_error": state.skipped_error,
        "sample_size": len(picked),
        "actual_per_question_type": actual_per_type,
        "seed": args.seed,
    }, indent=2))
    console.print(f"cost ledger: {cost_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-path", default="outputs/phase_a_eval/test_judged.jsonl")
    p.add_argument("--out-path", default="outputs/phase_a_eval/position_bias_judged.jsonl")
    p.add_argument("--sample-size", type=int, default=300)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--max-cost-usd", type=float, default=5.0)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
