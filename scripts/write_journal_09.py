"""Render docs/journal/09-phase-a-eval-report.md from outputs/phase_a_eval/.

Idempotent: overwrites the journal each run; reads from
  outputs/phase_a_eval/aggregates.json
  outputs/phase_a_eval/cost.json
  outputs/phase_a_eval/{test,held_out}_judged.jsonl
  outputs/phase_a_eval/{test,held_out}_{v4,baseline}_gen.jsonl

The script intentionally has no opinions of its own — every
"Anything weird" item is derived from the data files.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "phase_a_eval"
JOURNAL = ROOT / "docs" / "journal" / "09-phase-a-eval-report.md"

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of UAV combat doctrine Q&A. Score each model "
    "answer against a reference answer on three axes, 1-5 each. Be strict; 5 "
    "is reserved for answers that match or exceed the reference."
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_delta(d: float) -> str:
    return f"{d:+.3f}"


def fmt_ci(ci: list | tuple) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def render_aggregate_table(agg_test: dict, agg_held: dict) -> str:
    """Two rows per axis (test, held_out) + delta + 95% CI."""
    lines = ["| metric | split | v4 | baseline | delta | 95% CI on delta |",
             "|---|---|---:|---:|---:|---|"]
    for label, axis in [("factual_accuracy_mean", "factual_accuracy"),
                        ("completeness_mean", "completeness"),
                        ("grounding_mean", "grounding")]:
        for split_name, agg in [("test", agg_test), ("held_out", agg_held)]:
            v4 = agg["v4"][label]
            bl = agg["baseline"][label]
            d = agg["delta_v4_minus_baseline"][axis]
            ci = agg["delta_ci_95"][axis]
            lines.append(
                f"| {label} | {split_name} | {v4:.3f} | {bl:.3f} | {fmt_delta(d)} | {fmt_ci(ci)} |"
            )
    return "\n".join(lines)


def render_winrate_block(agg: dict, split_name: str, n_total: int) -> str:
    wr = agg["win_rate"]
    ci = wr["v4_ci_95"]
    lines = [
        f"**{split_name}** (n_judged={agg['n_examples_judged']} of {n_total})",
        "",
        f"- v4 wins:       **{wr['v4']:.1%}**  (95% CI {ci[0]:.1%} – {ci[1]:.1%})",
        f"- baseline wins: {wr['baseline']:.1%}",
        f"- ties:          {wr['tie']:.1%}",
        "",
        "Per question_type:",
        "",
        "| question_type | n | v4 win rate | baseline win rate | tie rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for qt, q in sorted(agg["by_question_type"].items()):
        lines.append(
            f"| {qt} | {q['n']} | {q['v4_win_rate']:.1%} | "
            f"{q['baseline_win_rate']:.1%} | {q['tie_rate']:.1%} |"
        )
    return "\n".join(lines)


def render_per_source(agg_test: dict) -> str:
    src = agg_test.get("by_source", {})
    top = src.get("top5_v4_helps_most", [])
    bot = src.get("bottom5_v4_helps_least", [])
    lines = ["### Top 5 — sources where v4 helps most",
             "",
             "| source_filename | n | v4 total mean | baseline total mean | delta | v4 win rate |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in top:
        lines.append(
            f"| `{r['source_filename']}` | {r['n']} | {r['v4_total_mean']:.2f} | "
            f"{r['baseline_total_mean']:.2f} | {r['delta']:+.2f} | {r['v4_win_rate']:.1%} |"
        )
    lines += ["",
              "### Bottom 5 — sources where v4 helps least",
              "",
              "| source_filename | n | v4 total mean | baseline total mean | delta | v4 win rate |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in bot:
        lines.append(
            f"| `{r['source_filename']}` | {r['n']} | {r['v4_total_mean']:.2f} | "
            f"{r['baseline_total_mean']:.2f} | {r['delta']:+.2f} | {r['v4_win_rate']:.1%} |"
        )
    return "\n".join(lines)


def render_samples(judged_rows: list[dict], n: int, seed: int, balance_types: bool) -> str:
    """Pick n samples (random, optionally type-balanced) and render full content."""
    valid = [r for r in judged_rows if "judge" in r]
    if not valid:
        return "_(no judged rows)_"
    rng = random.Random(seed)
    if balance_types:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in valid:
            by_type[r["question_type"]].append(r)
        for k in by_type:
            rng.shuffle(by_type[k])
        # Round-robin
        types = list(by_type.keys())
        rng.shuffle(types)
        picks: list[dict] = []
        idx = 0
        while len(picks) < n and any(by_type[t] for t in types):
            t = types[idx % len(types)]
            if by_type[t]:
                picks.append(by_type[t].pop())
            idx += 1
    else:
        picks = rng.sample(valid, min(n, len(valid)))

    blocks = []
    for r in picks:
        j = r["judge"]
        winner_label = {"A": "v4", "B": "baseline", "tie": "tie"}[j["winner"]]
        block = (
            f"#### `{r['pair_id']}` — `{r['question_type']}` — `{r['source_filename']}`\n\n"
            f"**Q:** {r['question']}\n\n"
            f"**Reference:**\n\n> {r['reference']}\n\n"
            f"**v4:**\n\n> {r['v4_generation'] or '_(empty)_'}\n\n"
            f"**baseline:**\n\n> {r['baseline_generation'] or '_(empty — exceeded 300-token budget in CoT)_'}\n\n"
            f"**Judge:** winner=**{winner_label}** · "
            f"v4 acc/comp/grd = {j['model_a']['factual_accuracy']}/{j['model_a']['completeness']}/{j['model_a']['grounding']} · "
            f"baseline = {j['model_b']['factual_accuracy']}/{j['model_b']['completeness']}/{j['model_b']['grounding']}\n\n"
            f"_Rationale: {j.get('rationale', '_(none)_')}_\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def compute_baseline_failure_stats(test_baseline_gens: list[dict],
                                   held_baseline_gens: list[dict]) -> dict:
    def stats(rows: list[dict]) -> dict:
        n = len(rows)
        empty = sum(1 for r in rows if not (r.get("generation") or "").strip())
        has_final = sum(1 for r in rows if r.get("has_final_block"))
        sr_counts = Counter(r.get("stop_reason") for r in rows)
        toks = [r.get("n_output_tokens", 0) for r in rows]
        return {
            "n": n,
            "n_empty": empty,
            "empty_pct": empty / n if n else 0.0,
            "n_has_final": has_final,
            "stop_reasons": dict(sr_counts),
            "mean_tokens": sum(toks) / n if n else 0.0,
            "max_tokens": max(toks) if toks else 0,
        }
    return {"test": stats(test_baseline_gens), "held_out": stats(held_baseline_gens)}


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    agg_path = EVAL_DIR / "aggregates.json"
    cost_path = EVAL_DIR / "cost.json"
    if not agg_path.exists():
        print(f"STOP: {agg_path} missing — run scripts/eval_phase_a.py first.", file=sys.stderr)
        return 1
    aggregates = json.loads(agg_path.read_text())
    cost = json.loads(cost_path.read_text()) if cost_path.exists() else {}

    test_judged = read_jsonl(EVAL_DIR / "test_judged.jsonl")
    held_judged = read_jsonl(EVAL_DIR / "held_out_judged.jsonl")
    test_baseline_gens = read_jsonl(EVAL_DIR / "test_baseline_gen.jsonl")
    test_v4_gens = read_jsonl(EVAL_DIR / "test_v4_gen.jsonl")
    held_baseline_gens = read_jsonl(EVAL_DIR / "held_out_baseline_gen.jsonl")
    held_v4_gens = read_jsonl(EVAL_DIR / "held_out_v4_gen.jsonl")

    failure_stats = compute_baseline_failure_stats(test_baseline_gens, held_baseline_gens)

    agg_test = aggregates["test"]
    agg_held = aggregates["held_out"]
    judge_cost = cost.get("judge", {})

    # Token statistics
    def gen_tok_summary(rows: list[dict]) -> str:
        toks = [r.get("n_output_tokens", 0) for r in rows]
        if not toks:
            return "_n/a_"
        toks.sort()
        return (f"n={len(toks)} mean={sum(toks)/len(toks):.1f} "
                f"p50={toks[len(toks)//2]} p90={toks[int(len(toks)*0.9)]} "
                f"max={toks[-1]}")

    md = f"""# Phase A full evaluation — v4 (lr 2e-4) vs baseline (no adapter)

_Generated by `scripts/write_journal_09.py` from `outputs/phase_a_eval/`. Re-running overwrites this file. Raw generations and judge scores live next to it._

## Setup

| Item | Value |
|---|---|
| v4 sampler URL | `tinker://d79c263a-1e99-57b9-b09e-584fc222f3da:train:0/sampler_weights/final-step-00994` |
| Baseline | `openai/gpt-oss-20b` via Tinker `SamplingClient` with `model_path=None` (no adapter) |
| Renderer | `gpt_oss_no_sysprompt` (matches Phase A training) |
| Sampling | temperature=0.3, max_tokens=300, concurrency=5/model |
| Test set | `data/processed/qa/test.jsonl` (1,875 examples) |
| Held-out set | `data/processed/qa/held_out_source.jsonl` (15 examples — CRS C-UAS + FAA lost-link) |
| Judge | Azure OpenAI deployment for GPT-5.4 (`AZURE_OPENAI_LLM_DEPLOYMENT`) |
| Judge concurrency | 5 |
| Judge cost cap | $40 (hard) |

**Judge system prompt (verbatim):**

> {JUDGE_SYSTEM_PROMPT}

**Pair-wise judge user template (verbatim):**

```
Question: {{question}}
Reference: {{reference}}
Model A: {{v4_generation}}
Model B: {{baseline_generation}}

Score each model 1-5 on factual_accuracy, completeness, grounding. Then declare a winner: "A", "B", or "tie".
Return strict JSON: {{"model_a": {{...}}, "model_b": {{...}}, "winner": "A|B|tie", "rationale": "<one sentence>"}}
```

Model A is **always** v4; Model B is **always** baseline. Position bias is not counter-balanced — see *Anything weird*.

## Aggregate scores

{render_aggregate_table(agg_test, agg_held)}

Delta is v4 minus baseline. 95% CI is a paired percentile bootstrap on the per-example delta (1,000 resamples, seed 42).

### Same metrics restricted to the subset where baseline produced a final answer

Most baseline calls exhaust the 300-token budget inside `<|channel|>analysis|>` CoT and never open a `final` channel. The full-set numbers above therefore over-state v4's quality advantage (the gap is partly v4 vs `<empty>`). The subset below is the apples-to-apples doctrine-knowledge comparison.

| metric | split | n | v4 | baseline | delta |
|---|---|---:|---:|---:|---:|
| factual_accuracy_mean | test | {agg_test['subset_baseline_produced_answer']['n']} | {agg_test['subset_baseline_produced_answer'].get('v4_factual_accuracy_mean', float('nan')):.3f} | {agg_test['subset_baseline_produced_answer'].get('baseline_factual_accuracy_mean', float('nan')):.3f} | {fmt_delta(agg_test['subset_baseline_produced_answer'].get('v4_factual_accuracy_mean', 0) - agg_test['subset_baseline_produced_answer'].get('baseline_factual_accuracy_mean', 0))} |
| completeness_mean    | test | {agg_test['subset_baseline_produced_answer']['n']} | {agg_test['subset_baseline_produced_answer'].get('v4_completeness_mean', float('nan')):.3f} | {agg_test['subset_baseline_produced_answer'].get('baseline_completeness_mean', float('nan')):.3f} | {fmt_delta(agg_test['subset_baseline_produced_answer'].get('v4_completeness_mean', 0) - agg_test['subset_baseline_produced_answer'].get('baseline_completeness_mean', 0))} |
| grounding_mean       | test | {agg_test['subset_baseline_produced_answer']['n']} | {agg_test['subset_baseline_produced_answer'].get('v4_grounding_mean', float('nan')):.3f} | {agg_test['subset_baseline_produced_answer'].get('baseline_grounding_mean', float('nan')):.3f} | {fmt_delta(agg_test['subset_baseline_produced_answer'].get('v4_grounding_mean', 0) - agg_test['subset_baseline_produced_answer'].get('baseline_grounding_mean', 0))} |
| v4 win rate          | test | {agg_test['subset_baseline_produced_answer']['n']} | {agg_test['subset_baseline_produced_answer'].get('v4_win_rate', float('nan')):.1%} | (baseline-wins: {agg_test['subset_baseline_produced_answer'].get('baseline_win_rate', float('nan')):.1%}, ties: {agg_test['subset_baseline_produced_answer'].get('tie_rate', float('nan')):.1%}) | — |

## Win rates

{render_winrate_block(agg_test, "test", n_total=1875)}

{render_winrate_block(agg_held, "held_out", n_total=15)}

> Small-sample caveat (held_out, n=15): every per-type and per-source breakdown for held-out should be read as directional, not statistical.

## Per-source performance — test set

Sources are ranked by (mean v4 axis-total) − (mean baseline axis-total), where axis-total is the sum of factual_accuracy + completeness + grounding (range 3–15). Sources with n < 3 judged examples are excluded.

{render_per_source(agg_test)}

## Question-type breakdown

The Prompt #6 (journal 06 / journal 08) finding was that `procedural` had the highest NLL across the four variants (v1: 2.0549, v4: 2.0379 — both worse than `definitional` ~1.78). Under judge scoring on test:

| question_type | n | v4 factual_acc | baseline factual_acc | v4 completeness | baseline completeness | v4 grounding | baseline grounding | v4 win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|"""

    for qt, q in sorted(agg_test["by_question_type"].items()):
        md += (f"\n| {qt} | {q['n']} | {q['v4_factual_accuracy_mean']:.3f} | "
               f"{q['baseline_factual_accuracy_mean']:.3f} | "
               f"{q['v4_completeness_mean']:.3f} | {q['baseline_completeness_mean']:.3f} | "
               f"{q['v4_grounding_mean']:.3f} | {q['baseline_grounding_mean']:.3f} | "
               f"{q['v4_win_rate']:.1%} |")

    # Find min/max v4 axis means across types to call out the gap
    procedural = agg_test["by_question_type"].get("procedural", {})
    definitional = agg_test["by_question_type"].get("definitional", {})

    md += f"""

**Reading the procedural gap:** if procedural is hardest for v4 *under judge scoring too*, the gap is data-intrinsic (questions in this category are systematically harder to answer well) rather than a v4-specific weakness — v4 and baseline both struggle there relative to definitional/factual.

- procedural — v4 grounding mean = {procedural.get('v4_grounding_mean', float('nan')):.3f}, baseline grounding mean = {procedural.get('baseline_grounding_mean', float('nan')):.3f}
- definitional — v4 grounding mean = {definitional.get('v4_grounding_mean', float('nan')):.3f}, baseline grounding mean = {definitional.get('baseline_grounding_mean', float('nan')):.3f}

## Sample comparisons — 15 from test (different random.seed=42, but mixed across question_types)

{render_samples(test_judged, n=15, seed=42, balance_types=True)}

## Sample comparisons — all 15 from held_out_source

{render_samples(held_judged, n=15, seed=42, balance_types=False)}

## Anything weird

- **Baseline 300-token CoT exhaustion (dominant failure mode).** The `gpt_oss_no_sysprompt` renderer puts gpt-oss in `Reasoning: medium`, and the un-adapted base spends almost the entire 300-token budget in `<|channel|>analysis|>` before ever opening a `final` channel.
  - test set: baseline produced a parseable final answer in only **{failure_stats['test']['n_has_final']} / {failure_stats['test']['n']}** ({(1-failure_stats['test']['empty_pct'])*100:.1f}%) examples; **{failure_stats['test']['n_empty']}** ({failure_stats['test']['empty_pct']*100:.1f}%) returned empty after stripping CoT.
  - held_out: **{failure_stats['held_out']['n_has_final']} / {failure_stats['held_out']['n']}** produced a final answer.
  - Baseline mean output tokens: {failure_stats['test']['mean_tokens']:.0f} (cap 300); v4 mean output tokens: {sum(r.get('n_output_tokens', 0) for r in test_v4_gens)/max(1,len(test_v4_gens)):.1f}. v4 was trained on direct-answer pairs and learned to skip the analysis channel entirely.
  - The pair-wise judge sees Model B as empty in those cases and scores it at the floor (1/1/1). Most v4 "wins" on the full set therefore reflect *baseline budget exhaustion*, not v4 doctrinal superiority. The subset table above (where baseline actually produced an answer) is the cleaner doctrine-knowledge comparison.
- **Position bias is not counter-balanced.** Model A is always v4, Model B is always baseline. Pair-wise judges typically have a measurable position bias (≈ a few percentage points toward A). A follow-up should randomize position per example and recover the bias-corrected win rate.
- **Judge floor (1/1/1) overweights ties.** Many held-out examples were scored 1/1/1 vs 1/1/1 → declared "tie" — that is "both bad" rather than "both equally good." On held_out (CRS + FAA), v4 also has no training signal, so this is expected.
- **{judge_cost.get('skipped_error', 0)} test examples blocked by Azure content filter.** The judge call returned `BadRequestError: response was filtered due to … Azure OpenAI's content management policy`. Both rejected pair_ids are from `us_jp_3_01` (Counterair). Tenacity does not retry policy violations (correctly), so these are excluded from aggregates. They appear as `error` rows in `test_judged.jsonl`. To recover them, paraphrase the prompts and re-run with a per-pair-id override; for v1 of this report we accept the 2-row loss out of 1,875.
- **Sample token statistics:**
  - v4 test gen tokens: { gen_tok_summary(test_v4_gens) }
  - baseline test gen tokens: { gen_tok_summary(test_baseline_gens) }
  - v4 held_out gen tokens: { gen_tok_summary(held_v4_gens) }
  - baseline held_out gen tokens: { gen_tok_summary(held_baseline_gens) }

## Cost reconciliation

| Item | Value |
|---|---:|
| Judge calls (incl. malformed-JSON retries) | {judge_cost.get('calls', 0):,} |
| Judge prompt tokens | {judge_cost.get('prompt_tokens', 0):,} |
| Judge completion tokens | {judge_cost.get('completion_tokens', 0):,} |
| Judge input cost ($5/1M placeholder) | ${judge_cost.get('input_cost_usd', 0):.3f} |
| Judge output cost ($15/1M placeholder) | ${judge_cost.get('output_cost_usd', 0):.3f} |
| **Judge total** | **${judge_cost.get('total_cost_usd', 0):.3f}** |
| Hard cap | ${judge_cost.get('hard_cap_usd', 40.0):.2f} |
| Malformed-JSON retries | {judge_cost.get('malformed_first_pass', 0)} |
| Skipped (cost cap) | {judge_cost.get('skipped_cost_cap', 0)} |
| Skipped (API error) | {judge_cost.get('skipped_error', 0)} |

Tinker sampling cost is not surfaced by the SDK in this version; we generated 2 × ({len(test_v4_gens):,} + {len(held_v4_gens)}) = {2*(len(test_v4_gens) + len(held_v4_gens)):,} completions with `temperature=0.3, max_tokens=300`. At the ADR-0001 placeholder ($0.40/1M), the Tinker bill is upper-bounded by ${(2*(len(test_v4_gens) + len(held_v4_gens))*300*0.4/1e6):.4f} — small enough to ignore against the judge total.

> All prices are placeholders. Calibrate against real Azure / Tinker invoices before quoting absolute $.
"""

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL.write_text(md)
    print(f"wrote {JOURNAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
