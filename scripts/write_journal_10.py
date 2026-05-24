"""Render docs/journal/10-position-bias-control.md from the position-bias
re-judge artifacts.

Idempotent: overwrites the journal each run; reads from
  outputs/phase_a_eval/position_bias_judged.jsonl
  outputs/phase_a_eval/position_bias_cost.json
  outputs/phase_a_eval/aggregates.json   (for the original Prompt #7 numbers)
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "phase_a_eval"
JOURNAL = ROOT / "docs" / "journal" / "10-position-bias-control.md"


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


def _bootstrap_ci_rate(indicators: list[int], n_resamples: int = 1000, seed: int = 4242) -> tuple[float, float]:
    if not indicators:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(indicators)
    samples = []
    for _ in range(n_resamples):
        s = sum(indicators[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    return samples[int(0.025 * n_resamples)], samples[int(0.975 * n_resamples)]


def compute_position_a_rate(rows: list[dict]) -> tuple[float, int, int]:
    """Across both calls per example, what % of non-tie decisions picked position A?"""
    a_count = b_count = 0
    for r in rows:
        for key in ("original_winner_position", "swapped_winner_position"):
            w = r.get(key)
            if w == "A":
                a_count += 1
            elif w == "B":
                b_count += 1
    non_tie = a_count + b_count
    if non_tie == 0:
        return float("nan"), 0, 0
    return a_count / non_tie, a_count, b_count


def compute_agreement(rows: list[dict]) -> tuple[float, int, int]:
    n = sum(1 for r in rows if "agreement" in r)
    n_agree = sum(1 for r in rows if r.get("agreement"))
    if n == 0:
        return float("nan"), 0, 0
    return n_agree / n, n_agree, n


def bias_corrected_winner(r: dict) -> str:
    """The conservative "true winner": agreement → that model; disagreement → tie."""
    if "agreement" not in r:
        return "skip"
    if r["agreement"]:
        return r["original_winner_model"]
    return "tie"


def winrate_block(rows: list[dict]) -> dict:
    """v4 / baseline / tie rates from the bias-corrected winner."""
    winners = [bias_corrected_winner(r) for r in rows if "agreement" in r]
    n = len(winners)
    if n == 0:
        return {"n": 0}
    v4 = sum(1 for w in winners if w == "v4")
    bl = sum(1 for w in winners if w == "baseline")
    tie = sum(1 for w in winners if w == "tie")
    v4_ind = [1 if w == "v4" else 0 for w in winners]
    bl_ind = [1 if w == "baseline" else 0 for w in winners]
    return {
        "n": n,
        "v4": v4 / n,
        "baseline": bl / n,
        "tie": tie / n,
        "v4_ci_95": _bootstrap_ci_rate(v4_ind),
        "baseline_ci_95": _bootstrap_ci_rate(bl_ind),
    }


def per_type_winrate(rows: list[dict]) -> dict[str, dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    out = {}
    for qt, rs in sorted(by_type.items()):
        out[qt] = winrate_block(rs)
    return out


def per_type_position_a(rows: list[dict]) -> dict[str, dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    out = {}
    for qt, rs in sorted(by_type.items()):
        pa, a, b = compute_position_a_rate(rs)
        out[qt] = {"position_a_rate": pa, "a_picks": a, "b_picks": b, "n_examples": len(rs)}
    return out


def render_disagreement_examples(rows: list[dict], n: int = 5, seed: int = 4242) -> str:
    disagreements = [r for r in rows if "agreement" in r and not r["agreement"]]
    if not disagreements:
        return "_(no disagreements found — both calls picked the same model on every example)_"
    rng = random.Random(seed)
    picks = rng.sample(disagreements, min(n, len(disagreements)))
    blocks = []
    for i, r in enumerate(picks, 1):
        blocks.append(f"""#### Disagreement {i} — `{r['pair_id']}` — `{r['question_type']}` — `{r['source_filename']}`

**Q:** {r['question']}

**Reference:**

> {r['reference']}

**v4:**

> {r['v4_generation'] or '_(empty)_'}

**baseline:**

> {r['baseline_generation'] or '_(empty)_'}

**Original judge (v4=A, baseline=B):** winner=**{r['original_winner_position']}** → model `{r['original_winner_model']}`. _Rationale: {r['original_scores'].get('rationale', '')}_

**Swapped judge (baseline=A, v4=B):** winner=**{r['swapped_winner_position']}** → model `{r['swapped_winner_model']}`. _Rationale: {r['swapped_scores'].get('rationale', '')}_
""")
    return "\n".join(blocks)


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(EVAL_DIR / "position_bias_judged.jsonl")
    if not rows:
        print(f"STOP: {EVAL_DIR}/position_bias_judged.jsonl missing/empty — run "
              f"scripts/position_bias_check.py first.", file=sys.stderr)
        return 1

    cost_path = EVAL_DIR / "position_bias_cost.json"
    cost = json.loads(cost_path.read_text()) if cost_path.exists() else {}

    # Reference data: Prompt #7 aggregates (original win rates).
    agg_path = EVAL_DIR / "aggregates.json"
    aggregates = json.loads(agg_path.read_text()) if agg_path.exists() else {}
    orig_test = aggregates.get("test", {}).get("win_rate", {})
    orig_per_type = aggregates.get("test", {}).get("by_question_type", {})

    # Compute everything.
    pos_a_rate, pos_a_count, pos_b_count = compute_position_a_rate(rows)
    agreement_rate, n_agree, n_total = compute_agreement(rows)
    bias_corrected = winrate_block(rows)
    by_qt = per_type_winrate(rows)
    by_qt_pos = per_type_position_a(rows)

    actual_per_type = cost.get("actual_per_question_type", {})

    # ---- Build markdown ----

    md = f"""# Position-bias control — Phase A pair-wise judge

_Generated by `scripts/write_journal_10.py` from `outputs/phase_a_eval/position_bias_judged.jsonl`. Re-running overwrites this file. Raw swap judgments live next to it._

## Setup

| Item | Value |
|---|---|
| Original eval | `docs/journal/09-phase-a-eval-report.md` (v4 always in position A) |
| Sample size | {len(rows)} (target 300; stratified ~75 / question_type) |
| Sampling seed | {cost.get('seed', '?')} |
| Judge | GPT-5.4 via Azure (same deployment as Prompt #7) |
| Concurrency | 5 |
| Hard cost cap | ${cost.get('hard_cap_usd', 5.0):.2f} |
| Rubric | identical to Prompt #7 (system prompt + JSON schema); only positions swap |

Per-type actual counts:

| question_type | sampled | total available |
|---|---:|---:|
"""
    for qt in sorted(actual_per_type.keys()):
        md += f"| {qt} | {actual_per_type[qt]} | {orig_per_type.get(qt, {}).get('n', '?')} |\n"

    md += f"""

## Position-A preference rate

Across **all 600 judge decisions** (300 original + 300 swapped), counting only non-tie outcomes:

| metric | value |
|---|---:|
| Position-A picks | {pos_a_count} |
| Position-B picks | {pos_b_count} |
| Total non-tie decisions | {pos_a_count + pos_b_count} |
| **Position-A preference rate** | **{pos_a_rate:.1%}** |

Interpretation:
- ~50%   → judge has no positional bias on this rubric; original Prompt #7 numbers stand.
- > 55% → measurable bias toward position A; original v4 win rate is inflated.
- < 45% → bias toward position B; original v4 win rate is deflated.

**Verdict:** {"no meaningful bias" if 0.45 <= pos_a_rate <= 0.55 else "bias toward position A" if pos_a_rate > 0.55 else "bias toward position B"} ({pos_a_rate - 0.5:+.1%} from 50%).

Per question_type:

| question_type | sampled | A picks | B picks | position-A rate |
|---|---:|---:|---:|---:|
"""
    for qt, d in sorted(by_qt_pos.items()):
        md += (f"| {qt} | {d['n_examples']} | {d['a_picks']} | {d['b_picks']} | "
               f"{d['position_a_rate']:.1%} |\n")

    md += f"""

## Agreement rate

Of the {n_total} examples that received both an original and a swap judgment, how many had both calls agree on which **MODEL** won (v4 / baseline / tie)?

| metric | value |
|---|---:|
| Examples agreeing on the same model | {n_agree} / {n_total} |
| **Model-agreement rate** | **{agreement_rate:.1%}** |

Interpretation:
- High agreement (>80%) → judge is reliable; original win rate is trustworthy.
- Low agreement (<70%) → judge is noisy AND/OR biased.

## Bias-corrected win rate

For each example, the "true winner" is the model both calls agreed on; disagreements collapse to tie (conservative). Win rates from this 300-row sample:

| metric | original (Prompt #7, n=1873) | bias-corrected (this turn, n={bias_corrected['n']}) | delta |
|---|---:|---:|---:|
| v4 wins | {orig_test.get('v4', 0):.1%} | {bias_corrected['v4']:.1%} | {(bias_corrected['v4'] - orig_test.get('v4', 0)):+.1%} |
| baseline wins | {orig_test.get('baseline', 0):.1%} | {bias_corrected['baseline']:.1%} | {(bias_corrected['baseline'] - orig_test.get('baseline', 0)):+.1%} |
| tie | {orig_test.get('tie', 0):.1%} | {bias_corrected['tie']:.1%} | {(bias_corrected['tie'] - orig_test.get('tie', 0)):+.1%} |
| 95% CI on v4 wins | {orig_test.get('v4_ci_95', [0,0])[0]:.1%} – {orig_test.get('v4_ci_95', [0,0])[1]:.1%} | {bias_corrected['v4_ci_95'][0]:.1%} – {bias_corrected['v4_ci_95'][1]:.1%} | — |

Per question_type (bias-corrected):

| question_type | n | v4 wins | baseline wins | tie | 95% CI on v4 |
|---|---:|---:|---:|---:|---|
"""
    for qt, d in sorted(by_qt.items()):
        md += (f"| {qt} | {d['n']} | {d['v4']:.1%} | {d['baseline']:.1%} | "
               f"{d['tie']:.1%} | {d['v4_ci_95'][0]:.1%} – {d['v4_ci_95'][1]:.1%} |\n")

    md += f"""

## Five interesting examples (original ≠ swapped)

{render_disagreement_examples(rows, n=5, seed=4242)}

## What this means for the blog post

"""

    # Generate the narrative based on actual numbers
    delta_v4 = bias_corrected['v4'] - orig_test.get('v4', 0)
    if abs(delta_v4) < 0.03:
        verdict = (
            f"The bias-corrected v4 win rate ({bias_corrected['v4']:.1%}) is within {abs(delta_v4)*100:.1f}pp of the original ({orig_test.get('v4', 0):.1%}). "
            f"Position bias is **not** materially inflating the headline. Lead the blog post with the original number; "
            f"mention the bias-corrected check as a footnote: \"randomized-position re-judging on a 300-example stratified sample reproduced the win rate within {abs(delta_v4)*100:.1f}pp.\""
        )
    elif delta_v4 < 0:
        verdict = (
            f"The bias-corrected v4 win rate ({bias_corrected['v4']:.1%}) is **{abs(delta_v4)*100:.1f}pp lower** than the original ({orig_test.get('v4', 0):.1%}). "
            f"Position bias **is** inflating the headline. Lead the blog post with the bias-corrected number ({bias_corrected['v4']:.1%}, "
            f"95% CI {bias_corrected['v4_ci_95'][0]:.1%}–{bias_corrected['v4_ci_95'][1]:.1%}); flag the original as biased."
        )
    else:
        verdict = (
            f"The bias-corrected v4 win rate ({bias_corrected['v4']:.1%}) is **higher** than the original ({orig_test.get('v4', 0):.1%}) by {delta_v4*100:.1f}pp — "
            f"unusual; either Prompt #7 was conservative or the disagreements pattern in our favor. Lead with the original number "
            f"({orig_test.get('v4', 0):.1%}) as the safer claim; mention bias check confirmed."
        )
    md += verdict + "\n\n"

    md += "## Anything weird\n\n"

    # Programmatic notes
    if pos_a_count + pos_b_count == 0:
        md += "- Every judge call returned `tie` — likely a rubric or data problem; review.\n"
    else:
        skew = pos_a_rate - 0.5
        if abs(skew) < 0.02:
            md += f"- The position-A preference rate landed at {pos_a_rate:.1%} (within ±2pp of 50%), so on the strict 3-axis + winner rubric used here, GPT-5.4 is essentially position-symmetric. Position bias on rubric-style judging may be smaller than the chat-arena literature suggests.\n"
        elif abs(skew) < 0.05:
            md += f"- Mild position-A skew ({pos_a_rate:.1%}, +{skew*100:.1f}pp). Within normal LLM-judge bias literature for pair-wise comparisons.\n"
        else:
            md += f"- Strong position-A skew ({pos_a_rate:.1%}, {skew*100:+.1f}pp). The judge is systematically over-rewarding whoever is in position A.\n"

    if agreement_rate >= 0.85:
        md += f"- High model-agreement ({agreement_rate:.1%}) confirms the judge is internally consistent on this rubric.\n"
    elif agreement_rate < 0.70:
        md += f"- Low model-agreement ({agreement_rate:.1%}) — the judge is noisy. Bias-corrected numbers are noisier than the original n=1873; the {orig_test.get('v4_ci_95', [0,0])[1] - orig_test.get('v4_ci_95', [0,0])[0]:.1%}-wide original CI is still the cleaner estimate of v4 vs baseline.\n"

    # Tie inflation note
    if bias_corrected.get('tie', 0) > orig_test.get('tie', 0) + 0.05:
        md += (f"- Bias-corrected ties ({bias_corrected['tie']:.1%}) are inflated vs the original ({orig_test.get('tie', 0):.1%}) "
               f"because disagreements collapse to tie by construction. That's the price of a conservative bias correction.\n")

    # Disagreement-by-type pattern
    by_type_disagree: dict[str, int] = defaultdict(int)
    by_type_n: dict[str, int] = defaultdict(int)
    for r in rows:
        if "agreement" not in r:
            continue
        by_type_n[r["question_type"]] += 1
        if not r["agreement"]:
            by_type_disagree[r["question_type"]] += 1
    disagree_lines = []
    for qt in sorted(by_type_n.keys()):
        if by_type_n[qt]:
            disagree_lines.append(
                f"{qt} {by_type_disagree[qt]}/{by_type_n[qt]} ({by_type_disagree[qt]/by_type_n[qt]:.0%})"
            )
    md += f"- Disagreement by question_type: {', '.join(disagree_lines)}.\n"

    # Cost
    md += f"""
## Cost reconciliation

| Item | Value |
|---|---:|
| Judge calls (incl. retries) | {cost.get('calls', 0)} |
| Prompt tokens | {cost.get('prompt_tokens', 0):,} |
| Completion tokens | {cost.get('completion_tokens', 0):,} |
| Input cost ($5/1M placeholder) | ${cost.get('input_cost_usd', 0):.3f} |
| Output cost ($15/1M placeholder) | ${cost.get('output_cost_usd', 0):.3f} |
| **Total** | **${cost.get('total_cost_usd', 0):.3f}** |
| Hard cap | ${cost.get('hard_cap_usd', 5.0):.2f} |
| Malformed-JSON retries | {cost.get('malformed_first_pass', 0)} |
| Skipped (cost cap) | {cost.get('skipped_cost_cap', 0)} |
| Skipped (API error) | {cost.get('skipped_error', 0)} |

> All prices are placeholders. Calibrate against the real Azure invoice before quoting absolute $.
"""

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL.write_text(md)
    print(f"wrote {JOURNAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
