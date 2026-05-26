"""Generate the result charts embedded in the README (docs/images/).

All numbers are the final, journal-verified figures (docs/journal/07, 09, 10).
The charts are produced from the data, not hand-drawn, so they stay accurate.

Run: uv run python scripts/make_charts.py   (matplotlib is in the dev extra)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "images"
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
V4, BASE, TIE, ACCENT, RED = "#1f77b4", "#9e9e9e", "#dcdcdc", "#2ca02c", "#e74c3c"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (IMG / name).relative_to(ROOT))


def off(xs, d):
    return [x + d for x in xs]


# 1. Win rates (stacked) — overall + per question type  [journal 09]
cats = ["overall", "conceptual", "definitional", "procedural", "factual"]
v4w = [82.6, 90.0, 86.4, 81.5, 73.2]
tiew = [16.1, 8.5, 11.1, 18.0, 25.8]
basew = [1.3, 1.5, 2.5, 0.4, 1.0]
x = list(range(len(cats)))
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.bar(x, v4w, color=V4, label="v4 wins")
ax.bar(x, tiew, bottom=v4w, color=TIE, label="tie")
ax.bar(x, basew, bottom=[a + b for a, b in zip(v4w, tiew)], color=RED, label="baseline wins")
for i, v in enumerate(v4w):
    ax.text(i, v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("share of judged pairs (%)"); ax.set_ylim(0, 100)
ax.set_title("v4 vs un-adapted baseline — GPT-5.4 pair-wise judge (test, n=1,873)")
ax.legend(loc="upper right", framealpha=0.9)
save(fig, "win_rates.png")

# 2. Mean judge scores by axis  [journal 09]
axes_lbl = ["factual\naccuracy", "completeness", "grounding"]
v4s = [2.719, 2.340, 2.613]
bases = [1.081, 1.061, 1.054]
x = list(range(len(axes_lbl))); w = 0.36
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(off(x, -w / 2), v4s, w, color=V4, label="v4")
ax.bar(off(x, w / 2), bases, w, color=BASE, label="baseline")
for i, (a, b) in enumerate(zip(v4s, bases)):
    ax.text(i - w / 2, a + 0.06, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w / 2, b + 0.06, f"{b:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(axes_lbl)
ax.set_ylabel("mean judge score (1–5)"); ax.set_ylim(0, 5)
ax.set_title("Mean judge scores by axis (test set)")
ax.legend()
save(fig, "score_axes.png")

# 3. Validation NLL over training  [journal 07, baseline run]
steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
         550, 600, 650, 700, 750, 800, 850, 900, 950, 994]
val = [2.2539, 2.1090, 2.0357, 1.9682, 1.9230, 1.8792, 1.8409, 1.8350, 1.8022, 1.7656,
       1.7020, 1.6512, 1.6066, 1.5727, 1.5492, 1.5214, 1.5030, 1.4866, 1.4770, 1.4706]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, val, marker="o", ms=3, color=V4)
ax.set_xlabel("training step"); ax.set_ylabel("validation NLL (training-time)")
ax.set_title("Validation NLL over Phase A SFT (2 epochs, 994 steps)")
save(fig, "nll_curve.png")

# 4. Mean answer length  [journal 09]
fig, ax = plt.subplots(figsize=(5.6, 4.2))
bars = ax.bar(["v4", "baseline"], [49.4, 297.5], color=[V4, BASE])
for b, v in zip(bars, [49.4, 297.5]):
    ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}", ha="center", fontweight="bold")
ax.set_ylabel("mean output tokens"); ax.set_ylim(0, 335)
ax.set_title("Mean answer length (300-token budget)")
ax.text(0.5, -0.20, "v4 answers directly; the base model burns its budget on hidden CoT",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#555")
save(fig, "output_length.png")

# 5. Cost by phase  [journals 05a/05b/07/09/10]
labels = ["Q&A gen\n(GPT-5.4)", "Eval judge\n(GPT-5.4)", "Bias\nre-judge", "Pilot", "LoRA train\n(Tinker*)"]
costs = [47.81, 5.64, 0.90, 0.13, 1.59]
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(labels, costs, color=[V4, V4, V4, V4, ACCENT])
for b, v in zip(bars, costs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"${v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("USD"); ax.set_ylim(0, 53)
ax.set_title("Cost by phase  (~$56 single run;  *Tinker = ADR-0001 placeholder rate)")
save(fig, "cost.png")

print("all charts written to", IMG.relative_to(ROOT))
