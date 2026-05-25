# Documentation

Start here, then read the build journal in order for the full story — or jump to a specific entry.

## Decision records (ADRs)

Load-bearing technical decisions and the reasoning behind them, in [`decisions/`](decisions/).

- [0001 — Base model and platform](decisions/0001-base-model-and-platform.md) — why `gpt-oss-20b` + Tinker (future-safe MoE, no local GPU required).
- [0002 — Phased roadmap](decisions/0002-phased-roadmap-a-then-b.md) — Phase A (doctrine Q&A) first, Phase B (scenario reasoning) second, sharing one base model and eval harness.

## Build journal

A step-by-step log of how the model was built — each entry is roughly one CLI session, in [`journal/`](journal/).

| # | Topic | Key finding |
|---|-------|-------------|
| [01](journal/01-data-inventory.md) | Initial data inventory | 2.37M tokens across 30 PDFs |
| [02](journal/02-data-expansion.md) | Corpus expansion | +28 docs → 58 total |
| [03](journal/03-cleaning-and-chunking.md) | Cleaning + chunking pipeline | 4,051 chunks @ 1,200 tok; 95.2% retained |
| [04](journal/04-chunk-quality-filter.md) | Chunk quality filter | 92.5% pass; dropped TOC/bibliography/low-content |
| [05a](journal/05a-qa-pilot.md) | Q&A generation pilot | $0.13 / 50 pairs; methodology validated |
| [05b](journal/05b-qa-full-run.md) | Full Q&A generation | 18,730 pairs; $47.81 actual |
| [06](journal/06-chatml-formatting.md) | ChatML formatting + splits | chunk-level group-aware split (no leakage) |
| [07](journal/07-tinker-sft-phase-a.md) | Phase A SFT (v1 baseline) | first Tinker LoRA run on gpt-oss-20b |
| [08](journal/08-phase-a-ablation-sweep.md) | 4-variant ablation sweep | v4 (lr 2e-4) wins; three methodology findings |
| [09](journal/09-phase-a-eval-report.md) | Full eval with GPT-5.4 judge | 82.6% win rate; baseline budget-exhaustion finding |
| [10](journal/10-position-bias-control.md) | Position-bias control | bias-corrected 80.7%; rubric judging is position-symmetric |
| [11](journal/11-hf-push.md) | HuggingFace push | public model + model card |
| [12](journal/12-gradio-ui.md) | Gradio UI | local + HF Spaces bundle |
| [13](journal/13-hf-spaces-deploy.md) | HF Spaces deployment | live demo, end-to-end verified |
| [14](journal/14-github-public-push.md) | GitHub public release | secret audit clean; repo renamed; README overhaul + examples |

Read top-to-bottom for the full narrative, or jump to a specific journal if you have a question about that step.
