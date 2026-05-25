# finetuneSLM — UAV Combat Doctrine Fine-Tuning

Fine-tune **OpenAI GPT-OSS-20B** on a public-domain UAV / UAS combat-doctrine
corpus using the **Tinker API**, in two phases:

- **Phase A — Doctrine Q&A.** A grounded question-answering model over
  joint / service / NATO / civil-aviation doctrine. Produces the
  `phase-a-qa` checkpoint.
- **Phase B — Tactical scenario reasoning.** Continues training from the
  Phase A checkpoint on tactical-scenario data (lost-link, C-UAS,
  airspace deconfliction, ROE-bounded decisions). Produces the
  `phase-b-scenario` checkpoint.

Shared data pipeline and shared eval harness with pluggable graders sit
under `src/finetune_slm/`.

## Try the model

- **🤗 Model card / LoRA adapter:** https://huggingface.co/meftun/gpt-oss-20b-uav-doctrine
- **🚀 Live demo (Gradio on HF Spaces):** https://huggingface.co/spaces/meftun/gpt-oss-20b-uav-doctrine-demo

The demo serves the Phase A **v4** adapter (and the un-adapted baseline, side by
side) through the Tinker API — nothing runs on the Space's own hardware. Research
artifact: verify every output against the source doctrine; not for operational use.

## Stack

- **Base model:** OpenAI GPT-OSS-20B (MoE, future-safe — not on Tinker's
  June 12 retirement list)
- **Training platform:** [Tinker](https://thinkingmachines.ai/tinker) —
  managed LoRA fine-tuning, no local GPU required
- **Training recipe:** LoRA via [`tinker-cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook)
- **Tokenizer:** `openai/gpt-oss-20b` (HF) with `tiktoken` `o200k_base`
  as the gate-free fallback
- **Package manager:** `uv` (Python 3.11)
- **Tracking:** Weights & Biases

## Corpus

`data/raw/` already contains the public-domain doctrine corpus —
US joint / service publications, NATO AJPs, UK JDPs, EASA / ICAO / FAA
civil-aviation lost-link anchors, plus DoDD 3000.09 and the DoD
Dictionary. Provenance metadata lives next to each PDF in `manifest.json`.
The ingest pipeline that produced this lives at `src/ingest/`.

`data/interim/` and `data/processed/` hold derived artifacts and are
git-ignored — regenerate with the pipeline scripts in `scripts/`.

## Setup

```bash
# 1. Install dependencies (creates .venv and uv.lock)
uv sync

# 2. Configure credentials
cp .env.example .env
# then fill in TINKER_API_KEY (required)
# and optionally WANDB_API_KEY (for training run tracking)
# and HF_TOKEN (for the openai/gpt-oss-20b tokenizer, if gated)

# 3. Inspect the corpus
uv run python scripts/inventory_data.py
```

## Documentation

- `docs/decisions/` — Architecture Decision Records (ADRs) for the
  load-bearing choices: base model, platform, phased roadmap.
- `docs/journal/` — Operational journal: dated inventories, run logs,
  cost reconciliation, eval reports.

## Layout

```
configs/                     # Training + data configs (YAML / TOML)
  data/                      # Dataset-building configs
  tinker/
    phase_a_qa/              # Phase A training configs
    phase_b_scenario/        # Phase B training configs
data/
  raw/                       # Provenance corpus (tracked)
  interim/                   # Derived intermediates (ignored)
  processed/
    qa/                      # Phase A datasets
    scenario/                # Phase B datasets
src/
  ingest/                    # Pre-existing corpus ingest pipeline
  finetune_slm/
    data/                    # Doc loaders, chunkers, dataset builders
    training/                # Tinker training loops, LoRA adapters
    eval/                    # Eval harness + pluggable graders
    utils/                   # Shared helpers
scripts/                     # CLI entrypoints (inventory, build, train)
checkpoints/                 # Saved Tinker checkpoints (ignored)
outputs/                     # Eval reports, generations (ignored)
docs/
  decisions/                 # ADRs
  journal/                   # Dated operational notes
```
