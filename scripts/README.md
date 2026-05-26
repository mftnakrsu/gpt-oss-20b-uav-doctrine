# Scripts

CLI entrypoints for the project. All run via `uv run python scripts/<name>.py` and load credentials from `.env` (see `.env.example`).

## Reproduction run order

Each step consumes the previous step's outputs. (The committed `data/processed/qa/` splits let you skip steps 1–7 and train directly.)

1. **inventory_data.py** — File census + tokenization stats over `data/raw/`. Validates the corpus.
2. **expand_corpus.py** — Gathers additional public doctrine PDFs from .gov/.mil + academic/think-tank sources, recording provenance per source.
3. **prepare_corpus.py** — Text extraction, cleaning (headers/footers, hyphenation, ligatures), and chunking → `data/interim/chunks.jsonl`.
4. **filter_chunks.py** — Multi-signal quality filter (TOC, bibliography, low-content, page-ref, cover-metadata detectors).
5. **pilot_qa.py** — 10-chunk Q&A-generation pilot to validate the prompt and cost before the full run.
6. **full_qa.py** — Full-corpus Q&A generation via GPT-5.4 (5 grounded pairs per passing chunk).
7. **build_splits.py** — ChatML rendering + chunk-level group-aware train/val/test + held-out-source splits.
8. **train_phase_a.py** — Tinker LoRA SFT on `gpt-oss-20b` (includes the v1–v4 ablation harness).
9. **eval_phase_a.py** — Full evaluation: v4 vs the un-adapted baseline, GPT-5.4 as a pair-wise judge.
10. **position_bias_check.py** — Randomized-position re-judging to bias-correct the win rate.
11. **export_lora_adapter.py** — Download the v4 adapter from Tinker and convert it to PEFT format.
12. **push_to_hf.py** — Publish the adapter + model card to HuggingFace.
13. **ui_app.py** — Gradio UI (runs locally or on HF Spaces; serves v4 + baseline via Tinker).

## Utility scripts

- **inspect_qa_pairs.py** — Read-only verifier / browser for the generated Q&A dataset.
- **preflight_azure.py** — Connection tests for the Azure GPT-5.4 + embedding endpoints.
- **verify_hf_adapter.py** — Downloads the HF-published adapter and structure-validates it.
- **write_journal_07.py … write_journal_10.py** — Regenerate journals 07–10 from the run/eval artifacts in `outputs/`.
- **make_charts.py** — Regenerate the README result charts (`docs/images/*.png`) from the journal-verified numbers.

## Environment

All scripts run under `uv` (Python 3.11). See the repo [README](../README.md) — "Quickstart — reproduce training" — for the full sequence and the required API keys.
