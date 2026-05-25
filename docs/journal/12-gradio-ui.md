# Gradio UI for the Phase A v4 doctrine assistant

_Hand-written (not script-generated). Companion to `scripts/ui_app.py` and the
`deploy/hf_space/` bundle. Date: 2026-05-25._

## Setup

What was built this turn:

- **`scripts/ui_app.py`** — a self-contained Gradio **Blocks** app that serves
  the Phase A **v4** LoRA adapter (and the un-adapted baseline) through the
  Tinker `SamplingClient`. Four tabs: **Ask**, **Compare v4 vs Baseline**,
  **Examples**, **About**.
- **`deploy/hf_space/`** — a deployable Hugging Face Spaces bundle:
  `app.py` (verbatim copy of `ui_app.py`), `requirements.txt`, `README.md`
  (with the Spaces YAML metadata header), and `.env.example`.
- **Dependency added:** `gradio==6.14.0` (via `uv add gradio`; pulled in
  fastapi/starlette/uvicorn/typer and friends).

The app reuses the **exact** Phase A wiring so the UI shows what the eval
measured, not a different code path:

- System prompt verbatim from training (`docs/journal/06`), asserted equal at
  validation time:
  `"You are an expert assistant on Unmanned Aerial Vehicle (UAV) combat doctrine. Provide accurate, concise answers grounded in US, NATO, and allied doctrinal publications."`
- Renderer `gpt_oss_no_sysprompt` (system → `developer` role as `# Instructions`),
  tokenizer `openai/gpt-oss-20b`, final-channel-only parsing — identical to
  `scripts/eval_phase_a.py` (`extract_final_text` logic re-implemented locally
  so `app.py` needs no `finetune_slm` import).
- v4 sampler URL: `tinker://d79c263a-1e99-57b9-b09e-584fc222f3da:train:0/sampler_weights/final-step-00994`;
  baseline = same base with `model_path=None`.

## Architecture

**Why Gradio (not Streamlit / FastAPI).** The deliverable is a thin, demo-grade
front-end over two remote text generators with a side-by-side view — exactly
Gradio's sweet spot. `gr.Blocks` gives the custom two-column compare layout
without hand-writing HTML/JS, ships its own queue (rate-limit control), and is
the **native Hugging Face Spaces SDK** — the same file runs locally and on
Spaces with no rewrite. Streamlit would also work but isn't a first-class Spaces
citizen and reruns the whole script per interaction; raw FastAPI would mean
building the entire UI by hand for no benefit at this scale.

**Why the Tinker `SamplingClient` backend (not a local model load).** The base
is `gpt-oss-20b` — a 20B-parameter MoE. It does not fit on the Mac mini (or on a
free HF Spaces CPU box), and we deliberately never hold local weights (ADR
0001). Tinker already hosts the v4 sampler checkpoint, so the UI is a pure API
client: `build_generation_prompt → sample_async → parse_response`. This also
keeps the local and Spaces deployments identical — both are just API callers.

**Why side-by-side comparison is its own tab.** The v4-vs-baseline comparison is
the project's headline result, but it behaves very differently from a normal
"ask a question" flow: the un-adapted baseline almost always returns **no final
answer** (see below), so its panel needs explanatory scaffolding that would only
confuse the simple **Ask** path. Isolating it in its own tab lets **Ask** stay
clean (one box, one answer) while **Compare** carries the caveats, the dual
columns, and the "same prompt, same parameters" caption.

**Models are module-level singletons.** The tokenizer, renderer, stop sequences,
and both `SamplingClient`s are created once at import (wrapped in a try/except
that records `INIT_ERROR` and surfaces it in the UI rather than crashing the
process). Handlers are `async def` generators: they `yield` a "⏳ generating…"
status immediately, then yield the result. **Compare** runs both models
concurrently with `asyncio.gather`, which is what "stream in parallel" reduces
to here (see streaming note under *Known limitations*).

## Local run

Launched cleanly:

```
$ uv run python -u scripts/ui_app.py
* Running on local URL:  http://0.0.0.0:7860
```

- `http://localhost:7860` returns **HTTP 200**; `/config` reports a Gradio
  6.14.0 `blocks` app with analytics disabled.
- Live smoke through the real handler code (single asyncio loop, as Gradio uses):
  - **Ask** "What is UAV doctrine?" → v4 clean final answer, **26 tokens**, ~6–8s
    on the first (cold) call.
  - **Compare** "Define the rules of engagement in NATO doctrine." → v4 answered
    (33 tokens, ~1.5s warm); baseline returned the documented "no final answer"
    note.
  - Empty-question guard returns a prompt-to-type message (no API call).
- `gradio.launch()` returned without error; the server is left running for the
  human to interact with.

## HF Spaces readiness

`deploy/hf_space/` is staged but **not pushed** this turn (the model HF push is
owned by a parallel session). Contents:

- `app.py` — byte-identical copy of `scripts/ui_app.py`. (Copy, not symlink, so
  it survives a Spaces upload. **If `ui_app.py` changes, re-copy it.**)
- `requirements.txt` — `gradio`, `tinker`, `tinker-cookbook`, `transformers`,
  `tiktoken`, `python-dotenv`. (The prompt listed only the first two + dotenv;
  `tinker-cookbook` is required because `app.py` imports its renderer/tokenizer,
  and `transformers`/`tiktoken` back the gpt-oss tokenizer download.)
- `README.md` — Spaces YAML header (`sdk: gradio`, `sdk_version: 6.14.0`,
  `app_file: app.py`, `emoji: 🛰️`, `colorFrom/colorTo`, `license: apache-2.0`)
  + body with model-card and repo links and the research-artifact disclaimer.
- `.env.example` — documents `TINKER_API_KEY` (required) and `HF_TOKEN`.

What's still needed to actually push:

1. **Spaces Secret `TINKER_API_KEY`** — *Settings → Variables and secrets*. The
   backend `INIT_ERROR`s without it; one token serves all visitors (see below).
2. **Spaces Secret `HF_TOKEN`** — the `openai/gpt-oss-20b` tokenizer is gated on
   HF; the renderer needs it to build prompts. (Set locally here, so the gating
   is real.)
3. **Model dependency notice** — the Space depends on the Tinker sampler
   checkpoint staying alive at the URL hard-coded in `app.py`; it does *not*
   bundle the adapter. Coordinate with the model push so the model card link
   resolves before going public.

## Known limitations

- **Tinker rate limits → `queue(max_size=4)`** plus `default_concurrency_limit=2`.
  Bursts of visitors queue rather than fanning out into parallel Tinker calls.
- **20B inference latency.** Observed v4 generations: ~6–8s on a cold first call
  (connection warm-up), ~1.5–3s warm, for the ~30–50-token answers v4 emits.
  Baseline runs to the full token budget (~250–300 tokens) every time, so it is
  *slower despite usually returning nothing usable*.
- **Baseline returns no final answer (by design of the base model).** With
  `Reasoning: medium`, the un-adapted base spends its whole budget in the
  `analysis` CoT channel; per `docs/journal/09`, only ~8% of baseline answers
  reach a `final` channel at 300 tokens. The Compare panel detects the empty
  final channel and shows an explanatory note + the truncated reasoning trace,
  so users understand it isn't an app bug.
- **No token streaming.** Tinker SDK 0.22.0 exposes `sample` / `sample_async`
  but no streaming method, and partial harmony-format text can't be reliably
  split into the `final` channel mid-stream. We generate the full response then
  render. v4's short outputs keep this acceptable; the UI fakes responsiveness
  with an immediate "⏳ generating…" yield.
- **Cost per request** — *placeholder.* At the ADR-0001 placeholder of
  $0.40 / 1M tokens, a v4 answer (~50 tokens out + a few hundred prompt tokens)
  is a fraction of a cent; the baseline's full-budget CoT is the costlier side
  of a Compare. Calibrate against the real Tinker invoice before quoting numbers.
- **Single-tenant Tinker token for all visitors (not isolated).** Every request
  uses the one `TINKER_API_KEY` secret — no per-user auth, quota, or isolation.
  Fine for a gated demo; do **not** expose publicly without a rate limiter / auth
  in front, or the token's spend is open to the internet.

## Anything weird

- **stdout block-buffering hid the launch banner.** Backgrounded as
  `python scripts/ui_app.py`, the output file stayed empty for 60s even though
  the server was already serving HTTP 200 — Python block-buffers stdout when it
  is redirected to a file. Relaunched with `python -u` and the
  `* Running on local URL: http://0.0.0.0:7860` banner appeared immediately.
  (The server was healthy the whole time; only the log was delayed.)
- **Gradio 6 moved `theme=` from `Blocks()` to `launch()`.** Passing it to
  `Blocks()` raised a `UserWarning` and silently dropped the theme; moved
  `gr.themes.Soft()` to `demo.launch(...)`.
- **`load_dotenv()` is location-sensitive.** The throwaway pre-flight lived in
  `/tmp`, so `find_dotenv()` walked up from `/tmp` and missed the repo `.env`
  (→ `TINKER_API_KEY not set`). `scripts/ui_app.py` doesn't hit this — it lives
  in the repo, so `load_dotenv()` resolves the repo `.env`; on Spaces the key is
  an injected env var and `load_dotenv()` is a harmless no-op.
- **License field mismatch to flag.** The Spaces README YAML says
  `license: apache-2.0` (as specified in the prompt), but the repo code `LICENSE`
  is **MIT**. Reconcile before publishing the Space if consistency matters.
- **Commit scope.** A parallel session owns the in-progress HF push
  (`checkpoints/phase_a_qa/v4/adapter/` plus the untracked `export_lora_adapter.py`
  / `push_to_hf.py` / `verify_hf_adapter.py` and the `huggingface-hub`/`safetensors`
  /`peft` dependency additions). This turn's commit is **scoped to the UI work**
  and deliberately does **not** `git add -A`, to avoid bundling that session's
  untested files into a "Gradio UI" commit.
