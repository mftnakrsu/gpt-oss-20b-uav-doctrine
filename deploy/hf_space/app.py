"""Gradio UI for the Phase A v4 UAV-doctrine assistant.

Interactive front-end for the fine-tuned `gpt-oss-20b` LoRA adapter (Phase A,
variant v4) served through the **Tinker** `SamplingClient`. Nothing runs
locally on the host — the 20B MoE base is far too large for a Mac mini — so
both models are sampled remotely via Tinker.

Two models are wired up:
  * **v4**  — the LoRA adapter (sampler URL below).
  * **baseline** — the un-adapted `openai/gpt-oss-20b` (`model_path=None`).

The rendering + sampling path is identical to `scripts/eval_phase_a.py`: the
`gpt_oss_no_sysprompt` renderer (system prompt → `developer` role as
`# Instructions`), the `openai/gpt-oss-20b` tokenizer, and final-channel-only
parsing. The exact training/eval SYSTEM_PROMPT is reused verbatim.

This file is intentionally self-contained (no `finetune_slm` imports) so it can
be copied verbatim to `deploy/hf_space/app.py` and run on Hugging Face Spaces
with only `gradio`, `tinker`, `tinker-cookbook`, and `python-dotenv` installed.

Run locally:
    uv run python scripts/ui_app.py
then open http://localhost:7860.
"""

from __future__ import annotations

import os

# Disable Gradio telemetry before importing gradio (keeps startup network-free).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import asyncio  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Resolve TINKER_API_KEY from a local .env when present (no-op on HF Spaces,
# where the key is injected as an environment secret instead).
load_dotenv()

import tinker  # noqa: E402
from tinker import SamplingParams  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

# =============================================================================
# Constants — keep in lock-step with scripts/eval_phase_a.py
# =============================================================================

V4_SAMPLER_URL = (
    "tinker://d79c263a-1e99-57b9-b09e-584fc222f3da:train:0/sampler_weights/final-step-00994"
)
BASE_MODEL = "openai/gpt-oss-20b"
RENDERER_NAME = "gpt_oss_no_sysprompt"

# EXACT system prompt used during Phase A training (docs/journal/06). Do not
# reword — the adapter was trained against this precise string, placed in the
# gpt-oss `developer` role by the gpt_oss_no_sysprompt renderer.
SYSTEM_PROMPT = (
    "You are an expert assistant on Unmanned Aerial Vehicle (UAV) combat "
    "doctrine. Provide accurate, concise answers grounded in US, NATO, and "
    "allied doctrinal publications."
)

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 300

HF_MODEL_URL = "https://huggingface.co/meftun/gpt-oss-20b-uav-doctrine"
GITHUB_REPO_URL = "https://github.com/mftnakrsu/finetuneSLM"

# Trace shown to users is capped so a runaway baseline CoT can't flood the panel.
_REASONING_PREVIEW_CHARS = 1500

# =============================================================================
# Harmony-format response parsing (mirrors eval_phase_a.extract_final_text)
# =============================================================================

_FINAL_BLOCK_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|\Z)", re.DOTALL
)
_ANALYSIS_BLOCK_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|<\|channel\|>|\Z)",
    re.DOTALL,
)


def parse_generation(content) -> dict:
    """Split a parsed gpt-oss response into final-channel text + reasoning.

    Two parser shapes, exactly as in the eval harness:
      * **list of typed parts** (clean parse): keep ``type=="text"`` parts as the
        final answer and ``type=="thinking"`` parts as the reasoning trace.
      * **raw string** (``stop_reason='malformed'``): the baseline routinely
        burns its whole token budget inside ``<|channel|>analysis|>`` and never
        opens a ``final`` channel, so the renderer hands back un-segmented text.
        Regex out the final block if present; otherwise treat everything as the
        (unfinished) reasoning trace.
    """
    final_text = ""
    reasoning_text = ""
    if isinstance(content, list):
        final_text = "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ).strip()
        reasoning_text = "".join(
            p.get("thinking", "") for p in content
            if isinstance(p, dict) and p.get("type") == "thinking"
        ).strip()
    elif isinstance(content, str):
        m = _FINAL_BLOCK_RE.search(content)
        final_text = m.group(1).strip() if m else ""
        a = _ANALYSIS_BLOCK_RE.search(content)
        if a:
            reasoning_text = a.group(1).strip()
        elif not final_text:
            # No channel markers at all — show the raw text as the trace.
            reasoning_text = content.strip()
    else:
        final_text = str(content).strip()
    return {
        "final_text": final_text,
        "reasoning_text": reasoning_text,
        "had_final": bool(final_text),
    }


# =============================================================================
# Tinker backend — module-level singletons (created once at startup)
# =============================================================================

INIT_ERROR: str | None = None
_tokenizer = None
_renderer = None
_stop_sequences = None
v4_client = None
baseline_client = None

try:
    _tokenizer = get_tokenizer(BASE_MODEL)
    _renderer = get_renderer(RENDERER_NAME, _tokenizer, model_name=BASE_MODEL)
    _stop_sequences = _renderer.get_stop_sequences()
    _service = tinker.ServiceClient()
    # v4 = LoRA adapter; baseline = un-adapted base (model_path=None).
    v4_client = _service.create_sampling_client(
        model_path=V4_SAMPLER_URL, base_model=BASE_MODEL
    )
    baseline_client = _service.create_sampling_client(
        model_path=None, base_model=BASE_MODEL
    )
except Exception as exc:  # noqa: BLE001 — surface any startup failure in the UI
    INIT_ERROR = f"{type(exc).__name__}: {exc}"


def _friendly_error(exc: Exception) -> str:
    """Short, user-facing error string. Never leaks credentials (Tinker errors
    carry no key material, and we only echo the type + message)."""
    msg = str(exc).strip() or repr(exc)
    if len(msg) > 300:
        msg = msg[:300] + "…"
    return f"{type(exc).__name__}: {msg}"


async def _generate(client, question: str, temperature: float, max_tokens: int) -> dict:
    """Render the prompt and sample one completion from `client`.

    Returns parse_generation(...) augmented with `n_tokens`, `stop_reason`,
    and `wall_seconds`. Raises on transport/sampling errors (caller handles).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = _renderer.build_generation_prompt(messages)
    params = SamplingParams(
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        stop=_stop_sequences,
    )
    t0 = time.monotonic()
    resp = await client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
    dt = time.monotonic() - t0
    tokens = resp.sequences[0].tokens
    parsed, stop_reason = _renderer.parse_response(tokens)
    result = parse_generation(parsed.get("content"))
    result["n_tokens"] = len(tokens)
    result["stop_reason"] = stop_reason
    result["wall_seconds"] = dt
    return result


def _meta_line(result: dict, label: str) -> str:
    return (
        f"<sub>{label} · {result['n_tokens']} tokens · "
        f"{result['wall_seconds']:.1f}s · stop=`{result['stop_reason']}`</sub>"
    )


_NO_FINAL_NOTE = (
    "⚠️ **No final answer was emitted.** The model spent its entire "
    "`max_tokens` budget inside its reasoning (`analysis`) channel before "
    "opening a `final` channel. For the un-adapted baseline this is the "
    "*dominant* behaviour (only ~8% of baseline answers reach a final channel "
    "at 300 tokens) — the base `gpt-oss-20b` defaults to *Reasoning: medium* "
    "and reasons until it runs out of room. v4 was fine-tuned to answer "
    "directly and almost never does this."
)


def _format_panel(result: dict, label: str) -> str:
    """Markdown for one model's output panel (used in the Compare tab)."""
    if result["had_final"]:
        return f"{result['final_text']}\n\n{_meta_line(result, label)}"
    body = _NO_FINAL_NOTE
    if result["reasoning_text"]:
        trace = result["reasoning_text"][:_REASONING_PREVIEW_CHARS]
        if len(result["reasoning_text"]) > _REASONING_PREVIEW_CHARS:
            trace += " …"
        body += "\n\n**Reasoning trace (truncated):**\n\n" + "\n".join(
            f"> {line}" for line in trace.splitlines() or [trace]
        )
    return f"{body}\n\n{_meta_line(result, label)}"


# =============================================================================
# Event handlers (async generators → immediate "generating…" feedback)
# =============================================================================


async def ask_fn(question: str, temperature: float, max_tokens: float):
    """Ask tab: generate from v4 only. Yields (answer_textbox, status_md)."""
    if INIT_ERROR is not None:
        yield "", f"⚠️ Backend unavailable at startup: {INIT_ERROR}"
        return
    if not question or not question.strip():
        yield "", "Enter a question above, then click **Generate**."
        return
    yield "", "⏳ Generating from **v4**…"
    try:
        r = await _generate(v4_client, question.strip(), temperature, max_tokens)
    except Exception as exc:  # noqa: BLE001
        yield "", f"⚠️ Generation failed — {_friendly_error(exc)}"
        return
    if r["had_final"]:
        yield r["final_text"], _meta_line(r, "v4")
    else:
        note = (
            "The model produced no final answer (budget consumed while "
            "reasoning). Try raising **max tokens**."
        )
        preview = r["reasoning_text"][:_REASONING_PREVIEW_CHARS]
        yield preview, f"⚠️ {note}\n\n{_meta_line(r, 'v4')}"


async def compare_fn(question: str, temperature: float, max_tokens: float):
    """Compare tab: generate from v4 and baseline in parallel.

    Yields (v4_panel_md, baseline_panel_md, status_md).
    """
    if INIT_ERROR is not None:
        yield "", "", f"⚠️ Backend unavailable at startup: {INIT_ERROR}"
        return
    if not question or not question.strip():
        yield "", "", "Enter a question above, then click **Compare**."
        return
    yield "", "", "⏳ Generating from **v4** and **baseline** in parallel…"
    q = question.strip()
    try:
        v4_r, bl_r = await asyncio.gather(
            _generate(v4_client, q, temperature, max_tokens),
            _generate(baseline_client, q, temperature, max_tokens),
        )
    except Exception as exc:  # noqa: BLE001
        yield "", "", f"⚠️ Generation failed — {_friendly_error(exc)}"
        return
    yield _format_panel(v4_r, "v4"), _format_panel(bl_r, "baseline"), "✓ Done."


# =============================================================================
# Static UI copy
# =============================================================================

TITLE_MD = "# 🛰️ UAV Doctrine Assistant — Phase A v4"

SUBTITLE_MD = (
    "Fine-tuned **GPT-OSS-20B** (LoRA) on US, NATO, and allied UAV combat "
    "doctrine, served via **Tinker**. "
    "**Research artifact — verify every output against the source documents.**"
)

COMPARE_CAPTION_MD = (
    "Both answers are generated from the **same prompt** with the **same "
    "sampling parameters**. v4 was trained on 18,730 synthetic Q&A pairs from "
    "58 doctrine sources. The un-adapted baseline frequently returns no final "
    "answer because it exhausts its token budget on chain-of-thought — that is "
    "the expected, documented baseline failure mode, not a bug in this app."
)

FOOTER_MD = (
    "---\n"
    "<sub>Generated by **gpt-oss-20b-uav-doctrine** (Phase A v4) via Tinker. "
    "Verify outputs against source documents. Research artifact — not for "
    "operational use.</sub>"
)

ABOUT_MD = f"""
## About this assistant

> ### ⚠️ Research artifact only — **NOT for operational use**
> This is an experimental fine-tuning research output. It is not a decision
> authority, not validated for any operational, targeting, or safety-of-flight
> purpose, and must never substitute for the authoritative doctrinal
> publications or a qualified human in the loop.

### What this is
A **LoRA adapter** (Phase A, variant **v4**) fine-tuned on top of OpenAI's
**`gpt-oss-20b`** (a 20B-parameter mixture-of-experts model), trained and
served through the [Tinker](https://thinkingmachines.ai/tinker) API. It answers
questions about UAV / UAS combat doctrine with concise, source-grounded
paraphrases. The base model is unchanged; only the small adapter is ours.

### Training data
- **58 doctrine sources**, **18,730 synthetic Q&A pairs** (GPT-5.4-generated,
  grounded in the source text; train/val/test split at the document-chunk level
  so no chunk leaks across splits).
- Sources are **public-domain / publicly available**: US joint & service
  publications (JP / ADP / ATP / FM / AFDP), NATO AJPs, UK JDPs, civil-aviation
  references (FAA, EASA, ICAO), DoDD 3000.09 and the DoD Dictionary, plus
  open analyses (RUSI, CNAS, ICRC, CRS, NPS).
- **No classified material** was used at any stage.

### Evaluation
- **82.6% win rate** vs the un-adapted `gpt-oss-20b` baseline on 1,873 held-out
  test Q&A pairs, judged pair-wise by GPT-5.4 (95% CI **80.9%–84.4%**).
- A randomized-position re-judge on a 300-example stratified sample reproduced
  the result at **80.7%** (bias-corrected, within 1.9pp) — position bias is not
  inflating the headline.
- **Honest caveat:** most baseline "losses" happen because the un-adapted base
  exhausts its token budget in chain-of-thought and never emits a final answer.
  On the subset where the baseline *did* answer, v4 still won **77.9%**.

### Limitations
- **Specificity drift.** v4 usually captures the gist but can omit or invent
  specifics (wrong message codes, missing conditions, approximate figures).
  Always check the cited publication.
- **No safety / alignment pass.** This is a doctrine-QA SFT adapter (project
  milestone M1), not a safety-tuned assistant.
- **English only.**
- **Snapshot of doctrine as of May 2026** (the corpus ingest date). Doctrine
  evolves; newer editions will not be reflected.
- **Held-out / unseen sources score lower** and on small samples.

### Links
- 🤗 Model card: [{HF_MODEL_URL}]({HF_MODEL_URL})
- 💻 Project repository: [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})
"""

# (label shown on the chip, question loaded into the box) — spans all four
# question types from the Phase A test taxonomy.
EXAMPLES: list[tuple[str, str]] = [
    ("factual · UAS in joint targeting", "What is the role of UAS in joint targeting?"),
    ("definitional · ROE (NATO)", "Define the rules of engagement in NATO doctrine."),
    ("procedural · lost-link in controlled airspace",
     "What steps must a UAS operator take upon experiencing lost-link in controlled airspace?"),
    ("conceptual · strike vs ISR separation",
     "Why does doctrine emphasize separation of strike and ISR functions?"),
    ("factual · FAA Notice JO 7110.65BB", "What is FAA Notice JO 7110.65BB?"),
    ("definitional · ROE meaning", "What does ROE stand for and what does it govern?"),
    ("procedural · control-station handover",
     "Describe the handover procedure for UAS operations between control stations."),
    ("conceptual · UAVs in Ukraine since 2022",
     "How has UAV use evolved in the Ukraine conflict since 2022?"),
]


# =============================================================================
# UI assembly
# =============================================================================


def _temperature_slider() -> gr.Slider:
    return gr.Slider(
        minimum=0.0, maximum=1.5, value=DEFAULT_TEMPERATURE, step=0.05,
        label="Temperature", info="0 = deterministic; eval used 0.3",
    )


def _max_tokens_slider() -> gr.Slider:
    return gr.Slider(
        minimum=50, maximum=500, value=DEFAULT_MAX_TOKENS, step=10,
        label="Max tokens", info="Eval used 300",
    )


def build_demo() -> gr.Blocks:
    # NB: in Gradio 6 the `theme` argument moved from Blocks() to launch().
    with gr.Blocks(title="UAV Doctrine Assistant — v4",
                   analytics_enabled=False) as demo:
        gr.Markdown(TITLE_MD)
        gr.Markdown(SUBTITLE_MD)
        if INIT_ERROR is not None:
            gr.Markdown(
                f"> ⚠️ Failed to start the Tinker backend: `{INIT_ERROR}`. "
                "Generation will not work until `TINKER_API_KEY` is set and the "
                "service is reachable."
            )

        with gr.Tabs():
            # ---- Tab 1: Ask --------------------------------------------------
            with gr.Tab("Ask"):
                ask_q = gr.Textbox(
                    label="Question", lines=3, max_lines=8,
                    placeholder="Ask about UAV doctrine...",
                )
                with gr.Row():
                    ask_temp = _temperature_slider()
                    ask_max = _max_tokens_slider()
                ask_btn = gr.Button("Generate", variant="primary")
                ask_out = gr.Textbox(
                    label="Answer (v4)", lines=8, max_lines=20,
                    interactive=False, show_label=True,
                )
                ask_status = gr.Markdown()
                ask_btn.click(
                    ask_fn, inputs=[ask_q, ask_temp, ask_max],
                    outputs=[ask_out, ask_status],
                )
                ask_q.submit(
                    ask_fn, inputs=[ask_q, ask_temp, ask_max],
                    outputs=[ask_out, ask_status],
                )

            # ---- Tab 2: Compare v4 vs Baseline -------------------------------
            with gr.Tab("Compare v4 vs Baseline"):
                cmp_q = gr.Textbox(
                    label="Question", lines=3, max_lines=8,
                    placeholder="Ask about UAV doctrine...",
                )
                with gr.Row():
                    cmp_temp = _temperature_slider()
                    cmp_max = _max_tokens_slider()
                cmp_btn = gr.Button("Compare", variant="primary")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### v4 (fine-tuned)")
                        cmp_v4 = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### baseline (GPT-OSS-20B)")
                        cmp_bl = gr.Markdown()
                cmp_status = gr.Markdown()
                gr.Markdown(COMPARE_CAPTION_MD)
                cmp_btn.click(
                    compare_fn, inputs=[cmp_q, cmp_temp, cmp_max],
                    outputs=[cmp_v4, cmp_bl, cmp_status],
                )
                cmp_q.submit(
                    compare_fn, inputs=[cmp_q, cmp_temp, cmp_max],
                    outputs=[cmp_v4, cmp_bl, cmp_status],
                )

            # ---- Tab 3: Examples ---------------------------------------------
            with gr.Tab("Examples"):
                gr.Markdown(
                    "Click an example to load it into the box, then **Generate** "
                    "(answers come from v4)."
                )
                ex_q = gr.Textbox(
                    label="Question", lines=2, max_lines=8,
                    placeholder="...or pick an example below.",
                )
                gr.Examples(
                    examples=[[q] for _, q in EXAMPLES],
                    inputs=[ex_q],
                    example_labels=[label for label, _ in EXAMPLES],
                    label="Example questions (all four question types)",
                )
                with gr.Row():
                    ex_temp = _temperature_slider()
                    ex_max = _max_tokens_slider()
                ex_btn = gr.Button("Generate", variant="primary")
                ex_out = gr.Textbox(
                    label="Answer (v4)", lines=8, max_lines=20, interactive=False,
                )
                ex_status = gr.Markdown()
                ex_btn.click(
                    ask_fn, inputs=[ex_q, ex_temp, ex_max],
                    outputs=[ex_out, ex_status],
                )

            # ---- Tab 4: About ------------------------------------------------
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD)

        # Footer — outside the Tabs, so it shows on every tab.
        gr.Markdown(FOOTER_MD)

    return demo


demo = build_demo()


if __name__ == "__main__":
    # Blocking launch keeps the server alive; queue caps concurrent load so a
    # burst of visitors doesn't trip Tinker rate limits.
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        ssr_mode=False,
        show_error=True,
        share=False,
    )
