---
title: UAV Doctrine Assistant
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: apache-2.0
---

# UAV Doctrine Assistant — Phase A v4

Interactive front-end for a **LoRA adapter** (Phase A, variant **v4**) fine-tuned
on top of OpenAI's **`gpt-oss-20b`** for US / NATO / allied **UAV combat
doctrine** question-answering. Both the fine-tuned model and the un-adapted
baseline are sampled remotely through the [Tinker](https://thinkingmachines.ai/tinker)
API — nothing runs on the Space's own hardware, so the 20B MoE base fits with
no GPU on the Space itself.

Tabs:
- **Ask** — query v4.
- **Compare v4 vs Baseline** — same prompt, same sampling parameters, side by side.
- **Examples** — one-click questions across all four question types.
- **About** — training data, evaluation, and limitations.

## ⚠️ Research artifact — not for operational use

This is an experimental fine-tuning research output. It is **not** a decision
authority, is **not** validated for any operational, targeting, or
safety-of-flight purpose, and must never substitute for the authoritative
doctrinal publications or a qualified human in the loop. Outputs can omit or
invent specifics — **always verify against the source documents.**

## Configuration (Space Secrets)

This Space needs API credentials supplied as **Space Secrets**
(*Settings → Variables and secrets*), **not** as a committed file:

| Secret | Required | Purpose |
|---|---|---|
| `TINKER_API_KEY` | **yes** | Serves both models via Tinker's `SamplingClient`. The backend will not start without it. |
| `HF_TOKEN` | likely | Downloads the gated `openai/gpt-oss-20b` tokenizer used by the renderer. |

See `.env.example` for the variable names. The model itself is **not** bundled
in this Space — it is served from the Tinker sampler checkpoint referenced in
`app.py`.

## Links

- 🤗 Model card: <https://huggingface.co/meftun/gpt-oss-20b-uav-doctrine>
- 💻 Project repository: <https://github.com/mftnakrsu/uav-doctrine-slm>

## How it was evaluated

82.6% pair-wise win rate vs the un-adapted `gpt-oss-20b` baseline on 1,873
held-out test Q&A pairs (GPT-5.4 judge; 95% CI 80.9–84.4%), reproduced at 80.7%
under randomized-position re-judging. Most baseline "losses" are budget
exhaustion in chain-of-thought; on the subset where the baseline produced a
final answer, v4 still won 77.9%. See the **About** tab in the app for the full
breakdown and limitations.
