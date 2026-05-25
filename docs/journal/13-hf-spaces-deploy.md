# Phase A v4 demo → HuggingFace Spaces

_Hand-written. Companion to `docs/journal/12-gradio-ui.md` (UI build) and
`deploy/hf_space/` (the deployed bundle). Date: 2026-05-25._

## Setup

| Item | Value |
|---|---|
| Space | `meftun/gpt-oss-20b-uav-doctrine-demo` |
| Visibility | **public** |
| SDK | `gradio` 6.14.0 (CPU basic / free tier) |
| License (Space) | `apache-2.0` (Space README YAML; matches the model card) |
| App | `deploy/hf_space/app.py` — pure Tinker `SamplingClient` front-end (v4 + baseline), no local model |
| Backend | Tinker hosts the v4 sampler + base model; the Space holds no weights |

The prompt head (Phases 1–3) arrived truncated this turn; the create/push/secret
flow was reconstructed from the Rules + the ready-made `deploy/hf_space/` bundle
that the parallel UI session had already committed (`44311af`, `a49f87c`).

## Pre-flight fixes

`deploy/hf_space/` needed **no edits** — the bundle (app.py, requirements.txt,
README.md with the Spaces YAML header, .env.example) was complete and correct.
One **deviation from the prompt's Rules** was required:

- The Rules name only `TINKER_API_KEY` as a Space Secret. But `app.py` calls
  `get_tokenizer("openai/gpt-oss-20b")` at import, and that tokenizer repo is
  **gated** on HF — without an `HF_TOKEN` the download 401s, the backend sets
  `INIT_ERROR`, and every generation fails (the Space would deploy "RUNNING" but
  be non-functional). So **`HF_TOKEN` was set as a second secret.** The
  functional test below confirms this was necessary and sufficient.

## Push

`huggingface_hub.upload_folder(repo_type="space", allow_patterns=[...])` — hard
allow-list of exactly the 4 bundle files (nothing else can leak):

- Files: `app.py`, `requirements.txt`, `README.md`, `.env.example` (+ auto `.gitattributes`)
- Space commit: `8f84e1b7b6caa35b2a0b93d03fb4fc65b473f70b`
- Ordering: `create_repo` → set secrets → `upload_folder`, so the build that the
  upload triggers already has both secrets present.

## Secret configuration

Both set via `HfApi.add_space_secret` (values never printed/logged):

| Secret | Status | Purpose |
|---|---|---|
| `TINKER_API_KEY` | set | Serves v4 + baseline via Tinker `SamplingClient`; backend won't start without it |
| `HF_TOKEN` | set | Downloads the **gated** `openai/gpt-oss-20b` tokenizer the renderer needs |

Note: the `HF_TOKEN` secret is the project's write-scoped fine-grained token
(`canReadGatedRepos: true`). Space secrets are injected as env vars and are not
exposed by the Space being public, and `app.py` never logs them — but for
least-privilege, a **read-only** HF token could replace it in *Settings →
Variables and secrets* with no code change.

## Build verification

- Stage progression (polled): `APP_STARTING → RUNNING` in **~107 s** (fast —
  Tinker does the GPU work, so the CPU build only installs the thin client deps).
- **End-to-end functional test** (`gradio_client` → `/ask_fn`, public Space, no
  token needed): asked *"What is the role of UAS in joint targeting?"* →

  > "UAS are used to support all phases of the joint targeting cycle. They help
  > find and identify targets, assess them, and evaluate the results of action."
  > — `v4 · 35 tokens · 7.8 s · stop=stop_sequence`

  A real v4 answer (clean final-channel, ~35 tokens, ~8 s cold) — so the Tinker
  clients **and** the gated-tokenizer download (i.e. the `HF_TOKEN` secret) both
  work. The Space is functional, not merely RUNNING.

## Model card update

The model repo README (`meftun/gpt-oss-20b-uav-doctrine`) now carries a
**🚀 Live demo** link to the Space, added near the top.

- Model repo commit: `9f9aa4b2ff81ebdcad57f35be890906a9405257b` ("Add live demo link")
- The local source of that README is `checkpoints/phase_a_qa/v4/adapter/README.md`
  (git-ignored; the model card lives on HF, not in git).

## Cost note

**Placeholder** (calibrate against the real Tinker invoice). At the ADR-0001
placeholder of $0.40 / 1M tokens:

- **Ask (v4 only):** ~400 prompt + ~50 output ≈ 450 tokens ⇒ **~$0.0002/request**.
- **Compare (v4 + baseline):** baseline burns its full ~300-token budget on CoT,
  so ≈ 450 (v4) + ~700 (baseline) ≈ 1,150 tokens ⇒ **~$0.0005/request**.

The **free Spaces CPU-basic tier is sufficient** — the Space is a thin API client;
Tinker performs all 20B-MoE inference remotely, so the Space needs no GPU.

Single-tenant caveat (carried from journal 12): every visitor's request uses the
one `TINKER_API_KEY` secret — no per-user quota or isolation. Fine for a gated/
low-traffic demo; add a rate limiter or auth before promoting it widely, or the
token's spend is open to anyone who finds the Space.

## Anything weird

- **Prompt head truncated.** Only Phase 4 (tail) onward arrived this turn;
  reconstructed Phases 1–3 from the Rules + the committed deploy bundle. Nothing
  ambiguous remained once `deploy/hf_space/` was treated as the source of truth.
- **`HF_TOKEN` was a required second secret** the Rules didn't mention (see
  Pre-flight fixes) — the single biggest "do the right thing vs. the literal
  spec" call this turn.
- **`gradio_client.Client(...)` rejects `hf_token=`** in the installed version;
  the public Space needs no auth to query, so the kwarg was dropped.
- **License field:** Space YAML is `apache-2.0` (per the prompt and matching the
  model card); the repo *code* `LICENSE` is MIT. Different artifacts, left as-is.
- **Concurrent sessions:** the parallel terminal built `ui_app.py` +
  `deploy/hf_space/` and committed them (`44311af`, `a49f87c`); this session owns
  the model push (journal 11) and this Spaces deploy. Working tree was clean at
  deploy time, so no collision.
