# Phase A v4 → HuggingFace push

_Hand-written narrative of the export + push, not script-generated. Artifacts:
`checkpoints/phase_a_qa/v4/{raw_tinker,adapter,export_summary.json,hf_push.json,hf_verify.json}`._

## Setup

| Item | Value |
|---|---|
| Repo | `meftun/gpt-oss-20b-uav-doctrine` |
| Visibility | **public** |
| License | `apache-2.0` (matches the gpt-oss-20b base license) |
| Base model | `openai/gpt-oss-20b` |
| Source checkpoint | v4 sampler `tinker://d79c263a-1e99-57b9-b09e-584fc222f3da:train:0/sampler_weights/final-step-00994` |
| Token | fine-grained, displayName `uav_combat`, `repo.write` scoped to user `meftun` (write verified via `whoami` before push) |

## Adapter download

Extracted with `scripts/export_lora_adapter.py` (the prior session's script; re-used, not regenerated — the artifacts were already present and valid):

1. `publish_checkpoint_from_tinker_path` (idempotent) → make the checkpoint downloadable.
2. `get_checkpoint_archive_url_from_tinker_path` → signed GCS archive URL. The default Tinker transport (pyqwest, Rust HTTP/2) resets the long-running stream while the server assembles the archive, so the client is pinned to httpx HTTP/1.1 for the URL fetch; the archive GET itself is a plain `urllib` download of the signed URL.
3. Safe tar extraction (rejects symlinks/hardlinks + path traversal) → `raw_tinker/`.
4. Convert Tinker-native → PEFT with `tinker_cookbook.weights._adapter.build_lora_adapter`, patched to read the base model's *parameter key names* from `model.safetensors.index.json` only (a few hundred KB) instead of `snapshot_download`-ing the 13.76 GB of base weights — the gpt_oss merge profile never needs tensor data.

File sizes:

| file | size | tensors | dtype | notes |
|---|---:|---:|---|---|
| `raw_tinker/adapter_model.safetensors` | 482.8 MB | 338 | F32 | Tinker-native, **fused** MoE experts |
| `adapter/adapter_model.safetensors` | 894.3 MB | 4,802 | F32 | PEFT format, **un-fused** per-expert |
| `adapter/adapter_config.json` | 500 B | — | — | PEFT config |
| `adapter/README.md` | ~14 KB | — | — | model card |

`export_summary.json` (local export verification): `r=16`, `lora_alpha=32`, `peft_type=LORA`, `task_type=CAUSAL_LM`, 8 target modules (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head`), 4,802 LoRA tensors, base `openai/gpt-oss-20b`.

**Why 894 MB (not the ~100–500 MB rule of thumb).** Tinker exports the adapter in **F32**, and gpt-oss-20b is a 32-expert MoE with LoRA on every expert's gate/up/down plus attention q/k/v/o plus `lm_head`. The cookbook conversion **un-fuses** the per-layer fused expert LoRA into per-expert PEFT tensors: 338 → 4,802 tensors, 483 → 894 MB. Both files are valid F32 LoRA; the growth is the un-fusing, not corruption. (~24 layers × (32 experts × 3 + 4 attn) + 1 lm_head ≈ 2,401 LoRA modules × 2 tensors = 4,802.)

## Model card

`adapter/README.md`, ~206 lines, 11 H2 sections (TL;DR, Quick start, Training data, Training procedure, Evaluation, Limitations, Intended/out-of-scope, Methodological findings, Reproducibility, Citation, Acknowledgments) plus a scope/intent warning.

Every number traces to journals 02–10 (verified this turn against the journals, not the prompt draft). Two **corrections** vs the prompt's draft card:

- **`lora_alpha` = 32**, not 16 — the training config (journal 07) set rank only; Tinker's default α is 2×rank, and the exported config confirms 32.
- **`lora_dropout` = 0.0**, not 0.05 — confirmed in the exported config.

Key claims: 82.6% win rate (95% CI 80.9–84.4) original / 80.7% (76.3–85.0) bias-corrected; baseline-budget-exhaustion caveat (91.8% empty → cleaner 77.9% on the 154-answer subset); honest "per-expert MoE LoRA may need a serving stack that supports it" loading note; `SYSTEM_PROMPT` in the Quick start matches the training renderer verbatim.

## Push outcome

| Item | Value |
|---|---|
| Public URL | https://huggingface.co/meftun/gpt-oss-20b-uav-doctrine |
| Commit hash | `e3fe5970f775a78eb945ea385b9d43725c9d0a81` |
| Commit message | `Initial Phase A v4 LoRA adapter + model card` |
| Files uploaded | `README.md`, `adapter_config.json`, `adapter_model.safetensors` (+ auto `.gitattributes`) |
| Method | `huggingface_hub.upload_folder` with `allow_patterns` hard allow-list = exactly the 3 files |

The `allow_patterns` allow-list is the safety guarantee that no training data / qa_pairs / chunks / journals / `.env` / secrets can leak — only the 3 named files are eligible to upload.

## Verification

`scripts/verify_hf_adapter.py` — a clean `snapshot_download` of the repo (run with `HF_HUB_DOWNLOAD_TIMEOUT=30` so a flaky chunk retries/resumes instead of hanging like the upload did), then `peft.PeftConfig.from_pretrained` + a `safetensors.safe_open` structure check. **PASS.** Report: `checkpoints/phase_a_qa/v4/hf_verify.json`.

- Downloaded fresh from HF: `adapter_model.safetensors` 894.94 MB, `README.md` ~0.01 MB, `adapter_config.json` <0.01 MB, `.gitattributes`.
- PEFT config round-trips: `peft_type=LORA`, base `openai/gpt-oss-20b`, `r=16`, `lora_alpha=32`, `lora_dropout=0.0`, `task_type=CAUSAL_LM`, 8 target modules (`down_proj, gate_proj, k_proj, lm_head, o_proj, q_proj, up_proj, v_proj`).
- safetensors: **4,802 tensors, all 4,802 are LoRA tensors** — matches the local export exactly, so the LFS round-trip is byte-faithful.
- Inference check (load base + adapter, generate) **skipped** — needs the ~14 GB base model; out of scope on this machine, and explicitly optional per the task. The adapter is structure-validated, not generation-validated; first real generation will happen on a GPU host with the per-expert-LoRA serving caveat noted in the model card.

## Anything weird

- **The repo was already created-but-empty.** A prior session had run `create_repo` (repo existed, public, contained only `.gitattributes`) but `upload_folder` never completed — no adapter files, no `hf_push.json`, no journal 11. So this turn was *finishing* a partial push, not starting fresh. The scripts, deps (`huggingface-hub`, `safetensors`, `peft`), `raw_tinker/`, `adapter/`, and the model card were all already in place and valid; they just had never been pushed.
- **First push attempt hung for 25 minutes on a dead socket.** Upload stalled at 130/895 MB: the slow uplink dropped, and `huggingface_hub`'s LFS upload blocked on a socket with no timeout — the process stayed alive (state `S`, ~6 s CPU) with its progress output frozen for 25 min. Diagnosed via the output file's mtime vs wall clock. Fix: SIGKILL the process group, relaunch under `/tmp/push_watchdog.py` — a wrapper that SIGKILLs + retries (≤12×) whenever the output stops growing for 120 s. Resumable upload means retries make cumulative progress.
- **The link is highly variable.** First attempt crawled at 0.05→1.0 MB/s then dropped to ~0. The relaunch ran at a steady ~3.4 MB/s and pushed the full 895 MB in a **single** attempt (~4 min) — the watchdog never had to fire. The stall was a transient network event, not systemic.
- **Size reducer available if needed.** Casting the F32 adapter to bf16 would roughly halve it (~447 MB) and fit the usual LoRA size range, at negligible inference-quality cost (PEFT casts to compute dtype anyway). Kept F32 here to match the exported/verified artifact and the model card; revisit if upload size becomes a recurring pain on this link.
- **Only code + this journal enter git.** `checkpoints/` and `outputs/` are git-ignored, so the 895 MB safetensors, `hf_push.json`, and `hf_verify.json` are *not* committed — they live only on HF and locally. The commit carries `pyproject.toml`/`uv.lock` (the 3 added deps), the 3 scripts, and this journal.
