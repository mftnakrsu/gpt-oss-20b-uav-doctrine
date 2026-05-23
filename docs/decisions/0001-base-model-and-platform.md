# ADR 0001 — Base model and training platform

- Status: Accepted
- Date: 2026-05-24

## Context

We need to fine-tune a single small language model on a public-domain
UAV / UAS combat-doctrine corpus that has already been ingested into
`data/raw/` (~108 MB, ~1.78M raw tokens across 30 PDFs). The model has
to be:

1. **Future-safe on the chosen training platform.** Whatever model we
   commit to must not appear on the platform's near-term deprecation
   list — fine-tuning on a model that disappears mid-project is wasted
   spend.
2. **Small enough to fine-tune on a $5k budget**, but large enough to
   support both doctrine Q&A (Phase A) and tactical-scenario reasoning
   (Phase B) from the same base.
3. **MoE-friendly.** Doctrine spans heterogeneous styles (US joint /
   NATO / UK / civil-aviation / DoD policy). Mixture-of-experts handles
   this kind of stylistic and topical diversity well for its active-
   parameter count.
4. **Available via a managed platform that does not require us to own a
   GPU.** No local NVIDIA hardware is in scope; we're shipping from a
   Mac and using cloud-managed training.

## Decision

- **Base model:** OpenAI **GPT-OSS-20B** (MoE).
- **Training platform:** **Tinker** by Thinking Machines, using LoRA
  via [`tinker-cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook).

## Consequences

### Positive

- **Future-safe:** GPT-OSS-20B is *not* on Tinker's June 12 retirement
  list, so any checkpoint we produce remains trainable / re-trainable
  on the platform for the foreseeable future.
- **No local GPU required:** Tinker manages compute end-to-end. The
  laptop only has to drive the API, build datasets, and run eval.
- **Budget headroom:** at the placeholder rate of ~$0.40 per 1M
  training tokens, even three full epochs over the ~1.78M-token corpus
  cost a small fraction of the $5k budget — there is ample headroom for
  Phase B continued training, multiple eval-driven iterations, and
  recovery from a few false starts. (Actual unit price to be confirmed
  on the Tinker dashboard.)
- **MoE expressivity for doctrine diversity:** GPT-OSS-20B's MoE
  architecture is well-suited to a corpus that mixes US joint, NATO,
  UK, EASA, ICAO, FAA, and DoD-policy registers.
- **Shared base across both phases:** Phase B continues training from
  the Phase A LoRA checkpoint — no need to re-pick a base.

### Negative / risks

- **Tokenizer access:** `openai/gpt-oss-20b` may be gated on Hugging
  Face. The corpus tokenizer path therefore falls back to `tiktoken`
  `o200k_base`, which is the same family and a safe approximation for
  data-inventory purposes. Training itself uses the platform-provided
  tokenizer.
- **Platform lock-in:** Migrating away from Tinker later requires
  re-exporting LoRA adapters and re-validating against another
  platform's tokenizer / serving stack.
- **Price assumption is a placeholder.** $0.40/1M train tokens is used
  for budgeting until verified against the live Tinker dashboard. All
  cost estimates in the journal are flagged accordingly.

## Related

- [[0002-phased-roadmap-a-then-b]] — phased A→B fine-tuning roadmap that
  this base-model choice supports.
