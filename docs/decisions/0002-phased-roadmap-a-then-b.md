# ADR 0002 — Phased roadmap: doctrine Q&A first, scenario reasoning second

- Status: Accepted
- Date: 2026-05-24

## Context

The end product is a small language model that can make grounded
tactical decisions in UAV combat scenarios — for example, recommending
a lost-link procedure consistent with the relevant ATC publication,
applicable ROE, and the platform's pre-briefed contingency plan.

That capability has two distinct ingredients:

1. **Knowing the doctrine.** Retrieval-and-paraphrase over a defined
   public-domain corpus — what the publication actually says, with
   citations.
2. **Applying the doctrine in a scenario.** Given a tactical situation
   description, selecting and adapting the right doctrinal procedure,
   respecting hard limits.

Trying to learn both simultaneously from a single dataset risks the
model learning "scenario answer shapes" without the grounding, or vice
versa, with no clear signal of where it failed.

## Decision

Train in two sequential phases, sharing a single base model, data
pipeline, and eval harness:

- **Phase A — Doctrine Q&A.** Fine-tune GPT-OSS-20B (LoRA, via Tinker)
  on grounded Q&A pairs derived from `data/raw/`. The model learns to
  answer "what does publication X say about Y" with a paraphrase plus a
  citation. Output: `phase-a-qa` checkpoint.
- **Phase B — Tactical scenario reasoning.** *Continue* training from
  the Phase A checkpoint on a tactical-scenario dataset (lost-link,
  C-UAS, airspace deconfliction, ROE-bounded decisions). The model
  inherits doctrinal recall from Phase A and adds situated
  decision-making. Output: `phase-b-scenario` checkpoint.

The data pipeline (`src/finetune_slm/data/`), training loop
(`src/finetune_slm/training/`), and eval harness with pluggable graders
(`src/finetune_slm/eval/`) are shared. Phase A and Phase B differ only
in their dataset-builder configuration and the grader set the eval
harness loads.

Dataset roots are pre-separated under `data/processed/qa/` and
`data/processed/scenario/`. Tinker configs are pre-separated under
`configs/tinker/phase_a_qa/` and `configs/tinker/phase_b_scenario/`.

## Consequences

### Positive

- **Isolates failures.** A Phase A eval regression points at doctrine
  recall; a Phase B regression points at scenario reasoning. We never
  have to disentangle "did the model forget the publication or
  mis-apply it?"
- **Reuses Phase A as a stepping stone.** Phase B does not start from
  the base — it starts from a model that already grounds answers in
  the corpus. Less data, fewer tokens, fewer dollars to get Phase B to
  a working state.
- **Shared infra, halved maintenance.** One data pipeline, one eval
  harness, one Tinker setup. Phase B is a config flip plus new graders.
- **Releasable midstream.** `phase-a-qa` is itself a useful artifact
  (citable doctrine Q&A) — it can be evaluated and shipped before
  Phase B starts.

### Negative / risks

- **Catastrophic forgetting in Phase B.** Continued training on
  scenario data can degrade Phase A's Q&A skill. Mitigation: keep the
  Phase A eval set in the Phase B eval harness and watch for
  regression; consider replay of a small Phase A slice in Phase B.
- **Phase B data is harder to produce.** Doctrine Q&A pairs come from
  the corpus more or less mechanically; scenario data needs SME-shaped
  authoring. This is a known cost — the phasing makes it explicit
  rather than burying it.
- **Eval graders must be pluggable.** Phase A graders are mostly
  exact-match / citation-grounding checks; Phase B graders need to
  judge tactical appropriateness, hard-limit compliance, and ROE
  fidelity. The harness must support swapping graders per phase.

## Related

- [[0001-base-model-and-platform]] — the GPT-OSS-20B + Tinker choice
  that this roadmap is built on.
