# Example gallery — v4 (fine-tuned) vs baseline

These are **real, unedited outputs** from the Phase A held-out evaluation: a GPT-5.4 pair-wise judge scored every answer from the v4 LoRA fine-tune (Model A) against the un-adapted `openai/gpt-oss-20b` baseline (Model B) on three axes — factual accuracy, completeness, and grounding (1–5 each). Every question, reference answer, generation, score, and rationale below is copied verbatim from `outputs/phase_a_eval/`; nothing here is hand-edited or paraphrased. This is **not** a cherry-picked highlight reel — it deliberately includes ties, borderline calls, and cases where v4 is itself wrong.

## Selection methodology

We drew 16 comparisons — 4 from each of the four question types (factual, definitional, procedural, conceptual) — from the fully-formatted sample comparisons in `docs/journal/09-phase-a-eval-report.md` (15 from the test split, 15 from the held-out split). The set spans the full range of judge outcomes (clear v4 wins, borderline wins where v4 is only partially correct, and outright ties where both models fail) and a mix of source types: official doctrine (US joint pubs, NATO AJPs, US Army FM, UK JDP, FAA orders, DoD Dictionary), academic / war-college work (NPS theses), a think-tank position paper (ICRC), and a modern legislative case study (CRS counter-UAS report, FAA UAS lost-link notice). Per-example output token counts are taken from the generation JSONL files.

A note on the baseline: in ~92% of test cases the un-adapted `gpt-oss-20b` never produces a final answer — it spends its entire 300-token budget in the hidden `analysis` reasoning channel and emits nothing in the `final` channel. The judge then scores it at the floor (1/1/1). Examples 01, 08, and 15 are the rarer cases where the baseline did reach a (still truncated) answer; the rest show the dominant empty-output behavior.

## Distribution

### By question type

| question type | count |
|---------------|:-----:|
| factual       | 4 |
| definitional  | 4 |
| procedural    | 4 |
| conceptual    | 4 |
| **total**     | **16** |

### By judge winner

| winner          | count |
|-----------------|:-----:|
| A (v4)          | 12 |
| tie             | 4 |
| B (baseline)    | 0 |
| **total**       | **16** |

## All examples

**Factual**
- [01. Electronic warfare support and its operational use](01_factual_ew_support.md) — A (v4)
- [02. Level 3 activity and who commands it](02_factual_level_3_activity.md) — tie
- [03. US government support in foreign internal defense](03_factual_fid_support.md) — A (v4)
- [04. UAS lost-link transponder code](04_factual_lost_link_7400.md) — A (v4)

**Definitional**
- [05. The weight factor f in a swarm allocation scheme](05_definitional_weight_factor_f.md) — A (v4)
- [06. Segregation in detainee operations](06_definitional_segregation.md) — A (v4)
- [07. What SLTT law enforcement refers to](07_definitional_sltt.md) — tie
- [08. How doctrine describes an air concept of operations](08_definitional_air_conops.md) — A (v4)

**Procedural**
- [09. Message format for a delayed parallel offset](09_procedural_parallel_offset.md) — tie
- [10. The four planning functions](10_procedural_planning_functions.md) — A (v4)
- [11. Controller actions when a UAS squawks 7400](11_procedural_code_7400_actions.md) — A (v4)
- [12. What commanders must assess before activating an AWS](12_procedural_aws_assessment.md) — A (v4)

**Conceptual**
- [13. Why support areas need protection planning on par with close areas](13_conceptual_support_area_protection.md) — A (v4)
- [14. How civilian protection changes in hostilities](14_conceptual_civilian_protection.md) — A (v4)
- [15. Why judging an adversary’s intentions is especially hard](15_conceptual_judging_intentions.md) — A (v4)
- [16. Why federal criminal laws limit counter-drone options](16_conceptual_federal_laws_cuas.md) — tie
