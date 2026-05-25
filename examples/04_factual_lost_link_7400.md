# 04. UAS lost-link transponder code

**Source:** `faa_notice_uas_lost_link/N_JO_7110.724_5-2-9_UAS_Lost_Link.pdf`
**Question type:** factual
**Judge winner:** A (v4)

## Question

> What transponder code does this notice assign for a UAS lost link, and how should controllers handle aircraft that still use the older code?

## Reference answer

The notice authorizes code 7400 as the dedicated beacon code for UAS lost link events. However, some unmanned aircraft may still squawk 7600, and controllers are instructed to treat those cases as lost link situations and use the same handling procedures.

## v4 (fine-tuned) — 54 tokens

The notice assigns code 7400 for UAS when the lost link function is activated. Aircraft that still use 1200 should be handled as if they are operating under an emergency, and controllers should not assign 1200 to other UAS.

## Baseline (gpt-oss-20b, un-adapted) — 300 tokens

> ⚠️ Baseline did not reach a final answer within its 300-token budget — it exhausted the budget in the hidden reasoning (`analysis`) channel. This is the dominant baseline behavior (~92% of cases); see the project README.

## Judge scores (GPT-5.4, 1–5 each)

|          | factual_accuracy | completeness | grounding |
|----------|:----------------:|:------------:|:---------:|
| v4       | 2 | 2 | 2 |
| baseline | 1 | 1 | 1 |

**Rationale:** Model A at least identifies 7400 correctly but invents incorrect handling about 1200 instead of the older 7600 code, while Model B provides no answer.
