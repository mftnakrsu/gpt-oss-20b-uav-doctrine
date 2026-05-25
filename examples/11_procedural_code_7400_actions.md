# 11. Controller actions when a UAS squawks 7400

**Source:** `faa_notice_uas_lost_link/N_JO_7110.724_5-2-9_UAS_Lost_Link.pdf`
**Question type:** procedural
**Judge winner:** A (v4)

## Question

> If a controller observes a UAS displaying code 7400, what actions are they expected to take?

## Reference answer

The controller should determine the aircraft’s lost link procedure from the applicable airworthiness certificate or waiver/authorization, then coordinate as needed so the UAS can fly that procedure. They should also notify the Front Line Manager when feasible for documentation. If the aircraft is not following its programmed lost link profile or shows another anomaly, the event should be handled as an emergency situation under existing FAA procedures.

## v4 (fine-tuned) — 37 tokens

The controller should issue traffic advisories and safety alerts to aircraft that may be affected. The chunk also notes that some UAS may not show any code at all.

## Baseline (gpt-oss-20b, un-adapted) — 300 tokens

> ⚠️ Baseline did not reach a final answer within its 300-token budget — it exhausted the budget in the hidden reasoning (`analysis`) channel. This is the dominant baseline behavior (~92% of cases); see the project README.

## Judge scores (GPT-5.4, 1–5 each)

|          | factual_accuracy | completeness | grounding |
|----------|:----------------:|:------------:|:---------:|
| v4       | 2 | 1 | 2 |
| baseline | 1 | 1 | 1 |

**Rationale:** Model A provides a partially relevant but largely incomplete action, while Model B provides no answer at all.
