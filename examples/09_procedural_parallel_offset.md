# 09. Message format for a delayed parallel offset

**Source:** `faa_jo_7110_65/7110.65BB_Bsc_w_Chg_1_and_2_dtd_1-22-26_Final.pdf`
**Question type:** procedural
**Judge winner:** tie

## Question

> If ATC wants an aircraft to start flying a parallel offset from the cleared route only after reaching a specific point, which message format should be used and what does it mean?

## Reference answer

Use UM65: at a specified position, offset a stated distance and direction from route. It tells the aircraft to begin the parallel track at that named position rather than immediately.

## v4 (fine-tuned) — 52 tokens

Use DM57, "CLIMB TO (altitude) AND FOLLOW (direction) (distance) (direction)." This tells the aircraft to begin the parallel offset at the specified altitude and in the stated direction and distance from the route.

## Baseline (gpt-oss-20b, un-adapted) — 300 tokens

> ⚠️ Baseline did not reach a final answer within its 300-token budget — it exhausted the budget in the hidden reasoning (`analysis`) channel. This is the dominant baseline behavior (~92% of cases); see the project README.

## Judge scores (GPT-5.4, 1–5 each)

|          | factual_accuracy | completeness | grounding |
|----------|:----------------:|:------------:|:---------:|
| v4       | 1 | 1 | 1 |
| baseline | 1 | 1 | 1 |

**Rationale:** Model A gives the wrong message format and meaning, while Model B provides no answer, so both fail against the reference.
