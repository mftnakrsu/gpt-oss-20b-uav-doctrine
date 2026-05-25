# 05. The weight factor f in a swarm allocation scheme

**Source:** `nps_swarm_vs_swarm_2013/ADA582919.pdf`
**Question type:** definitional
**Judge winner:** A (v4)

## Question

> In this allocation scheme, what does the weight factor f represent?

## Reference answer

The factor f expresses how much a UAV prefers offense versus defense. A larger f pushes it toward the primary objective of attacking the adversary’s home base, while values near zero emphasize engaging enemy UAVs instead.

## v4 (fine-tuned) — 42 tokens

f is the weight factor used to distribute the total number of UAVs among the different threat types. It is the key parameter that determines how many UAVs are assigned to each threat category.

## Baseline (gpt-oss-20b, un-adapted) — 300 tokens

> ⚠️ Baseline did not reach a final answer within its 300-token budget — it exhausted the budget in the hidden reasoning (`analysis`) channel. This is the dominant baseline behavior (~92% of cases); see the project README.

## Judge scores (GPT-5.4, 1–5 each)

|          | factual_accuracy | completeness | grounding |
|----------|:----------------:|:------------:|:---------:|
| v4       | 2 | 2 | 2 |
| baseline | 1 | 1 | 1 |

**Rationale:** Model A at least identifies f as an allocation weight parameter, but it misses the crucial offense-versus-defense preference described in the reference, while Model B provides no answer.
