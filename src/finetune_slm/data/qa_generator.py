"""Doctrine Q&A pair generation via Azure GPT-5.4.

Three pieces:

- ``build_qa_messages(chunk, k)`` — Build the system+user messages for
  generating ``k`` grounded Q&A pairs from a single chunk.
- ``parse_qa_response(text)`` — Robust JSON parsing: strips markdown
  fences and code blocks, validates schema, skips malformed entries
  within an otherwise valid response.
- ``generate_pairs_for_chunk(llm, chunk, k)`` — One LLM call producing
  up to ``k`` pairs; one retry with reduced temperature on JSON parse
  failure (gracefully degrades to ``([], usage, reason)`` if the model
  rejects the lower temperature, since GPT-5 family pins temperature
  to 1.0).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

QA_SYSTEM_PROMPT = (
    "You are an expert military doctrine educator generating supervised "
    "fine-tuning data for a small language model that will answer "
    "questions about Unmanned Aerial Vehicle (UAV) combat doctrine and "
    "operations."
)

QA_USER_TEMPLATE = """Below is a chunk of doctrinal text. Generate exactly {k} diverse, high-quality question-answer pairs grounded in this chunk.

Source: {source_filename}
Chunk text:
\"\"\"
{chunk_text}
\"\"\"

Requirements:
- Each question must be fully answerable from the chunk above. No external knowledge required.
- Answers must be grounded in the chunk; paraphrase clearly into natural prose. Avoid verbatim quotes longer than 15 words.
- Diverse question types across the {k} pairs: factual recall, definitional, procedural, conceptual. Avoid yes/no questions.
- Do NOT reference paragraph numbers, page numbers, figure numbers, or document structure ("according to paragraph 3..." is forbidden).
- Questions should sound natural — what a doctrine student or military professional might genuinely ask in their own words.
- Answers typically 1-4 sentences. Longer only if the chunk justifies it.

Output ONLY valid JSON in this exact shape, no preamble, no markdown, no commentary:
{{"pairs": [{{"question": "...", "answer": "...", "question_type": "factual"}}, ...]}}

question_type must be one of: "factual", "definitional", "procedural", "conceptual"."""

VALID_QUESTION_TYPES = {"factual", "definitional", "procedural", "conceptual"}

# Strip an optional ``` or ```json fence around a JSON body.
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(?P<body>.*?)\n?```\s*$",
    re.DOTALL,
)


def build_qa_messages(chunk: dict, k: int = 5) -> list[dict]:
    """Return the system+user messages for one Q&A generation call."""
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": QA_USER_TEMPLATE.format(
                k=k,
                source_filename=chunk["source_filename"],
                chunk_text=chunk["text"],
            ),
        },
    ]


def _strip_fences(text: str) -> str:
    """Strip a single outer ``` or ```json fence if present."""
    m = _FENCE_RE.match(text.strip())
    if m:
        return m.group("body")
    return text


def _validate_pair(obj: Any) -> Optional[dict]:
    """Return a clean pair dict or None if obj is malformed."""
    if not isinstance(obj, dict):
        return None
    q = obj.get("question")
    a = obj.get("answer")
    qt = obj.get("question_type")
    if not isinstance(q, str) or not q.strip():
        return None
    if not isinstance(a, str) or not a.strip():
        return None
    if not isinstance(qt, str) or qt not in VALID_QUESTION_TYPES:
        # Coerce unknown question types to "factual" rather than dropping the
        # whole pair — conservative bias, the question content matters more.
        qt = "factual"
    return {"question": q.strip(), "answer": a.strip(), "question_type": qt}


def parse_qa_response(text: str) -> list[dict]:
    """Robustly parse the LLM's JSON response into a list of valid pairs.

    Steps:
      1. Strip any markdown code fence.
      2. ``json.loads`` the body. On failure, return [].
      3. Look for ``pairs`` (list) at the top level. Other shapes (bare
         list, single object) accepted as best-effort.
      4. Per-pair validation: drop malformed entries, keep good ones.
    """
    if not text or not text.strip():
        return []
    body = _strip_fences(text.strip())
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        # Last-ditch: find the first { ... } block and try again.
        m = re.search(r"\{.*\}", body, re.DOTALL)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        if isinstance(parsed.get("pairs"), list):
            raw_pairs = parsed["pairs"]
        elif {"question", "answer"}.issubset(parsed.keys()):
            raw_pairs = [parsed]
        else:
            return []
    elif isinstance(parsed, list):
        raw_pairs = parsed
    else:
        return []

    pairs: list[dict] = []
    for obj in raw_pairs:
        clean = _validate_pair(obj)
        if clean is not None:
            pairs.append(clean)
    return pairs


async def generate_pairs_for_chunk(
    llm_client,
    chunk: dict,
    k: int = 5,
) -> tuple[list[dict], dict, Optional[str]]:
    """Generate up to ``k`` Q&A pairs grounded in ``chunk``.

    Returns ``(pairs, total_usage, error_reason)``. ``error_reason`` is
    None on success or a short string describing why no pairs were
    produced. Usage aggregates both the first attempt and the
    low-temperature retry (if it happened).
    """
    messages = build_qa_messages(chunk, k=k)
    cumulative = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # First attempt — default temperature (env-controlled).
    try:
        content, usage = await llm_client.chat_completion(
            messages,
            response_format={"type": "json_object"},
            max_tokens=2000,
        )
    except Exception as e:
        return [], cumulative, f"LLM call failed: {type(e).__name__}: {e}"

    cumulative = {k_: cumulative[k_] + usage[k_] for k_ in cumulative}

    pairs = parse_qa_response(content)
    if pairs:
        return pairs, cumulative, None

    # Retry once at lower temperature. GPT-5 family may reject — catch and
    # surface as a soft failure rather than blowing up the whole chunk.
    try:
        content2, usage2 = await llm_client.chat_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2000,
        )
        cumulative = {k_: cumulative[k_] + usage2[k_] for k_ in cumulative}
    except Exception as e:
        return [], cumulative, (
            f"JSON parse failed on first attempt; retry rejected by model: "
            f"{type(e).__name__}: {e}"
        )

    pairs = parse_qa_response(content2)
    if pairs:
        return pairs, cumulative, None
    return [], cumulative, "JSON parse failed on both attempts"
