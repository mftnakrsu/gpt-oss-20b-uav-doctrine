"""Token-aware chunking with backtracking to natural boundaries.

Algorithm
---------
1. Tokenize the whole cleaned text once with offset mapping (HF fast
   tokenizer ``return_offsets_mapping=True``).
2. Walk the token list with a sliding window:
   - For each chunk starting at token ``i``:
     - ``soft`` target end = ``i + target_tokens``
     - ``hard`` cap end    = ``i + max_tokens``
   - If the document ends inside the hard cap, take the rest as the
     final chunk.
   - Otherwise, search the character range ``[i + target_tokens // 2,
     soft]`` *backwards* for the LAST occurrence of (in order of
     preference): paragraph break (``\n\n``), line break (``\n``),
     sentence end (``.!?`` followed by whitespace), word boundary
     (any whitespace).
   - If nothing natural is found in that window, extend the search
     forward into ``[soft, hard]``.
   - If still nothing, force a break at the hard cap (always a token
     boundary, so never mid-word).
3. Snap the chosen character break to the nearest preceding token
   boundary so chunks never end mid-token.
4. Advance the cursor by ``end - overlap_tokens`` so adjacent chunks
   overlap by ``overlap_tokens`` tokens — this gives the downstream Q&A
   step context continuity at boundaries.

Deterministic: same input + same tokenizer + same params → same output.
Bumping ``CHUNKING_VERSION`` re-marks chunk records.
"""

from __future__ import annotations

import re

CHUNKING_VERSION = "v1"

PARAGRAPH_BREAK = "\n\n"
LINE_BREAK = "\n"
RE_SENTENCE_END = re.compile(r"[.!?][\s\n]")
RE_WORD_BOUNDARY = re.compile(r"\s")


def _find_best_break(text: str, lo: int, hi: int) -> int | None:
    """Find the best char position in ``text[lo:hi]`` to break at.

    Returns the character index *after* the chosen break (so
    ``text[:idx]`` ends at the break), or ``None`` if no natural
    boundary is present in the window.

    Preference order:
      1. last ``\\n\\n``  (paragraph)
      2. last ``\\n``     (line)
      3. last sentence end ``.!?`` followed by whitespace
      4. last whitespace (word boundary)
    """
    if lo >= hi:
        return None
    window = text[lo:hi]

    idx = window.rfind(PARAGRAPH_BREAK)
    if idx >= 0:
        return lo + idx + len(PARAGRAPH_BREAK)
    idx = window.rfind(LINE_BREAK)
    if idx >= 0:
        return lo + idx + len(LINE_BREAK)
    matches = list(RE_SENTENCE_END.finditer(window))
    if matches:
        return lo + matches[-1].end()
    matches = list(RE_WORD_BOUNDARY.finditer(window))
    if matches:
        return lo + matches[-1].end()
    return None


def chunk_text(
    text: str,
    tokenizer,
    target_tokens: int = 1200,
    max_tokens: int = 1500,
    overlap_tokens: int = 150,
) -> list[dict]:
    """Greedy token-windowed chunking with backtracking to natural boundaries.

    Parameters
    ----------
    text : str
        Cleaned source text.
    tokenizer : HF PreTrainedTokenizerFast
        Must support ``return_offsets_mapping=True``.
    target_tokens : int
        Soft target chunk size.
    max_tokens : int
        Hard cap. Chunks never exceed this.
    overlap_tokens : int
        Tokens of context to repeat at the start of the next chunk.

    Returns
    -------
    list[dict]
        One dict per chunk with fields ``chunk_index``, ``text``,
        ``char_offset_start``, ``char_offset_end``, ``token_count``,
        ``char_count``. ``chunk_id`` etc. are added by the orchestrator.
    """
    if target_tokens <= 0 or max_tokens < target_tokens or overlap_tokens < 0:
        raise ValueError("invalid chunking params")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be < target_tokens")
    if not text or not text.strip():
        return []

    enc = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets: list[tuple[int, int]] = enc["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []

    chunks: list[dict] = []
    i = 0
    safety = 0
    safety_cap = n + 16  # paranoia bound — should never trip in practice

    while i < n:
        safety += 1
        if safety > safety_cap:
            break

        soft_end_t = min(i + target_tokens, n)
        hard_end_t = min(i + max_tokens, n)

        if hard_end_t == n:
            end_t = n
        else:
            soft_char = offsets[soft_end_t - 1][1]
            hard_char = offsets[hard_end_t - 1][1]
            # Backtrack window: [half_target, soft]. Half-target is the
            # earliest token we are willing to break at (avoids tiny chunks).
            half_t = max(i + 1, i + target_tokens // 2)
            half_t = min(half_t, soft_end_t)
            min_back_c = offsets[half_t - 1][1]

            break_c = _find_best_break(text, min_back_c, soft_char)
            if break_c is None:
                # extend forward into the slack region (soft, hard]
                break_c = _find_best_break(text, soft_char, hard_char)
            if break_c is None:
                break_c = hard_char  # forced — but still on a token boundary

            # Snap to the largest token whose offset_end <= break_c.
            end_t = i + 1
            for k in range(hard_end_t - 1, i - 1, -1):
                if offsets[k][1] <= break_c:
                    end_t = k + 1
                    break

        start_c = offsets[i][0]
        end_c = offsets[end_t - 1][1]
        chunk_str = text[start_c:end_c]
        if chunk_str.strip():
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "text": chunk_str,
                    "char_offset_start": start_c,
                    "char_offset_end": end_c,
                    "token_count": end_t - i,
                    "char_count": len(chunk_str),
                }
            )

        if end_t >= n:
            break
        # Advance with overlap (but always make progress), then snap the
        # overlap-start forward to the next English-word boundary so chunks
        # never *begin* mid-word — BPE token boundaries are not English-word
        # boundaries, so without this we sometimes start a chunk with the
        # tail of a split word like "argame" from "wargame".
        candidate_i = max(end_t - overlap_tokens, i + 1)
        next_i = candidate_i
        search_cap = min(end_t, candidate_i + 64)
        for k in range(candidate_i, search_cap):
            char_start = offsets[k][0]
            if char_start == 0 or text[char_start - 1] in (" ", "\t", "\n"):
                next_i = k
                break
        i = next_i

    return chunks
