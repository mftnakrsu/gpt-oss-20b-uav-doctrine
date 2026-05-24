"""ChatML conversion and validation for Phase A SFT.

`to_chatml` converts a Q&A pair dict (as produced by the corpus pipeline)
into the OpenAI-style conversational format Tinker expects. `validate_chatml`
applies structural and token-budget rules; it accepts any tokenizer that
exposes `apply_chat_template(messages)` returning either token-ids (HF) or
through a small adapter returning ids (see `make_tokenizer` below).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

MIN_TOKENS = 20
MAX_TOKENS = 4096


def to_chatml(pair: dict, system_prompt: str) -> dict:
    """Convert a QA pair to the OpenAI conversational format Tinker expects."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ],
        "pair_id": pair["pair_id"],
        "chunk_id": pair["chunk_id"],
        "source_filename": pair["source_filename"],
        "question_type": pair["question_type"],
    }


def validate_chatml(example: dict, tokenizer: Any) -> tuple[bool, str | None]:
    """Validate a ChatML example.

    Returns (valid, error_reason). Rules:
    - messages must be a list with system/user/assistant in that order
    - every content field must be a non-empty string
    - must tokenize successfully via tokenizer.apply_chat_template
    - total tokens must be > MIN_TOKENS and < MAX_TOKENS
    """
    messages = example.get("messages")
    if not isinstance(messages, list):
        return False, "messages not a list"
    if len(messages) != 3:
        return False, f"expected 3 messages, got {len(messages)}"

    expected_roles = ["system", "user", "assistant"]
    for i, (msg, want_role) in enumerate(zip(messages, expected_roles)):
        if not isinstance(msg, dict):
            return False, f"message {i} not a dict"
        if msg.get("role") != want_role:
            return False, f"message {i} role={msg.get('role')!r}, expected {want_role!r}"
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            return False, f"message {i} ({want_role}) content empty or non-string"

    try:
        token_ids = tokenizer.apply_chat_template(messages)
    except Exception as exc:  # noqa: BLE001 — tokenizer can raise anything
        return False, f"tokenizer raised {type(exc).__name__}: {exc}"
    if not hasattr(token_ids, "__len__"):
        return False, "tokenizer.apply_chat_template did not return a sized object"

    n_tokens = len(token_ids)
    if n_tokens <= MIN_TOKENS:
        return False, f"too short: {n_tokens} tokens"
    if n_tokens >= MAX_TOKENS:
        return False, f"too long: {n_tokens} tokens"

    return True, None


# ---------------------------------------------------------------------------
# Tokenizer adapter
# ---------------------------------------------------------------------------


@dataclass
class TokenizerInfo:
    """Adapter exposing a uniform `apply_chat_template(messages) -> ids` plus
    a `render(messages) -> str` helper that returns the literal rendered chat.
    """

    name: str
    backend: str  # "hf" | "tiktoken_harmony" | "tiktoken_base"
    _hf_tokenizer: Any | None = None
    _tiktoken_encoding: Any | None = None

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        if self.backend == "hf":
            ids = self._hf_tokenizer.apply_chat_template(messages, tokenize=True)
            # transformers can return List[int] or a tensor — normalize.
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return list(ids)
        # tiktoken: render ChatML-ish manually, then encode
        rendered = self.render(messages)
        return self._tiktoken_encoding.encode(rendered, disallowed_special=())

    def render(self, messages: list[dict]) -> str:
        """Render a chat to its literal training-time string."""
        if self.backend == "hf":
            return self._hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        # Minimal ChatML rendering for tiktoken fallback. Matches the
        # widely-used <|im_start|>/<|im_end|> envelope so token counts are
        # comparable across runs even if Tinker reformats during training.
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        return "\n".join(parts)


def make_tokenizer() -> TokenizerInfo:
    """Build the best available tokenizer adapter.

    Order: HF `openai/gpt-oss-20b` (honoring HF_TOKEN) → tiktoken `o200k_harmony`
    → tiktoken `o200k_base`. The choice is logged via the returned `name`/`backend`.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "openai/gpt-oss-20b",
            token=os.environ.get("HF_TOKEN"),
        )
        if tok.chat_template is None:
            raise RuntimeError("HF tokenizer loaded but has no chat_template")
        return TokenizerInfo(
            name="openai/gpt-oss-20b",
            backend="hf",
            _hf_tokenizer=tok,
        )
    except Exception:
        pass

    import tiktoken

    for enc_name in ("o200k_harmony", "o200k_base"):
        try:
            enc = tiktoken.get_encoding(enc_name)
            return TokenizerInfo(
                name=f"tiktoken/{enc_name}",
                backend=f"tiktoken_{enc_name.split('_', 1)[1]}",
                _tiktoken_encoding=enc,
            )
        except Exception:
            continue

    raise RuntimeError("No tokenizer available (neither HF nor tiktoken)")
