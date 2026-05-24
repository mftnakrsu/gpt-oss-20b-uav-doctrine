"""Tests for chatml.to_chatml + validate_chatml.

A stub tokenizer keeps these tests independent of HF/tiktoken availability.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from finetune_slm.data.chatml import (  # noqa: E402
    MAX_TOKENS,
    to_chatml,
    validate_chatml,
)

SYSTEM = "You are a doctrine assistant."


class StubTokenizer:
    """Counts whitespace-separated tokens after rendering messages."""

    def apply_chat_template(self, messages):
        rendered = " ".join(m["content"] for m in messages)
        return rendered.split()


def _pair(**overrides):
    base = {
        "pair_id": "src__0001__00",
        "chunk_id": "src__0001",
        "source_filename": "src/x.pdf",
        "question_type": "factual",
        "question": (
            "What does the publication say about lost-link procedures "
            "for unmanned aerial systems in controlled airspace?"
        ),
        "answer": (
            "Per the publication, lost-link procedures specify a pre-briefed "
            "contingency route, a return-to-base profile, and ATC notification "
            "requirements. The remote pilot must squawk the assigned code and "
            "follow the published lost-link mission plan until link is "
            "reestablished or the recovery point is reached."
        ),
    }
    base.update(overrides)
    return base


def test_well_formed_pair_is_valid():
    example = to_chatml(_pair(), SYSTEM)
    ok, reason = validate_chatml(example, StubTokenizer())
    assert ok, reason
    assert reason is None
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert example["messages"][2]["role"] == "assistant"
    assert example["pair_id"] == "src__0001__00"


def test_missing_field_makes_to_chatml_raise():
    pair = _pair()
    del pair["question"]
    with pytest.raises(KeyError):
        to_chatml(pair, SYSTEM)


def test_empty_answer_is_invalid():
    example = to_chatml(_pair(answer=""), SYSTEM)
    ok, reason = validate_chatml(example, StubTokenizer())
    assert not ok
    assert "assistant" in reason and "empty" in reason


def test_whitespace_only_answer_is_invalid():
    example = to_chatml(_pair(answer="   \n   "), SYSTEM)
    ok, reason = validate_chatml(example, StubTokenizer())
    assert not ok
    assert "assistant" in reason and "empty" in reason


def test_overlong_answer_is_invalid():
    long_answer = ("word " * (MAX_TOKENS + 50)).strip()
    example = to_chatml(_pair(answer=long_answer), SYSTEM)
    ok, reason = validate_chatml(example, StubTokenizer())
    assert not ok
    assert "too long" in reason


def test_too_short_is_invalid():
    example = to_chatml(
        _pair(question="What?", answer="Yes."),
        "Sys.",
    )
    ok, reason = validate_chatml(example, StubTokenizer())
    assert not ok
    assert "too short" in reason


def test_role_order_must_be_system_user_assistant():
    example = to_chatml(_pair(), SYSTEM)
    example["messages"][1], example["messages"][2] = (
        example["messages"][2],
        example["messages"][1],
    )
    ok, reason = validate_chatml(example, StubTokenizer())
    assert not ok
    assert "role" in reason


def test_tokenizer_exception_is_reported_as_invalid():
    class Boom:
        def apply_chat_template(self, messages):
            raise RuntimeError("kaboom")

    example = to_chatml(_pair(), SYSTEM)
    ok, reason = validate_chatml(example, Boom())
    assert not ok
    assert "RuntimeError" in reason
