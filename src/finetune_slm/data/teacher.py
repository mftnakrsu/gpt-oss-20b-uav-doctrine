"""Azure OpenAI LLM client (Foundry-projects endpoint).

Wraps ``openai.AsyncOpenAI`` with:
- ``base_url`` pointing at the Foundry-projects path (the API version is
  encoded in the URL, **never** passed as a query parameter)
- ``api-key`` header auth (Azure's preferred shape) instead of Bearer
- tenacity retry: 3 attempts, exp backoff 4-30s, retry on
  RateLimitError / APITimeoutError / APIConnectionError / 5xx
- model-aware fallback: newer reasoning-family deployments accept
  ``max_completion_tokens``; legacy ones still want ``max_tokens``. We
  default to the newer name and fall back if the API rejects it.

The API key is read from ``AZURE_API_KEY`` and never echoed anywhere.
"""

from __future__ import annotations

import os
from typing import Optional

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class AzureLLMClient:
    """Async LLM client for Azure Foundry-hosted OpenAI deployments."""

    def __init__(self) -> None:
        self.api_key = os.environ["AZURE_API_KEY"]
        self.base_url = os.environ["AZURE_OPENAI_LLM_BASE_URL"]
        self.deployment = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]
        self.api_version = os.environ.get("AZURE_OPENAI_LLM_API_VERSION", "")
        # Some deployments (gpt-5 / o1 family) pin temperature to 1.0; honor
        # the env override so callers don't have to know.
        self.default_temperature = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
        self.default_max_tokens = int(os.environ.get("LLM_MAX_COMPLETION_TOKENS", "2000"))
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={"api-key": self.api_key},
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
        ),
        reraise=True,
    )
    async def chat_completion(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, dict]:
        """Return ``(content, usage)``.

        ``usage`` is a dict with ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``.
        """
        temp = self.default_temperature if temperature is None else temperature
        max_tok = self.default_max_tokens if max_tokens is None else max_tokens

        kwargs: dict = {
            "model": self.deployment,
            "messages": messages,
            "temperature": temp,
            "max_completion_tokens": max_tok,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            msg = str(e).lower()
            # Older deployments still want max_tokens — translate and retry once.
            if "max_completion_tokens" in msg or "unrecognized request argument" in msg:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                resp = await self._client.chat.completions.create(**kwargs)
            else:
                raise

        content = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        return content, usage

    async def close(self) -> None:
        await self._client.close()
