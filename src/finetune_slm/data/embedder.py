"""Azure OpenAI embedding client (text-embedding-3-small).

Wraps ``openai.AsyncOpenAI`` against the Azure ``/openai/v1`` legacy
path (different base URL from the LLM deployment). Used for batched
embedding requests and for the cosine-similarity quality score we
substituted for the Cohere reranker — see
``docs/journal/05a-qa-pilot.md`` for the rationale.

The API key is read from ``AZURE_API_KEY`` and never echoed anywhere.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard cosine similarity in [-1, 1]. Returns 0.0 for either vector
    being all-zero to avoid div-by-zero.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class AzureEmbeddingClient:
    """Async embedding client for an Azure-hosted text-embedding deployment."""

    def __init__(self) -> None:
        self.api_key = os.environ["AZURE_API_KEY"]
        self.base_url = os.environ["AZURE_OPENAI_EMBEDDING_BASE_URL"]
        self.deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
        self.expected_dim = int(
            os.environ.get("AZURE_OPENAI_EMBEDDING_DIMENSION", "1536")
        )
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
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, preserving order.

        Batches the whole list into a single API call. text-embedding-3-small
        supports up to ~8k tokens per item and many items per request.
        """
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            input=texts,
            model=self.deployment,
        )
        # The response preserves input order in resp.data[i].embedding.
        return [item.embedding for item in resp.data]

    async def close(self) -> None:
        await self._client.close()
