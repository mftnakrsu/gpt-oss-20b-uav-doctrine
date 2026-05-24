"""Azure-hosted Cohere reranker client (direct httpx, not the cohere SDK).

The Azure Foundry provider path doesn't match the cohere SDK's expected
shape, so we POST directly. The endpoint is::

    POST {AZURE_RERANKER_BASE_URL}{AZURE_RERANKER_PATH}
    headers: api-key, content-type
    body: {model, query, documents, top_n}

Two operational guardrails read from env (with safe defaults):

- ``AZURE_RERANKER_MIN_INTERVAL_S`` — minimum gap between request starts.
  The Sweden Central deployment is heavily rate-limited; the operator's
  reference card sets this to ~15s. We serialize requests with a lock
  and a per-instance ``_last_request_time`` so concurrency higher up the
  stack still respects this floor.
- ``AZURE_RERANKER_MAX_DOC_CHARS`` — documents are truncated to this
  many characters before being sent. Cohere v4 rerank has a per-document
  token cap and our doctrine chunks are 1k+ tokens, so without
  truncation many calls 4xx.

The API key is read from ``AZURE_API_KEY`` and never echoed anywhere.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class _RetryableHTTPError(Exception):
    """Marker for 429 / 5xx responses; matched by tenacity."""


class AzureCohereReranker:
    """Async client for the Cohere v2/rerank endpoint on Azure Foundry."""

    def __init__(self) -> None:
        self.api_key = os.environ["AZURE_API_KEY"]
        self.base_url = os.environ["AZURE_RERANKER_BASE_URL"].rstrip("/")
        path = os.environ["AZURE_RERANKER_PATH"]
        if not path.startswith("/"):
            path = "/" + path
        self.path = path
        self.deployment = os.environ["AZURE_RERANKER_DEPLOYMENT"]
        self.min_interval_s = float(
            os.environ.get("AZURE_RERANKER_MIN_INTERVAL_S", "1.0")
        )
        self.max_doc_chars = int(
            os.environ.get("AZURE_RERANKER_MAX_DOC_CHARS", "1000")
        )
        self._url = f"{self.base_url}{self.path}"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def _respect_min_interval(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self.min_interval_s - (now - self._last_request_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_time = asyncio.get_event_loop().time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(_RetryableHTTPError),
        reraise=True,
    )
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """Rerank ``documents`` against ``query``.

        Returns a list of ``{"index": int, "relevance_score": float}``
        sorted by ``relevance_score`` desc. ``index`` is the index in
        the original ``documents`` list (NOT the truncated copy).
        """
        if not documents:
            return []
        truncated = [d[: self.max_doc_chars] for d in documents]
        if top_n is None:
            top_n = len(truncated)
        top_n = max(1, min(top_n, len(truncated)))

        await self._respect_min_interval()

        try:
            resp = await self._client.post(
                self._url,
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.deployment,
                    "query": query,
                    "documents": truncated,
                    "top_n": top_n,
                },
            )
        except httpx.RequestError as e:
            raise _RetryableHTTPError(f"network: {type(e).__name__}: {e}") from e

        if resp.status_code == 429 or resp.status_code >= 500:
            # Body may include rate-limit advice; cap length to keep logs tidy.
            raise _RetryableHTTPError(
                f"HTTP {resp.status_code}: {resp.text[:300]}"
            )
        if resp.status_code != 200:
            # Non-retryable 4xx — surface the API's error message.
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code}: {resp.text[:300]}",
                request=resp.request,
                response=resp,
            )

        data = resp.json()
        results = data.get("results", [])
        return sorted(results, key=lambda r: -r["relevance_score"])

    async def close(self) -> None:
        await self._client.aclose()
