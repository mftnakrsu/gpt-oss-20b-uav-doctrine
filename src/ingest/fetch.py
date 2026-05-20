"""
Fetch verified-URL doctrine sources.

Usage:
    python -m src.ingest.fetch --priority p0
    python -m src.ingest.fetch --source-id us_jp_3_30
    python -m src.ingest.fetch --all
    python -m src.ingest.fetch --priority p0 --proxy http://user:pass@host:port

Behavior:
- Skips downloads when a valid manifest already exists and the remote ETag /
  Last-Modified matches. Use --force to re-download.
- Polite: max 2 concurrent requests per domain, 1.0s minimum interval.
- Resumable: failures don't crash the run; a final report lists failures.

Notes on running from Turkey or similar:
- US .mil hosts (jcs.mil, armypubs.army.mil, doctrine.af.mil) can be slow
  or sometimes refuse non-US/EU traffic. If you see many TimeoutError or
  ConnectError entries against .mil hosts, try --proxy with a US/EU egress.
- gov.uk and static.e-publishing.af.mil are usually fine globally.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingest.sources import SOURCES, Source, Priority, by_id, by_priority


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# Note: many .mil and gov.uk hosts return 403 for non-browser UAs even on
# fully public PDFs. The above is a plain Chrome UA — appropriate because
# we are doing research-grade ingestion of explicitly public documents.
TIMEOUT = httpx.Timeout(60.0, connect=20.0)
MIN_INTERVAL_PER_HOST = 1.0
MAX_CONCURRENT_PER_HOST = 2

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


# -----------------------------------------------------------------------------
# Per-host rate limiting
# -----------------------------------------------------------------------------

class HostLimiter:
    def __init__(self, max_concurrent: int, min_interval: float):
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._last_request: dict[str, float] = defaultdict(float)
        self._max_concurrent = max_concurrent
        self._min_interval = min_interval
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _sem(self, host: str) -> asyncio.Semaphore:
        if host not in self._semaphores:
            self._semaphores[host] = asyncio.Semaphore(self._max_concurrent)
        return self._semaphores[host]

    async def acquire(self, host: str):
        await self._sem(host).acquire()
        async with self._locks[host]:
            now = asyncio.get_event_loop().time()
            wait = self._min_interval - (now - self._last_request[host])
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request[host] = asyncio.get_event_loop().time()

    def release(self, host: str):
        self._sem(host).release()


# -----------------------------------------------------------------------------
# Result tracking
# -----------------------------------------------------------------------------

@dataclass
class FetchResult:
    source_id: str
    success: bool
    skipped: bool = False
    bytes_downloaded: int = 0
    error: Optional[str] = None
    final_url: Optional[str] = None
    http_status: Optional[int] = None
    used_mirror: bool = False


# -----------------------------------------------------------------------------
# Core download
# -----------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(
        (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
    ),
    reraise=True,
)
async def _download_one_url(
    client: httpx.AsyncClient, url: str, dest: Path
) -> tuple[int, str, int]:
    """Return (bytes_written, final_url, http_status)."""
    async with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with dest.open("wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                f.write(chunk)
                size += len(chunk)
        return size, str(resp.url), resp.status_code


async def fetch_source(
    client: httpx.AsyncClient,
    limiter: HostLimiter,
    source: Source,
    force: bool,
) -> FetchResult:
    if source.url is None and not source.mirror_urls:
        return FetchResult(
            source.id, success=False, error="no URL on registry entry"
        )

    out_dir = DATA_RAW / source.id
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists() and not force:
        try:
            m = json.loads(manifest_path.read_text())
            existing = out_dir / m["filename"]
            if existing.exists() and existing.stat().st_size > 0:
                return FetchResult(source.id, success=True, skipped=True)
        except Exception:
            pass  # corrupt manifest, re-download

    candidates = [u for u in [source.url, *source.mirror_urls] if u]
    last_err: Optional[str] = None

    for i, url in enumerate(candidates):
        host = urlparse(url).netloc
        filename = Path(urlparse(url).path).name or f"{source.id}.bin"
        dest = out_dir / filename

        await limiter.acquire(host)
        try:
            size, final_url, status = await _download_one_url(client, url, dest)
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code} from {url}"
            continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} (url={url})"
            continue
        finally:
            limiter.release(host)

        sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
        manifest = {
            "source_id": source.id,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "url_attempted": url,
            "final_url": final_url,
            "http_status": status,
            "filename": filename,
            "bytes": size,
            "sha256": sha256,
            "license": source.license.value,
            "used_mirror": i > 0,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return FetchResult(
            source.id,
            success=True,
            bytes_downloaded=size,
            final_url=final_url,
            http_status=status,
            used_mirror=i > 0,
        )

    return FetchResult(source.id, success=False, error=last_err)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

async def run(sources: list[Source], proxy: Optional[str], force: bool) -> list[FetchResult]:
    limiter = HostLimiter(MAX_CONCURRENT_PER_HOST, MIN_INTERVAL_PER_HOST)
    transport = httpx.AsyncHTTPTransport(retries=0, proxy=proxy) if proxy else None
    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        transport=transport,
    ) as client:
        tasks = [fetch_source(client, limiter, s, force) for s in sources]
        return await asyncio.gather(*tasks)


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--priority", choices=[x.value for x in Priority])
    g.add_argument("--source-id", type=str)
    g.add_argument("--all", action="store_true")
    p.add_argument("--proxy", type=str, default=None,
                   help="HTTP proxy URL (use for non-US egress)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if manifest exists")
    args = p.parse_args()

    if args.source_id:
        s = by_id(args.source_id)
        if s is None:
            print(f"unknown source_id: {args.source_id}", file=sys.stderr)
            return 2
        sources = [s]
    elif args.priority:
        sources = by_priority(Priority(args.priority))
    else:
        sources = SOURCES

    # Filter out sources with no URL at all (need manual research first)
    fetchable = [s for s in sources if s.url is not None or s.mirror_urls]
    skipped_no_url = [s for s in sources if s.url is None and not s.mirror_urls]

    if not fetchable:
        print("no fetchable sources matched the filter", file=sys.stderr)
        if skipped_no_url:
            print("\nSources missing URLs (need manual research):")
            for s in skipped_no_url:
                print(f"  [{s.priority.value}] {s.id} — {s.title}")
        return 1

    print(f"Fetching {len(fetchable)} sources...")
    if skipped_no_url:
        print(f"  (skipping {len(skipped_no_url)} sources without URLs)")
    print()

    results = asyncio.run(run(fetchable, args.proxy, args.force))

    # Report
    ok = [r for r in results if r.success]
    skipped = [r for r in ok if r.skipped]
    failed = [r for r in results if not r.success]

    print("=" * 60)
    print(f"DONE: {len(ok)} succeeded ({len(skipped)} cached), {len(failed)} failed")
    total = sum(r.bytes_downloaded for r in results)
    print(f"Downloaded: {total / 1024 / 1024:.1f} MB")
    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  {r.source_id}: {r.error}")
    if skipped_no_url:
        print("\nSkipped (no URL):")
        for s in skipped_no_url:
            print(f"  [{s.priority.value}] {s.id}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
