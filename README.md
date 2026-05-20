# UAV Doctrine Corpus — Phase 1+2 (Ingest)

Working repository for assembling a public-domain English-language UAV / UAS
doctrine corpus to fine-tune Qwen 2.5 7B. This README covers the ingest
layer; the corpus serves as input to later phases (clean → chunk → synth →
CPT → SFT → eval → serve) which live outside this repo.

## What's in here

```
src/ingest/
  sources.py    31 doctrine sources with verified URLs + per-source
                license, edition, parser strategy, and notes
  fetch.py      Async httpx downloader with per-host rate limit,
                tenacity retry, content-type / PDF-magic guard,
                mirror fallback, and SHA-256 manifests
data/raw/       30 retrieved PDFs (LFS-tracked) + 30 manifest.json
.gitattributes  Git LFS configuration for data/raw/**/*.pdf
NOTICE          Per-source attributions and license terms
```

## Corpus state (as of last successful ingest)

| Metric | Value |
|---|---|
| Sources registered | 31 |
| Successfully retrieved | 30 |
| Total corpus size | ~108 MB |
| Estimated pages | ~4,451 |
| Estimated tokens | ~1.78M raw doctrine |
| Coverage | 97% by source count, ~96% by pages |
| Unrecoverable gaps | 1 (ATP 3-04.64, officially RESCINDED) |

License distribution: 19 × US public domain, 4 × NATO public, 2 × US
Distribution A, 2 × UK OGL v3.0, 2 × EASA public, 1 × ICAO public.

See `NOTICE` for full per-source attributions.

## Source inventory by priority

**P0 — Core doctrine (22 sources, 21 retrieved):**
- US Joint: JP 2-0, JP 3-0, JP 3-01, JP 3-09, JP 3-30, JP 3-52, JP 3-60, JP 3-85, JP 5-0 (1 retrieved is 2017 ed. due to edition gap)
- US Army: ADP 3-0 (Mar 2025), ADP 6-0 (Mission Command), ATP 3-01.81 (C-UAS), ATP 3-52.4
- US Air Force: AFDP 3-0 (Jan 2025), AFDP 3-52, Air Force Doctrine Smart Book
- US Policy / Reference: DoDD 3000.09 (autonomy directive), DoD Dictionary
- UK: JDP 0-30 (Air Power, 3rd ed.), JDP 0-30.2 (UAS archived)
- NATO: AJP-3, AJP-3.3
- Not retrieved: ATP 3-04.64 (RESCINDED)

**P1 — Supplementary (9 sources, 9 retrieved):**
- NATO: AJP-3.3.5 (Airspace), STO-EN-SCI-271/03 (STANAG 4586 proxy)
- US Joint (supplementary): JP 3-13.1 (Electronic Warfare)
- Civil aviation lost-link anchors: FAA JO 7110.65 BB, FAA Notice 7110.724 (UAS Lost Link), EASA EAR for UAS, EASA SAIL III MoC OSO#6 (C2 Link), ICAO RPAS CONOPS

## Quick start

```bash
# install
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# clone with LFS so binaries arrive too
# (if you cloned before installing git-lfs, run: git lfs pull)

# see what's in the registry
python -m src.ingest.sources

# re-fetch (succeeded sources are skipped via manifest cache)
python -m src.ingest.fetch --priority p0
python -m src.ingest.fetch --priority p1
python -m src.ingest.fetch --all       # both

# single source
python -m src.ingest.fetch --source-id us_jp_3_85

# force re-download (ignores manifest cache)
python -m src.ingest.fetch --all --force

# if running from a network where .mil / FAS are blocked,
# the registry already points at Wayback / academic mirrors for those.
# For the remaining ATP 3-04.64 gap, no public mirror exists.
```

After fetch, `data/raw/<source_id>/` contains the PDF plus a
`manifest.json` with sha256, fetched_at, final_url, http_status, license.

## Network notes

This registry was built from a Turkish IP, which surfaced a non-trivial
amount of geo-fenced .mil hosts. The verified mirror strategy uses, in
order of preference: original publisher → Wayback Machine snapshot →
allied gov mirror → academic / training mirror → industry nonprofit
mirror → leak-aggregator mirror (only for Distribution A content). See
`NOTICE` for the full mirror infrastructure list.

If you are running ingest from a US/EU egress, the original .mil URLs in
each source's `mirror_urls` field will succeed directly and you may
prefer to swap them to primary. From other geographies, leave the
registry as-is.

## Content-type guard

`fetch.py` validates every download with three checks before writing
the manifest:

1. HTTP status must be 200 (rejects 202 WAF queue responses)
2. `Content-Type` must not be `text/html`, `application/json`, or `text/*`
3. The body's first chunk must start with `%PDF-` magic bytes

This avoids saving Cloudflare WAF challenge HTML, FAA dynamic redirect
pages, or empty-body proxy responses as if they were valid PDFs.

## What this is NOT (yet)

This intentionally stops at "raw PDFs on disk with provenance metadata."
The next phases (clean → chunk → synth → CPT → SFT → eval → serve) are
specified in `../PROMPTS.md` and should be built phase-by-phase against
this corpus.

## Provenance & licensing posture

- Every entry in `sources.py` carries an explicit `license` field.
- Manifests preserve the original URL, final URL after redirects, http
  status, and content hash — required for any later audit.
- Mirrors are listed as fallbacks only; primary URL is always preferred.
- For sources where no public URL could be verified, the entry stays in
  the registry with `url=None` and a `notes` field explaining what to
  search for — no fabricated URLs.
- Edition gaps (JP 3-0, JP 3-60) and the one unrecoverable source
  (ATP 3-04.64) are documented per-source in `sources.py`.
