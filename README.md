# UAV Doctrine Corpus — Real Data Phase

Working repository for assembling a public-domain English-language UAV /
UAS doctrine corpus to fine-tune Qwen 2.5 7B.

## What's in here right now

This is Phase 1+2 of the larger pipeline (see ../PROMPTS.md for the full
9-phase plan). Just the source registry and the downloader, with **real,
manually verified URLs** as of 2026-05-20.

```
src/ingest/
  sources.py    14 doctrine sources, 11 with verified primary URLs
  fetch.py      async downloader with retry + rate limiting + manifests
data/raw/       (created on first fetch)
```

## Source inventory

| Priority | Count | Notes |
|---|---|---|
| P0 — US Joint doctrine | 6 | JP 3-30, 3-52, 3-01, 3-09, 3-60, 3-85 |
| P0 — US Service doctrine / TTP | 5 | AFDP 3-52, AFDP Smart Book, ATP 3-01.81, ATP 3-04.64, ATP 3-52.4 |
| P0 — UK doctrine | 2 | JDP 0-30.2 (archived but valuable), JDP 0-30 (URL TODO) |
| P1 — NATO | 1 | AJP-3.3 (URL TODO — NSO portal changes) |

License breakdown:
- 8 × US public domain (Joint Staff publications)
- 3 × US Distribution Statement A (Army & multi-service TTPs)
- 2 × UK Open Government Licence v3.0
- 1 × NATO public release (pending URL)

All 11 ready-to-fetch sources are redistributable under their respective
public terms. Nothing in this set is think-tank-proprietary.

## Quick start

```bash
# install
pip install -e .

# see what we have
python -m src.ingest.sources

# pull the P0 set (will take ~10-20 min, mostly throttled by per-host rate limit)
python -m src.ingest.fetch --priority p0

# single source
python -m src.ingest.fetch --source-id us_jp_3_30

# if you're behind a Turkish ISP and .mil hosts are slow/blocked:
python -m src.ingest.fetch --priority p0 --proxy http://user:pass@eu-host:3128
```

After fetch, `data/raw/<source_id>/` will contain the PDF plus a
`manifest.json` with sha256, fetched_at, final_url, http_status, license.

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

## Three sources still need URLs

These were intentionally left without primary URLs because verification
required interactive browsing or NSO portal access:

1. **ATP 3-04.64** — the multi-service UAS MTTP. The 2015 edition is
   mirrored on `info.publicintelligence.net` and that's wired in as a
   mirror, so it will still download. If you have access to a newer
   edition through armypubs (CAC-gated), substitute that URL.
2. **JDP 0-30** — UK Air and Space Power, current edition. Find via
   `site:gov.uk "JDP 0-30" "UK air and space power"`.
3. **AJP-3.3** — NATO Allied Joint Doctrine for Air and Space Operations,
   current public release. NSO portal: `nso.nato.int`.
