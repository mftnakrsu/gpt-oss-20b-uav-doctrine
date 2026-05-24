# Data expansion — 2026-05-24

_Targeted corpus expansion beyond the 30-source Phase 1+2 baseline at
`data/raw/`. Goal: add UAV/UAS combat-doctrine and operational sources
to broaden M1 Q&A coverage and seed M3 scenario reasoning._

- **Project root**: `/Users/suleakarsu/Desktop/uav_combat/finetuneSLM`
- **Corpus root**: `data/raw/`
- **Layout decision**: per-source-id subdirectory (PDF + `manifest.json`),
  matching the existing 30-source layout — **not** a separate
  `expanded/` tree. Tier number lives as a field inside each
  expansion-era `manifest.json`. A consolidated
  `data/raw/manifest_expansion.jsonl` was also written for downstream
  tooling. See `[[feedback-data-layout]]` in auto-memory for the
  rationale (user correction on 2026-05-24).

## 1. Outcome

- **New documents downloaded**: **28**
- **New bytes**:                 **60.8 MB**
- **New pages**:                 **2,026**
- **Estimated new tokens (bytes/4 heuristic)**: ~15.9M
- **Estimated new tokens (refined, using existing-corpus
  bytes/token ratio of ~41:1)**: ~**1.5M**
- **Total corpus after expansion**: **58 documents / 158.8 MB**
  (existing 30 + 28 new)

The bytes/4 heuristic from the task spec is a useful budget proxy but
substantially over-counts real LLM tokens for PDFs (font streams,
metadata, embedded images inflate raw bytes). The existing corpus
inventory at `docs/journal/01-data-inventory.md` measured 98 MB → 2.37M
tokens via the GPT-OSS-20B tokenizer (ratio ≈ 24K tokens/MB). Applying
the same ratio to the 60.8 MB expansion yields **~1.5M tokens** — a
reasonable working estimate. We will re-run `inventory_data.py` after
this expansion to get the authoritative count.

## 2. Per-tier breakdown

| Tier | Description                       | Docs | Bytes (MB) | Pages |
|-----:|-----------------------------------|-----:|-----------:|------:|
| 1    | Official doctrine                 |   12 |       40.7 | 1,329 |
| 2    | Academic & war college            |    5 |        9.4 |   392 |
| 3    | Think tank reports                |    6 |        3.7 |   130 |
| 4    | Modern case studies               |    5 |        7.0 |   175 |
| **Σ**|                                   | **28** | **60.8** | **2,026** |

## 3. Top 20 by size

| Tier |   KB | Pages | Source ID                                | Title (truncated) |
|-----:|-----:|------:|------------------------------------------|-------------------|
| T1   | 14,118 | 158 | `usarmy_fm_3_60`                        | FM 3-60, Army Targeting |
| T1   |  5,540 | 192 | `uk_jdp_2_00`                           | JDP 2-00, Intelligence support to joint operations |
| T4   |  4,340 |  69 | `rusi_2022_preliminary_lessons_ukraine` | Preliminary Lessons from Russia's Invasion of Ukraine (RUSI) |
| T1   |  4,274 | 196 | `usarmy_atp_3_04_7`                     | ATP 3-04.7, Army Aviation Maintenance |
| T1   |  4,046 | 188 | `usarmy_fm_3_81`                        | FM 3-81, Maneuver Enhancement Brigade |
| T1   |  3,320 | 104 | `usarmy_atp_3_04_6`                     | ATP 3-04.6, Forward Arming and Refueling Points |
| T2   |  3,278 | 115 | `nps_uas_swarm_marines_2014`            | Swarming UAVs (MAGTF comms) — NPS thesis |
| T1   |  3,119 |  50 | `usarmy_atp_3_04_13`                    | ATP 3-04.13, Helicopter and Small Aircraft BDA |
| T2   |  2,688 |  87 | `dtic_5g_uav_swarm_2023`                | 5G-Inspired Method for UAV Swarm Ranging |
| T2   |  2,677 | 119 | `nps_swarm_vs_swarm_2013`               | Agent-Based Simulation of UAV Swarm-on-Swarm |
| T1   |  2,306 | 126 | `nato_ajp_3_9`                          | AJP-3.9, NATO Joint Targeting (Ed B V1) |
| T3   |  1,678 |  44 | `crs_2025_counter_uas_R48477`           | CRS R48477, DoD Counter-UAS |
| T1   |  1,428 |  84 | `uk_jdn_3_19`                           | JDN 3/19, UK Approach to UAS (archived) |
| T1   |  1,153 |  35 | `us_afdp_3_01`                          | AFDP 3-01, Counterair Operations |
| T1   |  1,008 |  64 | `us_afdp_3_03`                          | AFDP 3-03, Counterland Operations |
| T4   |    993 |  42 | `rusi_protecting_force_from_uas`        | Protecting the Force from Uncrewed Aerial Systems (RUSI) |
| T4   |    991 |  25 | `rusi_2025_tactical_developments_third_year` | Tactical Developments — Third Year of Russo-Ukrainian War (RUSI) |
| T1   |    719 |  50 | `nato_ajp_3_20`                         | AJP-3.20, NATO Cyberspace Operations (Ed A V1) |
| T3   |    708 |  22 | `icrc_2025_position_autonomous_weapons_ihl` | ICRC Position Paper (Oct 2025) — AWS and IHL |
| T2   |    613 |  17 | `hdiac_ukraine_lessons_2023`            | HDIAC Technological Lessons — Russia/Ukraine |

## 4. Sources considered but skipped

The following high-value targets were attempted and failed; each is
documented here so a future session knows the blocker (and whether to
try a proxy / different egress IP).

| Tier | Target                                              | Blocker | Action |
|-----:|-----------------------------------------------------|---------|--------|
| 1    | FM 3-04, Army Aviation (`ARN21797`)                 | armypubs returns 1.2 KB HTML stub — superseded or path changed | retry with current ARN (search `armypubs.army.mil` catalog) |
| 1    | FM 3-09, Fire Support (`ARN21932`)                  | same — 1.2 KB HTML stub | as above |
| 1    | FM 3-34, Engineer Ops (`ARN31353`)                  | same — 1.2 KB HTML stub | as above |
| 1    | AFDP 3-0.1 Command & Control                        | doctrine.af.mil → 403 (Akamai/WAF rule) | retry via Wayback dated snapshot or US-egress proxy |
| 1    | AFDP 3-60 Targeting                                 | doctrine.af.mil → 403 (intermittent — 3-01/3-03 succeeded) | as above |
| 1    | AFDP 3-30 (legacy C2)                               | doctrine.af.mil → 403 | as above |
| 1    | MCDP 1, 1-0, 1-1, 1-3 (USMC capstone × 4)           | marines.mil → 403 (Akamai bot block) | only US-egress fixes this |
| 1    | AJP-3.19 (CIMIC)                                    | `handbook.cimic-coe.org` DNS does not resolve | find official NSO mirror |
| 3    | CRS R42136, IF11426 (FAS mirror `sgp.fas.org`)      | Cloudflare WAF "challenge" — 202 with `x-amzn-waf-action: challenge` | retry via `crsreports.congress.gov` direct (also Cloudflare) or `everycrsreport.com` |
| 3    | CRS reports via `crsreports.congress.gov`           | Cloudflare bot mitigation (Sec-CH-UA challenge) | bypassing would violate task rules — skipped per policy |
| 4    | AUARC Occasional Paper 29 (Ukraine drones)          | `researchcentre.army.gov.au` read timeout (likely AU egress shapes non-AU traffic) | retry from AU/US egress |
| 4    | Military Review Sep-Oct 2025 "Lessons from Ukraine" | `armyupress.army.mil` → 403 (Akamai) | as above |
| —    | Wikipedia / news aggregators                        | per task rule 6 | excluded by policy |
| —    | Sci-Hub / LibGen / paywalled academic               | per task rule 1 | excluded by policy |
| —    | Anything FOUO / CUI / NOFORN                        | per task rule 5 | excluded by policy |

**Pattern across failures:** ~80% are Akamai or Cloudflare bot mitigation
on `.mil` and `.gov` PDF endpoints. The existing 30-source corpus was
fetched 2026-05-20 with the same downloader — those URLs all worked.
Conjecture: a WAF rule has tightened since, OR the egress IP from this
session has been flagged. The existing fetcher (`src/ingest/fetch.py`)
already documents this risk: _"Some .mil sites can be slow or
intermittently unreachable from non-US IP space. Try `--proxy` with a
US/EU egress."_ A clean way forward is to re-run the failed sources from
a US-egress proxy or staging host and rsync the new `<source_id>/`
directories back.

## 5. Search strategy & notes

Method:

1. Built `scripts/expand_corpus.py` mirroring `src/ingest/fetch.py`
   patterns: PDF magic-byte check, polite per-host rate limit (1.5–3.5s
   jitter), full Chrome User-Agent + `Referer`, redirect-following.
   Per-source `manifest.json` mirrors the existing schema and adds
   expansion-only fields (`title`, `author_or_publisher`,
   `date_published`, `source_tier`, `license_note`, `page_count_if_pdf`,
   `notes`).
2. Pre-flight: load every existing `manifest.json` SHA256 into a set;
   any newly downloaded PDF whose SHA256 collides is deleted and
   logged as `rejected_duplicate`. (No duplicates hit on this run.)
3. URL discovery: started from a curated list keyed on the project's
   existing source IDs (FM/ATP/AFDP/AJP numbers we know are
   UAS-relevant), then verified URLs via `web_search`. The first pass
   used pattern-extrapolated URLs (50%+ failure rate on guesses
   against `armypubs.army.mil` and `gov.uk` asset hashes); the
   second pass used only URLs surfaced by search results — the
   doc here reports the second pass.
4. **High-leverage finds** (non-obvious additions that materially
   change the corpus):
   - **FM 3-60 Army Targeting (Aug 2023, 158 pp)** — the sister to
     joint JP 3-60 (already in corpus). Gives Army-side D3A
     targeting context that joint doctrine does not cover at the
     corps-and-below echelon, directly grounding any armed-UAS
     strike-coordination reasoning.
   - **AJP-3.9 NATO Joint Targeting (Ed B V1, 126 pp)** — NATO
     analogue to JP 3-60; we now have all three (US joint, US Army,
     NATO) targeting cycles for cross-doctrine grounding.
   - **AJP-3.10 (Info Ops) and AJP-3.20 (Cyberspace Ops)** — covers
     the non-kinetic UAS employment surface (jamming, deception,
     cyber attack on UAS C2) that JP 3-85 only partially covers.
   - **AFDP 3-01 / 3-03 (Counterair / Counterland)** — the operational-
     level doctrine into which any armed-UAS employment plugs.
   - **ICRC 2026, 2025, and 2022 position papers on AWS** — three
     stable anchors for the Limitations / Ethics framing
     (`[[project-ethics-positioning]]`); explicit non-LAWS framing.
   - **RUSI Tactical Developments — Third Year of Russo-Ukrainian
     War** — empirical fieldwork data: "60–80% of Ukrainian FPVs
     fail to reach their target". Concrete grounding for what
     onboard-SLM UAS decision support must handle.
   - **NPS swarm-on-swarm thesis (ADA582919)** — agent-based
     simulation of swarm-vs-swarm decision-making, directly relevant
     to M3 multi-UAS coordination under degraded comms.

## 6. Full title list (for the eventual blog post bibliography)

### Tier 1 — Official doctrine (newly added)

- ATP 3-04.6, Techniques for Forward Arming and Refueling Points (US Army, 2022)
- ATP 3-04.7, Army Aviation Maintenance (US Army, 2020)
- ATP 3-04.13, Helicopter and Small Aircraft Battle Damage Assessment (US Army, 2021)
- AFDP 3-01, Counterair Operations (US Air Force, 2024)
- AFDP 3-03, Counterland Operations (US Air Force, 2022)
- AJP-3.9, Allied Joint Doctrine for Joint Targeting Ed B V1 (NATO, 2021)
- AJP-3.10, Allied Joint Doctrine for Information Operations (NATO, 2015)
- AJP-3.20, Allied Joint Doctrine for Cyberspace Operations Ed A V1 (NATO, 2020)
- FM 3-60, Army Targeting (US Army, 2023)
- FM 3-81, Maneuver Enhancement Brigade (US Army, 2021)
- JDN 3/19, UK Approach to Unmanned Aircraft Systems (archived) (UK MOD DCDC, 2017)
- JDP 2-00, Intelligence Support to Joint Operations Ed 4 (UK MOD DCDC, 2023)

### Tier 2 — Academic & war college (newly added)

- Agent-Based Simulation of a Defensive UAV Swarm vs Enemy UAV Swarm (NPS, 2013)
- Swarming UAVs: Extending MAGTF Communications (NPS, 2014)
- Drone Swarms (SAMS monograph, US Army, 2017)
- 5G-Inspired Method for Ranging of UAVs in Swarming Operations (DTIC, 2023)
- Technological Lessons Learned from Russia–Ukraine Conflict (HDIAC, 2023)

### Tier 3 — Think tank reports (newly added)

- Department of Defense Counter Unmanned Aircraft Systems (CRS R48477, 2025)
- Law Enforcement and the Evolving C-UAS Landscape (CRS IN12661, 2026)
- ICRC Position Paper: Autonomous Weapon Systems and IHL (ICRC, 2026)
- ICRC Position Paper Oct 2025: AWS and IHL — Selected Issues (ICRC, 2025)
- ICRC Position on Autonomous Weapon Systems (Int'l Review of the Red Cross, 2022)
- An Ethical Foundation for the Use of Autonomous Weapons (CNAS, 2015)

### Tier 4 — Modern case studies (newly added)

- Preliminary Lessons in Conventional Warfighting from Russia's Invasion of Ukraine, Feb–Jul 2022 (RUSI, 2022)
- Tactical Developments During the Third Year of the Russo-Ukrainian War (RUSI, 2025)
- Protecting the Force from Uncrewed Aerial Systems (RUSI, 2024)
- The 2022 Russo-Ukrainian War (US Army Infantry Magazine, 2023)
- The Drone Revolution: Lessons from Ukraine (Hugo Brown, LIIA Riga, 2025)

## 7. Next steps

1. Re-run `scripts/inventory_data.py` to refresh
   `docs/journal/01-data-inventory.md` against the new 58-doc total
   and get an authoritative tokenizer-based token count.
2. Re-fetch the 13 blocked sources from a US-egress proxy (per
   `src/ingest/fetch.py` `--proxy` flag) — these are real published
   PDFs we want, just blocked by host-level WAF from this session's
   egress.
3. Consider folding the new sources into `src/ingest/sources.py`
   (with proper `License` enum mapping) so the existing ingest
   pipeline can re-fetch them on demand instead of relying on
   `scripts/expand_corpus.py`. Open question: do we want the
   `source_tier` field in the existing pipeline's manifest too?

### Summary for Claude

```
Total new docs:    28
Total new size:    60.8 MB
Est. new tokens:   ~15.9M (bytes/4 heuristic) / ~1.5M (refined, bytes/40)
Per-tier counts:   T1=12  T2=5  T3=6  T4=5
Failures (blocked): 15 sources — 13 by Akamai/Cloudflare WAF on
                    .mil/.gov hosts, 1 DNS, 1 read-timeout. All real
                    public PDFs; re-fetchable from US-egress proxy.
Duplicates rejected by SHA256: 0
Corpus total now:  58 docs / 158.8 MB
Consolidated manifest: data/raw/manifest_expansion.jsonl (28 lines)
```
