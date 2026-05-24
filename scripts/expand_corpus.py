"""
Expansion downloader for the UAV/UAS doctrine corpus.

Adds NEW source-id directories under `data/raw/`, following the same
per-source layout the existing ingest pipeline uses (PDF + manifest.json).
Each manifest.json carries the existing fields plus expansion-only fields
(title, author_or_publisher, date_published, source_tier, license_note,
page_count_if_pdf, notes).

Every URL in this registry was verified via web_search/web_fetch in
session 2026-05-24 against the publisher's live page. The PDF magic-byte
check rejects HTML stubs / WAF challenges automatically.

Use:
    uv run python scripts/expand_corpus.py --tier 1
    uv run python scripts/expand_corpus.py --all
    uv run python scripts/expand_corpus.py --source-id usarmy_atp_3_04_7

Stops if the new sources alone reach 250 docs, 2 GB, or all tiers are done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT = httpx.Timeout(90.0, connect=30.0)
MIN_INTERVAL_PER_HOST = 1.5
MAX_PER_HOST_JITTER = 2.0

BUDGET_DOCS = 250
BUDGET_BYTES = 2 * 1024 * 1024 * 1024


@dataclass
class NewSource:
    id: str
    title: str
    author_or_publisher: str
    date_published: str
    url: str
    mirror_urls: list[str] = field(default_factory=list)
    license_note: str = ""
    source_tier: int = 1
    notes: str = ""


SOURCES: list[NewSource] = [
    # =========================================================================
    # TIER 1 — OFFICIAL DOCTRINE
    # =========================================================================

    # --- US Army Field Manuals & ATPs (UAS-adjacent, real ARN URLs) ---
    NewSource(
        id="usarmy_fm_3_04",
        title="FM 3-04, Army Aviation",
        author_or_publisher="US Army Headquarters",
        date_published="2020-04",
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/pdf/web/ARN21797_FM_3-04_FINAL_WEB_wfix.pdf",
        license_note="US Government work — public domain (17 U.S.C. § 105)",
        source_tier=1,
        notes="Capstone Army aviation doctrine; UAS integration framework.",
    ),
    NewSource(
        id="usarmy_fm_3_09",
        title="FM 3-09, Fire Support and Field Artillery Operations",
        author_or_publisher="US Army Headquarters",
        date_published="2020-04",
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/pdf/web/ARN21932_FM_3-09_FINAL_WEB.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="Army fire support; armed UAS / loitering munition coordination.",
    ),
    NewSource(
        id="usarmy_fm_3_34",
        title="FM 3-34, Engineer Operations",
        author_or_publisher="US Army Headquarters",
        date_published="2020-12",
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN31353-FM_3-34-000-WEB-1.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="Engineer ops; UAS for terrain assessment and obstacle planning.",
    ),
    NewSource(
        id="usarmy_fm_3_60",
        title="FM 3-60, Army Targeting",
        author_or_publisher="US Army Headquarters",
        date_published="2023-08",
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN39048-FM_3-60-000-WEB-1.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="Army targeting (D3A) — sister to joint JP 3-60. Sets the context "
              "for armed UAS strike coordination at corps and below.",
    ),
    NewSource(
        id="usarmy_fm_3_81",
        title="FM 3-81, Maneuver Enhancement Brigade",
        author_or_publisher="US Army Headquarters",
        date_published="2021-11",
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN34192-FM_3-81-000-WEB-1.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="MEB includes ADA / C-UAS at brigade level.",
    ),
    NewSource(
        id="usarmy_atp_3_04_6",
        title="ATP 3-04.6, Techniques for Forward Arming and Refueling Points",
        author_or_publisher="US Army Headquarters",
        date_published="2022-01",
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN35738-ATP_3-04.6-001-WEB-2.pdf",
        license_note="US Government work — Distribution A",
        source_tier=1,
        notes="Aviation FARP procedures, sustainment context for UAS operations.",
    ),
    NewSource(
        id="usarmy_atp_3_04_7",
        title="ATP 3-04.7, Army Aviation Maintenance",
        author_or_publisher="US Army Headquarters",
        date_published="2020-10",
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN31028-ATP_3-04.7-000-WEB-1.pdf",
        license_note="US Government work — Distribution A",
        source_tier=1,
        notes="Aviation maintenance with explicit UAS coverage; sustainment "
              "doctrine for the GCS/airframe side of an SLM-equipped UAS.",
    ),
    NewSource(
        id="usarmy_atp_3_04_13",
        title="ATP 3-04.13, Helicopter and Small Aircraft Battle Damage Assessment",
        author_or_publisher="US Army Headquarters",
        date_published="2021-11",
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN34174-ATP_3-04.13-000-WEB-1.pdf",
        license_note="US Government work — Distribution A",
        source_tier=1,
        notes="Battle damage assessment for rotary-wing and small aircraft "
              "(extends to small UAS) — relevant for lost-link-following "
              "damage-state reasoning.",
    ),

    # --- US Air Force Doctrine Publications (verified doctrine.af.mil URLs) ---
    NewSource(
        id="us_afdp_3_01",
        title="AFDP 3-01, Counterair Operations",
        author_or_publisher="US Air Force (LeMay Center)",
        date_published="2024-03",
        url="https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-01/3-01-AFDP-COUNTERAIR.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="USAF counter-air, includes C-UAS at the operational level.",
    ),
    NewSource(
        id="us_afdp_3_03",
        title="AFDP 3-03, Counterland Operations",
        author_or_publisher="US Air Force (LeMay Center)",
        date_published="2022-09",
        url="https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-03/3-03-AFDP-COUNTERLAND.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="Counterland: CAS, AI, armed-UAS employment, kill-box doctrine.",
    ),
    NewSource(
        id="us_afdp_3_0_1",
        title="AFDP 3-0.1, Command and Control",
        author_or_publisher="US Air Force (LeMay Center)",
        date_published="2025-01",
        url="https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-0_1/AFDP3-0.1CommandandControl.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="Current C2 doctrine (Jan 2025) — replaces AFDP 3-30. Establishes "
              "Centralized Command-Distributed Control-Decentralized Execution "
              "framework: directly relevant to delegated UAS decision-making.",
    ),
    NewSource(
        id="us_afdp_3_60",
        title="AFDP 3-60, Targeting",
        author_or_publisher="US Air Force (LeMay Center)",
        date_published="2021-11",
        url="https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-60/3-60-AFDP-TARGETING.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="USAF targeting doctrine, F2T2EA cycle.",
    ),
    NewSource(
        id="us_afdp_3_30_legacy",
        title="AFDP 3-30, Command and Control (legacy, pre-3-0.1)",
        author_or_publisher="US Air Force (LeMay Center)",
        date_published="2020-04",
        url="https://www.doctrine.af.mil/portals/61/documents/afdp_3-30/afdp%203-30-command-and-control.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="Prior C2 doctrine, retained on doctrine.af.mil as a legacy URL. "
              "Useful for comparing pre/post 3-0.1 C2 framing.",
    ),

    # --- US Marines doctrine (verified marines.mil URLs) ---
    NewSource(
        id="usmc_mcdp_1",
        title="MCDP 1, Warfighting",
        author_or_publisher="US Marine Corps",
        date_published="2018-04",
        url="https://www.marines.mil/Portals/1/Publications/MCDP%201%20Warfighting.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="USMC capstone warfighting philosophy. The doctrinal source for "
              "mission command and decentralized initiative — directly load-"
              "bearing for the project's onboard-SLM thesis.",
    ),
    NewSource(
        id="usmc_mcdp_1_0",
        title="MCDP 1-0, Marine Corps Operations (w/Change 1-3)",
        author_or_publisher="US Marine Corps",
        date_published="2017-08",
        url="https://www.marines.mil/Portals/1/Publications/MCDP%201-0%20w%20Ch%201-3.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="MAGTF operations capstone.",
    ),
    NewSource(
        id="usmc_mcdp_1_1",
        title="MCDP 1-1, Strategy",
        author_or_publisher="US Marine Corps",
        date_published="1997-11",
        url="https://www.marines.mil/Portals/1/Publications/MCDP%201-1%20Strategy.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="USMC strategy doctrine.",
    ),
    NewSource(
        id="usmc_mcdp_1_3",
        title="MCDP 1-3, Tactics",
        author_or_publisher="US Marine Corps",
        date_published="1997-07",
        url="https://www.marines.mil/Portals/1/Publications/MCDP%201-3%20Tactics.pdf",
        license_note="US Government work — public domain",
        source_tier=1,
        notes="USMC tactics doctrine; tactical decision-making and OODA.",
    ),

    # --- NATO AJPs publicly released via UK MOD ---
    NewSource(
        id="nato_ajp_3_9",
        title="AJP-3.9, Allied Joint Doctrine for Joint Targeting (Edition B Version 1)",
        author_or_publisher="NATO Standardization Office (mirrored via UK MOD)",
        date_published="2021-03",
        url="https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1033306/AJP-3.9_EDB_V1_E.pdf",
        license_note="NATO Unclassified — releasable to the public; UK OGL mirror",
        source_tier=1,
        notes="NATO joint targeting; full-spectrum F2T2EA including cyber & space.",
    ),
    NewSource(
        id="nato_ajp_3_10",
        title="AJP-3.10, Allied Joint Doctrine for Information Operations",
        author_or_publisher="NATO Standardization Office",
        date_published="2015-11",
        url="https://info.publicintelligence.net/NATO-IO.pdf",
        license_note="NATO Unclassified — releasable to the public",
        source_tier=1,
        notes="NATO information operations; relevant for UAS as information-"
              "operation enablers (loudspeakers, leaflet drop, ISR).",
    ),
    NewSource(
        id="nato_ajp_3_19",
        title="AJP-3.19, Allied Joint Doctrine for Civil-Military Cooperation (Edition A Version 1)",
        author_or_publisher="NATO Standardization Office",
        date_published="2018-11",
        url="https://www.handbook.cimic-coe.org/8.-annex/reference-docs/ajp-3.19-eda-v1-e.pdf",
        license_note="NATO Unclassified — releasable to the public; CIMIC COE mirror",
        source_tier=1,
        notes="CIMIC frame relevant to UAS in populated areas / civilian harm "
              "considerations.",
    ),
    NewSource(
        id="nato_ajp_3_20",
        title="AJP-3.20, Allied Joint Doctrine for Cyberspace Operations (Edition A Version 1)",
        author_or_publisher="NATO Standardization Office",
        date_published="2020-01",
        url="https://iwar.org.uk/wp-content/uploads/2021/06/AJP-3.20-EDA-V1-E.pdf",
        license_note="NATO Unclassified — releasable to the public",
        source_tier=1,
        notes="NATO cyberspace ops; cyber attack vectors against UAS C2.",
    ),

    # --- UK doctrine (additional) ---
    NewSource(
        id="uk_jdp_2_00",
        title="JDP 2-00, Intelligence, Counter-intelligence and Security Support to Joint Operations (Ed 4)",
        author_or_publisher="UK MOD DCDC",
        date_published="2023-08",
        url="https://assets.publishing.service.gov.uk/media/653a4b0780884d0013f71bb0/JDP_2_00_Ed_4_web.pdf",
        license_note="UK Open Government Licence v3.0",
        source_tier=1,
        notes="UK intelligence support to joint ops, includes ISR / UAS tasking.",
    ),
    NewSource(
        id="uk_jdn_3_19",
        title="JDN 3/19, UK Approach to Unmanned Aircraft Systems (archived)",
        author_or_publisher="UK MOD DCDC",
        date_published="2017-08",
        url="https://web.archive.org/web/2022/https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/673940/doctrine_uk_uas_jdp_0_30_2.pdf",
        license_note="UK Open Government Licence v3.0 (Wayback)",
        source_tier=1,
        notes="UK approach to UAS — historical doctrinal record.",
    ),

    # =========================================================================
    # TIER 2 — ACADEMIC & WAR COLLEGE (verified DTIC AD numbers)
    # =========================================================================

    NewSource(
        id="nps_swarm_vs_swarm_2013",
        title="Agent-Based Simulation of a Defensive UAV Swarm Against an Enemy UAV Swarm (NPS thesis)",
        author_or_publisher="Naval Postgraduate School (DTIC)",
        date_published="2013-09",
        url="https://apps.dtic.mil/sti/tr/pdf/ADA582919.pdf",
        license_note="Distribution A: approved for public release; unlimited",
        source_tier=2,
        notes="Swarm-on-swarm decision-making with autonomous agents — directly "
              "relevant to multi-UAS coordination under degraded comms.",
    ),
    NewSource(
        id="nps_uas_swarm_marines_2014",
        title="Swarming Unmanned Aerial Vehicles: Extending MAGTF Communications Using UAVs (NPS thesis)",
        author_or_publisher="Naval Postgraduate School (DTIC)",
        date_published="2014-12",
        url="https://apps.dtic.mil/sti/pdfs/AD1069733.pdf",
        license_note="Distribution A",
        source_tier=2,
    ),
    NewSource(
        id="dtic_drone_swarms_sams_2017",
        title="Drone Swarms (SAMS monograph) — MAJ Andrew William Sanders, US Army",
        author_or_publisher="School of Advanced Military Studies (DTIC)",
        date_published="2017-05",
        url="https://apps.dtic.mil/sti/pdfs/AD1039921.pdf",
        license_note="Distribution A",
        source_tier=2,
        notes="SAMS-level operational analysis of swarm warfare.",
    ),
    NewSource(
        id="dtic_5g_uav_swarm_2023",
        title="5G-Inspired Method for Ranging of UAVs in Swarming Operations",
        author_or_publisher="DTIC technical report",
        date_published="2023-03",
        url="https://apps.dtic.mil/sti/trecms/pdf/AD1164247.pdf",
        license_note="Distribution A",
        source_tier=2,
        notes="UAV swarm ranging via 5G primitives — useful for the comms/"
              "GNSS-denied environment the project's lost-link target assumes.",
    ),
    NewSource(
        id="hdiac_ukraine_lessons_2023",
        title="Technological Lessons Learned from the Conflict Between Russia and Ukraine (HDIAC/DTIC)",
        author_or_publisher="Homeland Defense and Security Information Analysis Center (DTIC)",
        date_published="2023-02",
        url="https://hdiac.dtic.mil/wp-content/uploads/2023/02/TI-Response-Report_HDIAC_Technological-Lessons-Learned-from-the-Conflict-Between-Russia-and-Ukraine_2212023.pdf",
        license_note="Distribution A",
        source_tier=2,
    ),

    # =========================================================================
    # TIER 3 — THINK TANK (FAS mirror for CRS, ICRC, Hudson, Atlantic Council)
    # =========================================================================

    NewSource(
        id="crs_2024_dod_uas_R42136",
        title="U.S. Unmanned Aerial Systems (CRS R42136, FAS mirror)",
        author_or_publisher="Congressional Research Service",
        date_published="2012-01",
        url="https://sgp.fas.org/crs/natsec/R42136.pdf",
        license_note="US Government work — public domain (CRS); FAS public mirror",
        source_tier=3,
        notes="Baseline CRS overview of DoD UAS programs.",
    ),
    NewSource(
        id="crs_2025_counter_uas_IF11426",
        title="DoD Counter-Unmanned Aircraft Systems (CRS IF11426, FAS mirror)",
        author_or_publisher="Congressional Research Service",
        date_published="2025-03",
        url="https://sgp.fas.org/crs/weapons/IF11426.pdf",
        license_note="US Government work — public domain; FAS public mirror",
        source_tier=3,
    ),
    NewSource(
        id="crs_2025_counter_uas_R48477",
        title="Department of Defense Counter Unmanned Aircraft Systems (CRS R48477)",
        author_or_publisher="Congressional Research Service",
        date_published="2025-03",
        url="https://www.congress.gov/crs_external_products/R/PDF/R48477/R48477.1.pdf",
        license_note="US Government work — public domain",
        source_tier=3,
        notes="Mar 2025 in-depth report on DoD C-UAS strategy and acquisition.",
    ),
    NewSource(
        id="crs_2026_law_enforcement_cuas_IN12661",
        title="Law Enforcement and the Evolving C-UAS Landscape (CRS IN12661, everycrsreport mirror)",
        author_or_publisher="Congressional Research Service",
        date_published="2026-02",
        url="https://www.everycrsreport.com/files/2026-02-27_IN12661_287865abb1cd221dda4fa6e56fa00fae90a7f86a.pdf",
        license_note="US Government work — public domain",
        source_tier=3,
    ),
    NewSource(
        id="icrc_2026_position_autonomous_weapons",
        title="ICRC Position Paper: Autonomous Weapon Systems and International Humanitarian Law",
        author_or_publisher="International Committee of the Red Cross",
        date_published="2026-03",
        url="https://www.icrc.org/sites/default/files/2026-03/4896_002_Autonomous_Weapons_Systems_-_IHL-ICRC.pdf",
        license_note="ICRC — published for public distribution",
        source_tier=3,
        notes="ICRC framing of IHL constraints on AWS. Direct anchor for the "
              "ethics section of the project paper (NOT a LAWS paper framing).",
    ),
    NewSource(
        id="icrc_2025_position_autonomous_weapons_ihl",
        title="ICRC Position Paper (Oct 2025): Autonomous Weapon Systems and IHL — Selected Issues",
        author_or_publisher="International Committee of the Red Cross",
        date_published="2025-10",
        url="https://www.icrc.org/sites/default/files/media_file/2025-10/ICRC-Position_Paper-Autonomous_Weapon_Systems_and_IHL-Selected_issues_Oct2025.pdf",
        license_note="ICRC — published for public distribution",
        source_tier=3,
    ),
    NewSource(
        id="icrc_2022_position_autonomous_weapons_review",
        title="ICRC Position on Autonomous Weapon Systems (International Review of the Red Cross PDF, 2022)",
        author_or_publisher="International Review of the Red Cross / ICRC",
        date_published="2022-01",
        url="https://international-review.icrc.org/sites/default/files/reviews-pdf/2022-01/icrc-position-on-autonomous-weapon-systems-icrc-position-and-background-paper-915.pdf",
        license_note="ICRC — published for public distribution",
        source_tier=3,
    ),
    NewSource(
        id="cnas_2015_ethical_autonomy",
        title="An Ethical Foundation for the Use of Autonomous Weapons (CNAS working paper)",
        author_or_publisher="Center for a New American Security",
        date_published="2015-02",
        url="http://files.cnas.org/documents/Ethical-Autonomy-Working-Paper_021015_v02.pdf",
        license_note="CNAS — free download with attribution",
        source_tier=3,
        notes="Scharre/Sayler foundational paper on autonomous-weapon ethics.",
    ),

    # =========================================================================
    # TIER 4 — MODERN CASE STUDIES (Ukraine, drone warfare lessons)
    # =========================================================================

    NewSource(
        id="rusi_2022_preliminary_lessons_ukraine",
        title="Preliminary Lessons in Conventional Warfighting from Russia's Invasion of Ukraine (Feb-Jul 2022) — RUSI",
        author_or_publisher="Royal United Services Institute",
        date_published="2022-11",
        url="https://static.rusi.org/359-SR-Ukraine-Preliminary-Lessons-Feb-July-2022-web-final.pdf",
        license_note="RUSI — free download for non-commercial use",
        source_tier=4,
    ),
    NewSource(
        id="rusi_2025_tactical_developments_third_year",
        title="Tactical Developments During the Third Year of the Russo-Ukrainian War — RUSI",
        author_or_publisher="Royal United Services Institute",
        date_published="2025-02",
        url="https://static.rusi.org/tactical-developments-third-year-russo-ukrainian-war-february-2205.pdf",
        license_note="RUSI — free download for non-commercial use",
        source_tier=4,
        notes="Fieldwork-based analysis of FPV/UAS efficacy; notes only 20-40% "
              "of FPV strikes reach target — concrete grounding for what UAS "
              "decision support needs to handle.",
    ),
    NewSource(
        id="rusi_protecting_force_from_uas",
        title="Protecting the Force from Uncrewed Aerial Systems — RUSI",
        author_or_publisher="Royal United Services Institute",
        date_published="2024-07",
        url="https://static.rusi.org/protecting-the-force-from-uncrewed-uas.pdf",
        license_note="RUSI — free download for non-commercial use",
        source_tier=4,
    ),
    NewSource(
        id="auarc_2024_drones_lessons_ukraine",
        title="Drones in Modern Warfare: Lessons Learnt from the War in Ukraine (Occasional Paper 29)",
        author_or_publisher="Australian Army Research Centre",
        date_published="2024-10",
        url="https://researchcentre.army.gov.au/sites/default/files/241022-Occasional-Paper-29-Lessons-Learnt-from-Ukraine_2.pdf",
        license_note="Commonwealth of Australia — published for free download",
        source_tier=4,
    ),
    NewSource(
        id="armyupress_2025_lessons_ukraine",
        title="Lessons from Ukraine (Military Review Sep-Oct 2025) — UA-focused volume",
        author_or_publisher="US Army University Press / Military Review",
        date_published="2025-09",
        url="https://www.armyupress.army.mil/Portals/7/military-review/Archives/English/so25/Lessons-from-Ukraine/Lessons-from-Ukraine-ua.pdf",
        license_note="US Government work — public domain",
        source_tier=4,
    ),
    NewSource(
        id="benning_infantry_mag_2023_russo_ukraine",
        title="The 2022 Russo-Ukrainian War (US Army Infantry Magazine 2023)",
        author_or_publisher="US Army Maneuver Center of Excellence (Fort Moore/Benning)",
        date_published="2023-07",
        url="https://www.benning.army.mil/infantry/magazine/issues/2023/Summer/pdf/13_Wilkins_txt.pdf",
        license_note="US Government work — public domain",
        source_tier=4,
    ),
    NewSource(
        id="liia_2025_drone_revolution",
        title="The Drone Revolution: Lessons from Ukraine — Hugo Brown (LIIA, Riga, 2025)",
        author_or_publisher="Latvian Institute of International Affairs",
        date_published="2025-09",
        url="https://liia.lv/en/publications/the-drone-revolution-lessons-from-ukraine-1476?get_file=1",
        license_note="LIIA — published for free download",
        source_tier=4,
    ),
]


# -----------------------------------------------------------------------------
# Pre-flight: existing SHA256s to avoid duplicates
# -----------------------------------------------------------------------------

def existing_sha256_set() -> set[str]:
    s: set[str] = set()
    for m in DATA_RAW.glob("*/manifest.json"):
        try:
            d = json.loads(m.read_text())
            if d.get("sha256"):
                s.add(d["sha256"])
        except Exception:
            continue
    return s


# -----------------------------------------------------------------------------
# Polite per-host rate limiter
# -----------------------------------------------------------------------------

_last_hit: dict[str, float] = {}


def polite_wait(host: str) -> None:
    now = time.monotonic()
    delta = now - _last_hit.get(host, 0.0)
    target = MIN_INTERVAL_PER_HOST + random.random() * MAX_PER_HOST_JITTER
    if delta < target:
        time.sleep(target - delta)
    _last_hit[host] = time.monotonic()


# -----------------------------------------------------------------------------
# Download one source
# -----------------------------------------------------------------------------

def page_count(pdf_path: Path) -> Optional[int]:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path), strict=False).pages)
    except Exception:
        return None


def try_download(client: httpx.Client, url: str, dest: Path) -> tuple[int, str, int]:
    host = urlparse(url).netloc
    polite_wait(host)
    headers = {"Referer": f"https://{host}/"}
    with client.stream("GET", url, follow_redirects=True, headers=headers) as resp:
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "").lower()
        if "html" in ct or "json" in ct or ct.startswith("text/"):
            raise RuntimeError(f"non-PDF content-type ({ct or 'missing'})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        first = True
        with dest.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                if first:
                    if not chunk.startswith(b"%PDF-"):
                        dest.unlink(missing_ok=True)
                        raise RuntimeError(f"missing PDF magic ({chunk[:8]!r})")
                    first = False
                f.write(chunk)
                size += len(chunk)
        if size == 0:
            dest.unlink(missing_ok=True)
            raise RuntimeError("zero bytes")
        return size, str(resp.url), resp.status_code


def fetch_source(
    client: httpx.Client,
    src: NewSource,
    existing_hashes: set[str],
) -> dict:
    out_dir = DATA_RAW / src.id
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            existing = out_dir / m["filename"]
            if existing.exists() and existing.stat().st_size > 0:
                return {"status": "skipped_existing", "source_id": src.id,
                        "bytes": existing.stat().st_size,
                        "tier": src.source_tier}
        except Exception:
            pass

    candidates = [src.url] + list(src.mirror_urls)
    last_err: Optional[str] = None

    for i, url in enumerate(candidates):
        filename = Path(urlparse(url).path).name or f"{src.id}.pdf"
        if not filename.lower().endswith(".pdf"):
            filename = filename.rstrip("/") + ".pdf"
        dest = out_dir / filename
        try:
            size, final_url, status = try_download(client, url, dest)
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code} from {url}"
            continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} (url={url})"
            continue

        sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
        if sha256 in existing_hashes:
            dest.unlink(missing_ok=True)
            try:
                out_dir.rmdir()
            except OSError:
                pass
            return {"status": "rejected_duplicate", "source_id": src.id,
                    "sha256": sha256, "url": url}

        pages = page_count(dest)
        manifest = {
            "source_id": src.id,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "url_attempted": url,
            "final_url": final_url,
            "http_status": status,
            "filename": filename,
            "bytes": size,
            "sha256": sha256,
            "license": "expanded_corpus",
            "used_mirror": i > 0,
            "title": src.title,
            "author_or_publisher": src.author_or_publisher,
            "date_published": src.date_published,
            "source_tier": src.source_tier,
            "license_note": src.license_note,
            "page_count_if_pdf": pages,
            "notes": src.notes,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        existing_hashes.add(sha256)
        return {"status": "ok", "source_id": src.id, "bytes": size,
                "tier": src.source_tier, "pages": pages,
                "used_mirror": i > 0, "sha256": sha256}

    return {"status": "failed", "source_id": src.id, "error": last_err,
            "tier": src.source_tier}


def run(sources: list[NewSource]) -> list[dict]:
    existing = existing_sha256_set()
    new_bytes = 0
    new_docs = 0
    results: list[dict] = []

    with httpx.Client(
        timeout=TIMEOUT,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        },
    ) as client:
        for src in sources:
            if new_docs >= BUDGET_DOCS or new_bytes >= BUDGET_BYTES:
                results.append({"status": "budget_stop", "source_id": src.id,
                                "tier": src.source_tier})
                break
            r = fetch_source(client, src, existing)
            results.append(r)
            if r["status"] == "ok":
                new_bytes += r["bytes"]
                new_docs += 1
            status_str = r["status"]
            extra = ""
            if r["status"] == "ok":
                extra = f"  ({r['bytes']/1024:.0f} KB, {r.get('pages') or '?'} pages)"
            elif r["status"] == "failed":
                extra = f"  err={r['error']}"
            elif r["status"] == "rejected_duplicate":
                extra = f"  sha256={r['sha256'][:12]}"
            elif r["status"] == "skipped_existing":
                extra = f"  ({r['bytes']/1024:.0f} KB on disk)"
            print(f"[t{src.source_tier}] {src.id:<42} {status_str}{extra}")
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None)
    p.add_argument("--source-id", type=str, default=None)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.source_id:
        sel = [s for s in SOURCES if s.id == args.source_id]
    elif args.tier is not None:
        sel = [s for s in SOURCES if s.source_tier == args.tier]
    elif args.all:
        sel = list(SOURCES)
    else:
        p.error("specify --tier N | --source-id ID | --all")

    if not sel:
        print("no sources matched the filter", file=sys.stderr)
        return 1

    print(f"Fetching {len(sel)} sources...")
    results = run(sel)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped_existing"]
    duped = [r for r in results if r["status"] == "rejected_duplicate"]
    failed = [r for r in results if r["status"] == "failed"]
    budget = [r for r in results if r["status"] == "budget_stop"]

    total = sum(r["bytes"] for r in ok) + sum(r.get("bytes", 0) for r in skipped)
    print("=" * 60)
    print(f"OK: {len(ok)}   SKIPPED: {len(skipped)}   "
          f"DUPED: {len(duped)}   FAILED: {len(failed)}   "
          f"BUDGET-STOP: {len(budget)}")
    print(f"Total bytes (new + skipped existing): {total/1024/1024:.1f} MB")
    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  {r['source_id']}: {r.get('error')}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
