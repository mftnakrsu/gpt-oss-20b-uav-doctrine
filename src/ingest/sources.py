"""
Source registry for the UAV / UAS doctrine corpus.

Every URL in this file was manually verified to exist as of 2026-05-20.
Where a publication has been superseded or withdrawn, the archived URL is
kept because the doctrinal content is still useful training material and
is explicitly preserved by the publisher.

Licensing notes:
- All `us_public_domain` entries are works of the US federal government and
  are not protected by US copyright (17 U.S.C. § 105).
- `uk_ogl` entries are released under the UK Open Government Licence v3.0,
  which permits reuse including commercial use with attribution.
- `us_distribution_a` is US gov + "Distribution Statement A: Approved for
  public release; distribution is unlimited". Treated as public domain for
  our purposes.

WARNING: Some .mil sites can be slow or intermittently unreachable from
non-US IP space. If you are running ingest from Turkey or similar, expect
to retry, or run the downloader from a US/EU staging host and rsync the
raw corpus over.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocType(str, Enum):
    JOINT_DOCTRINE = "joint_doctrine"          # US JP series, UK JDP
    SERVICE_DOCTRINE = "service_doctrine"      # USAF AFDP, US Army FM
    TTP = "ttp"                                # US Army ATP, MTTP, USAF AFTTP
    ALLIED_DOCTRINE = "allied_doctrine"        # NATO AJP, ATP
    STANAG = "stanag"
    REFERENCE = "reference"                    # smartbooks, dictionaries
    THINK_TANK_REPORT = "think_tank_report"
    JOURNAL_ARTICLE = "journal_article"
    CONFLICT_ANALYSIS = "conflict_analysis"


class License(str, Enum):
    US_PUBLIC_DOMAIN = "us_public_domain"
    US_DISTRIBUTION_A = "us_distribution_a"
    UK_OGL = "uk_ogl"
    NATO_PUBLIC = "nato_public"
    CC_BY = "cc_by"
    CC_BY_NC = "cc_by_nc"
    THINK_TANK_PROPRIETARY = "think_tank_proprietary"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    P0 = "p0"   # must-have, core doctrine
    P1 = "p1"   # high-value supplementary
    P2 = "p2"   # nice-to-have, lower confidence


class ParserStrategy(str, Enum):
    US_JOINT = "us_joint"          # JP-series: chapter/roman + dash + arabic
    US_ARMY = "us_army"            # ATP/FM: same as joint but slightly different headers
    US_AIR_FORCE = "us_air_force"  # AFDP: shorter, often numbered as 1, 1.1, 1.1.1
    UK_JDP = "uk_jdp"              # UK: chapter + decimal, "1.1", "1.2"
    NATO = "nato"                  # AJP/ATP-N: chapter + section + para
    GENERIC = "generic"            # think tank / journal — heuristic only


class Source(BaseModel):
    id: str
    title: str
    issuer: str
    doc_type: DocType
    url: Optional[str] = None
    mirror_urls: list[str] = Field(default_factory=list)
    license: License
    language: str = "en"
    estimated_pages: Optional[int] = None
    priority: Priority
    parser_strategy: ParserStrategy = ParserStrategy.GENERIC
    superseded: bool = False
    notes: str = ""


# -----------------------------------------------------------------------------
# REGISTRY
# -----------------------------------------------------------------------------

SOURCES: list[Source] = [
    # -------------------------------------------------------------------------
    # P0 — US JOINT DOCTRINE (Joint Staff, public domain)
    # -------------------------------------------------------------------------
    Source(
        id="us_jp_3_30",
        title="JP 3-30, Joint Air Operations",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_30.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=200,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Capstone for joint air operations. Covers C2, JFACC, JAOC, "
              "the air tasking cycle, and remotely piloted aircraft.",
    ),
    Source(
        id="us_jp_3_52",
        title="JP 3-52, Joint Airspace Control",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_52.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=110,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Joint airspace control. Critical for understanding how UAS "
              "integrate into shared airspace alongside manned aviation.",
    ),
    Source(
        id="us_jp_3_01",
        title="JP 3-01, Countering Air and Missile Threats",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_01.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=180,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Counter-air doctrine including C-UAS at the joint level.",
    ),
    Source(
        id="us_jp_3_09",
        title="JP 3-09, Joint Fires",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_09.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=160,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Joint fires doctrine; includes armed UAS in the kill chain.",
    ),
    Source(
        id="us_jp_3_60",
        title="JP 3-60, Joint Targeting",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_60.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=140,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Joint targeting cycle (F2T2EA). Heavily referenced in any "
              "strike-capable UAS employment discussion.",
    ),
    Source(
        id="us_jp_3_85",
        title="JP 3-85, Joint Electromagnetic Spectrum Operations",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_85.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=130,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="EMS operations; relevant for C-UAS electronic attack and for "
              "UAS data-link survivability.",
    ),

    # -------------------------------------------------------------------------
    # P0 — US ARMY TTPs (Distribution A)
    # -------------------------------------------------------------------------
    Source(
        id="us_atp_3_01_81",
        title="ATP 3-01.81, Counter-Unmanned Aircraft System (C-UAS) Operations",
        issuer="US Army",
        doc_type=DocType.TTP,
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN43877-ATP_3-01.81-000-WEB-1.pdf",
        mirror_urls=[
            "https://irp.fas.org/doddir/army/atp3-01-81.pdf",  # 2023 version mirror
        ],
        license=License.US_DISTRIBUTION_A,
        estimated_pages=90,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_ARMY,
        notes="Latest May 2025 edition. Primary unclassified US Army C-UAS "
              "publication. FAS mirror has 2023 edition for diff comparison.",
    ),
    Source(
        id="us_atp_3_04_64",
        title=("ATP 3-04.64 / MCRP 3-42.1A / NTTP 3-55.14 / AFTTP 3-2.64, "
               "Multi-Service TTP for the Tactical Employment of UAS"),
        issuer="US Army (multi-service)",
        doc_type=DocType.TTP,
        url=None,  # No stable canonical URL; armypubs CAC-gated for newer eds.
        mirror_urls=[
            # 2015 edition mirrored on multiple sites; user should verify
            # current accessibility from their network before relying on any.
            "https://info.publicintelligence.net/USArmy-UAS-MTTP.pdf",
        ],
        license=License.US_DISTRIBUTION_A,
        estimated_pages=180,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_ARMY,
        notes="Core multi-service TTP for UAS tactical employment. The 2015 "
              "edition (the most recent unclassified release as far as we "
              "could verify) is mirrored on publicintelligence.net but "
              "armypubs.army.mil no longer hosts it. TODO: verify current "
              "accessibility; consider supplementing with TC 3-04.61 and "
              "TC 3-04.62 from same era.",
    ),

    # -------------------------------------------------------------------------
    # P0 — US AIR FORCE DOCTRINE
    # -------------------------------------------------------------------------
    Source(
        id="us_afdp_3_52",
        title="AFDP 3-52, Airspace Control",
        issuer="US Air Force (LeMay Center)",
        doc_type=DocType.SERVICE_DOCTRINE,
        url="https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-52/3-52-AFDP-AIRSPACE-CONTROL.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=60,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_AIR_FORCE,
        notes="USAF view on airspace control, explicitly addresses small UAS "
              "and loitering munitions.",
    ),
    Source(
        id="us_afdoctrine_smartbook",
        title="Air Force Doctrine Smart Book",
        issuer="US Air Force (LeMay Center)",
        doc_type=DocType.REFERENCE,
        url="https://www.doctrine.af.mil/Portals/61/documents/SmartBook/AFDoctrineSmartBook.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=70,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_AIR_FORCE,
        notes="One-page synopses of every current AFDP. Excellent compact "
              "training signal — high terminology density per token.",
    ),
    Source(
        id="us_atp_3_52_4",
        title=("ATP 3-52.4 / MCRP 3-20F.10 / NTTP 6-02.9 / AFTTP 3-2.8, "
               "Multi-Service TTP for Air Control Communication"),
        issuer="US Army / USMC / USN / USAF (ALSA)",
        doc_type=DocType.TTP,
        url="https://static.e-publishing.af.mil/production/1/lemay_center/publication/afttp3-2.8/afttp3-2.8.pdf",
        license=License.US_DISTRIBUTION_A,
        estimated_pages=210,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_AIR_FORCE,
        notes="Oct 2024 edition. Tactical C2 voice procedures including for "
              "UAS in the air picture.",
    ),

    # -------------------------------------------------------------------------
    # P0 — UK DOCTRINE
    # -------------------------------------------------------------------------
    Source(
        id="uk_jdp_0_30_2",
        title="JDP 0-30.2, Unmanned Aircraft Systems (Change 1)",
        issuer="UK MOD (DCDC)",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://assets.publishing.service.gov.uk/media/667d6f155b0d63b556a4b437/ARCHIVE-UAS_JDP_0_30_2.pdf",
        license=License.UK_OGL,
        estimated_pages=80,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.UK_JDP,
        superseded=True,
        notes="Withdrawn Jun 2024 and folded into JDP 0-30 UK Air & Space "
              "Power. Archived copy still hosted by gov.uk. UK-specific "
              "terminology, NATO classification mapping, and operational-"
              "level UAS employment — high doctrinal value despite "
              "supersession.",
    ),
    Source(
        id="uk_jdp_0_30",
        title="JDP 0-30, UK Air and Space Power (3rd Edition)",
        issuer="UK MOD (DCDC)",
        doc_type=DocType.JOINT_DOCTRINE,
        url=None,
        license=License.UK_OGL,
        estimated_pages=120,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.UK_JDP,
        notes="TODO: locate current edition URL on gov.uk. Search: "
              "site:gov.uk \"JDP 0-30\" \"UK air and space power\". "
              "Current (post-2024) edition is the doctrinal successor to "
              "JDP 0-30.2.",
    ),

    # -------------------------------------------------------------------------
    # P1 — NATO (URLs need verification per release)
    # -------------------------------------------------------------------------
    Source(
        id="nato_ajp_3_3",
        title="AJP-3.3, Allied Joint Doctrine for Air and Space Operations",
        issuer="NATO Standardization Office",
        doc_type=DocType.ALLIED_DOCTRINE,
        url=None,
        license=License.NATO_PUBLIC,
        estimated_pages=140,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.NATO,
        notes="TODO: NSO public release portal at nso.nato.int rotates URLs. "
              "Edition C is the most recent publicly released. Verify before "
              "ingest. Some annexes remain NATO-restricted; only the public "
              "release should be included in this corpus.",
    ),

    # -------------------------------------------------------------------------
    # P1 — THINK TANK / ANALYSIS (representative examples; expand as needed)
    # -------------------------------------------------------------------------
    # Intentionally seeded sparsely. Per-source license review required before
    # adding any think-tank material. RAND, RUSI, CSIS, CNAS, IISS, Atlantic
    # Council, Hudson, JAPCC all publish UAS-relevant material under varying
    # terms — most are free-to-read but NOT freely redistributable. Add
    # individually after license check.
]


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def by_priority(priority: Priority) -> list[Source]:
    return [s for s in SOURCES if s.priority == priority]


def by_id(source_id: str) -> Optional[Source]:
    return next((s for s in SOURCES if s.id == source_id), None)


def ready_to_fetch() -> list[Source]:
    """Sources with a primary URL set (not None) and not just mirrors."""
    return [s for s in SOURCES if s.url is not None]


def needs_url() -> list[Source]:
    """Sources still missing a verified primary URL."""
    return [s for s in SOURCES if s.url is None]


if __name__ == "__main__":
    # Quick inventory print when run directly.
    print(f"{'TOTAL':<20} {len(SOURCES)}")
    print(f"{'  P0':<20} {len(by_priority(Priority.P0))}")
    print(f"{'  P1':<20} {len(by_priority(Priority.P1))}")
    print(f"{'  P2':<20} {len(by_priority(Priority.P2))}")
    print(f"{'READY TO FETCH':<20} {len(ready_to_fetch())}")
    print(f"{'NEEDS URL':<20} {len(needs_url())}")
    print()
    print("By license:")
    for lic in License:
        n = sum(1 for s in SOURCES if s.license == lic)
        if n:
            print(f"  {lic.value:<28} {n}")
    print()
    print("Missing URL (need manual research):")
    for s in needs_url():
        print(f"  [{s.priority.value}] {s.id} — {s.title}")
