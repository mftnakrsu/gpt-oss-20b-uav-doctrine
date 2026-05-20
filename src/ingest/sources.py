"""
Source registry for the UAV / UAS doctrine corpus.

Every URL in this file was manually verified to exist (last expansion
2026-05-20). Where a publication has been superseded or withdrawn, the
archived URL is kept because the doctrinal content is still useful training
material and is explicitly preserved by the publisher.

Licensing notes:
- All `us_public_domain` entries are works of the US federal government and
  are not protected by US copyright (17 U.S.C. § 105).
- `uk_ogl` entries are released under the UK Open Government Licence v3.0,
  which permits reuse including commercial use with attribution.
- `us_distribution_a` is US gov + "Distribution Statement A: Approved for
  public release; distribution is unlimited". Treated as public domain for
  our purposes.
- `nato_public` is NATO Unclassified, Releasable to the Public — text reuse
  with attribution is permitted; PDF redistribution mirrored via UK MOD.
- `icao_public` covers ICAO documents that ICAO has freely released to the
  public (typically CONOPS, working group outputs). Does NOT cover paywalled
  ICAO Annexes or paywalled manuals (e.g. Doc 10019).
- `easa_public` covers EASA outputs released for free public use with
  attribution (Easy Access Rules, Means of Compliance, Decisions).

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
    SERVICE_DOCTRINE = "service_doctrine"      # USAF AFDP, US Army FM/ADP
    TTP = "ttp"                                # US Army ATP, MTTP, USAF AFTTP
    ALLIED_DOCTRINE = "allied_doctrine"        # NATO AJP, ATP
    STANAG = "stanag"
    REFERENCE = "reference"                    # dictionaries, smartbooks, regs
    THINK_TANK_REPORT = "think_tank_report"
    JOURNAL_ARTICLE = "journal_article"
    CONFLICT_ANALYSIS = "conflict_analysis"


class License(str, Enum):
    US_PUBLIC_DOMAIN = "us_public_domain"
    US_DISTRIBUTION_A = "us_distribution_a"
    UK_OGL = "uk_ogl"
    NATO_PUBLIC = "nato_public"
    ICAO_PUBLIC = "icao_public"
    EASA_PUBLIC = "easa_public"
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
    US_ARMY = "us_army"            # ATP/FM/ADP: same as joint but slightly different headers
    US_AIR_FORCE = "us_air_force"  # AFDP: shorter, often numbered as 1, 1.1, 1.1.1
    UK_JDP = "uk_jdp"              # UK: chapter + decimal, "1.1", "1.2"
    NATO = "nato"                  # AJP/ATP-N/STO: chapter + section + para
    CIVIL_AVIATION = "civil_aviation"  # FAA Order / EASA EAR / ICAO CONOPS
    GENERIC = "generic"            # think tank / journal / refs — heuristic only


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
        id="us_jp_3_0",
        title="JP 3-0, Joint Operations (2017 edition — superseded by 2022 'Joint Campaigns and Operations')",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://web.archive.org/web/2024/https://www.safety.marines.mil/Portals/92/Ground%20Safety%20for%20Marines%20(GSM)/References%20Tab/JP%203-0%20Joint%20Operations%20PDF.pdf",
        mirror_urls=[
            "https://www.safety.marines.mil/Portals/92/Ground%20Safety%20for%20Marines%20(GSM)/References%20Tab/JP%203-0%20Joint%20Operations%20PDF.pdf",
            "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_0.pdf",
        ],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=200,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        superseded=True,
        notes="2017 edition (titled 'Joint Operations'). EDITION GAP: "
              "the current 2022 edition (retitled 'Joint Campaigns and "
              "Operations') has no public mirror — only paywalled (Scribd, "
              "CourseHero, B&N reprint) or a CAC-gated JDEIS portal. "
              "Primary URL is the Wayback snapshot of marines.mil/safety's "
              "training reference mirror (2024, 2.36 MB application/pdf, "
              "verified May 2026). Doctrinal core (commander's intent, "
              "branches/sequels) is stable across the edition gap; the "
              "2022 rewrite focused on Joint All-Domain Operations framing.",
    ),
    Source(
        id="us_jp_5_0",
        title="JP 5-0, Joint Planning",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://web.archive.org/web/20220715183645/https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp5_0.pdf",
        mirror_urls=[
            "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp5_0.pdf",
            "https://www.esd.whs.mil/Portals/54/Documents/FOID/Reading%20Room/Joint_Staff/18-F-1152_JP_5-0_Joint_Planning_2020.pdf",
        ],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=220,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="December 2020 edition. Joint planning — branches, sequels, "
              "contingency planning. Essential grounding for any 'what would "
              "the commander want?' inference. Primary URL is the verified "
              "Wayback snapshot (Jul 2022); .mil URL kept as mirror for "
              "US-egress runs.",
    ),
    Source(
        id="us_jp_3_30",
        title="JP 3-30, Joint Air Operations",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://web.archive.org/web/20241202080205/https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_30.pdf",
        mirror_urls=["https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_30.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=200,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Capstone for joint air operations. Covers C2, JFACC, JAOC, "
              "the air tasking cycle, and remotely piloted aircraft. Primary "
              "URL is the verified Wayback snapshot (Dec 2024); .mil URL "
              "kept as mirror for US-egress runs.",
    ),
    Source(
        id="us_jp_3_52",
        title="JP 3-52, Joint Airspace Control",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://web.archive.org/web/2021/https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_52.pdf",
        mirror_urls=["https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_52.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=110,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Joint airspace control. Critical for understanding how UAS "
              "integrate into shared airspace alongside manned aviation. "
              "Primary URL is the verified Wayback snapshot (2021); .mil "
              "URL kept as mirror for US-egress runs.",
    ),
    Source(
        id="us_jp_3_01",
        title="JP 3-01, Countering Air and Missile Threats",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://web.archive.org/web/2023/https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_01.pdf",
        mirror_urls=["https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_01.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=180,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Counter-air doctrine including C-UAS at the joint level. "
              "Primary URL is the verified Wayback snapshot (2023); .mil "
              "URL kept as mirror for US-egress runs.",
    ),
    Source(
        id="us_jp_3_09",
        title="JP 3-09, Joint Fires",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://web.archive.org/web/2023/https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_09.pdf",
        mirror_urls=["https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_09.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=160,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Joint fires doctrine; includes armed UAS in the kill chain. "
              "Primary URL is the verified Wayback snapshot (2023); .mil "
              "URL kept as mirror for US-egress runs.",
    ),
    Source(
        id="us_jp_3_60",
        title="JP 3-60, Joint Targeting (2013 edition — superseded but only public version)",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://www.justsecurity.org/wp-content/uploads/2015/06/Joint_Chiefs-Joint_Targeting_20130131.pdf",
        mirror_urls=[
            "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_60.pdf",
            "https://nsarchive.gwu.edu/media/19565/ocr",
        ],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=140,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        superseded=True,
        notes="Joint targeting cycle (F2T2EA). Heavily referenced in any "
              "strike-capable UAS employment discussion. EDITION GAP: "
              "primary URL is the 31 Jan 2013 edition (Just Security / "
              "NYU Law public mirror, verified May 2026, 877 KB "
              "application/pdf). The 2018 edition is only on esd.whs.mil "
              "(blocked from non-US IPs, no public Wayback snapshot). "
              "F2T2EA cycle and core targeting taxonomy are stable across "
              "the edition gap.",
    ),
    Source(
        id="us_jp_3_85",
        title="JP 3-85, Joint Electromagnetic Spectrum Operations",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://crows.org/download/cjcs-joint-publication-3-85-joint-electromagnetic-spectrum-operations-jemso/?wpdmdl=1511&ind=1699248928881&refresh=28ae7935&filename=JEMSO_JP3_85.pdf",
        mirror_urls=["https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_85.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=130,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="EMS operations; relevant for C-UAS electronic attack and for "
              "UAS data-link survivability — directly relevant to the "
              "project's lost-link target (link denial is the trigger event). "
              "Primary URL is Association of Old Crows mirror (EW industry "
              "nonprofit, JEMSO_JP3_85.pdf, 1.2 MB application/pdf, verified "
              "May 2026); .mil URL kept as mirror for US-egress runs.",
    ),

    # -------------------------------------------------------------------------
    # P0 — US ARMY (Distribution A / public domain)
    # -------------------------------------------------------------------------
    Source(
        id="us_adp_3_0",
        title="ADP 3-0, Operations",
        issuer="US Army",
        doc_type=DocType.SERVICE_DOCTRINE,
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN43323-ADP_3-0-000-WEB-1.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=80,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_ARMY,
        notes="March 2025 edition. Army capstone operations doctrine; "
              "integrates multi-domain operations framing post-Ukraine.",
    ),
    Source(
        id="us_adp_6_0",
        title="ADP 6-0, Mission Command: Command and Control of Army Forces",
        issuer="US Army",
        doc_type=DocType.SERVICE_DOCTRINE,
        url="https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN34403-ADP_6-0-000-WEB-3.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=50,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_ARMY,
        notes="July 2019 edition (with changes). Mission command, commander's "
              "intent, and disciplined initiative. THE doctrinal foundation "
              "for any delegated decision-making (including an onboard SLM "
              "acting under lost-link). Highest-priority addition.",
    ),
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
        url=None,  # armypubs marks this INACTIVE; no superseding public ed.
        mirror_urls=[
            "https://info.publicintelligence.net/USArmy-UAS-MTTP.pdf",
        ],
        license=License.US_DISTRIBUTION_A,
        estimated_pages=180,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_ARMY,
        notes="Multi-service TTP for UAS tactical employment. Verified May "
              "2026 (round 2): armypubs catalog page (PUB_ID=104824) "
              "confirms status RESCINDED/INACTIVE — armypubs serves no PDF. "
              "The 2015 mirror at info.publicintelligence.net (which we "
              "originally listed) is now 404 (URL removed). Newer editions "
              "are classified follow-ons not in public release. TRULY LOST "
              "without a non-public source. Accept as a gap; the C-UAS "
              "doctrine in ATP 3-01.81 (in corpus) covers the inverse "
              "doctrine (defending against UAS) and that's the higher-"
              "value half for our project anyway.",
    ),

    # -------------------------------------------------------------------------
    # P0 — US AIR FORCE DOCTRINE
    # -------------------------------------------------------------------------
    Source(
        id="us_afdp_3_0",
        title="AFDP 3-0, Operations and Planning",
        issuer="US Air Force (LeMay Center)",
        doc_type=DocType.SERVICE_DOCTRINE,
        url="https://web.archive.org/web/2025/https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-0/AFDP3-0Operations.pdf",
        mirror_urls=[
            "https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-0/AFDP3-0Operations.pdf",
        ],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=80,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_AIR_FORCE,
        notes="January 2025 edition. Capstone USAF operations doctrine; "
              "absorbed JADO content from rescinded AFDP 3-99. Correct "
              "filename is 'AFDP3-0Operations.pdf' (the earlier "
              "pattern-extrapolated '3-0-AFDP-OPERATIONS.pdf' was wrong). "
              "Primary URL is the verified Wayback snapshot (2025, "
              "1.58 MB application/pdf); .mil URL kept as mirror for "
              "US-egress runs.",
    ),
    Source(
        id="us_afdp_3_52",
        title="AFDP 3-52, Airspace Control",
        issuer="US Air Force (LeMay Center)",
        doc_type=DocType.SERVICE_DOCTRINE,
        url="https://web.archive.org/web/2024/https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-52/3-52-AFDP-AIRSPACE-CONTROL.pdf",
        mirror_urls=["https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-52/3-52-AFDP-AIRSPACE-CONTROL.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=60,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_AIR_FORCE,
        notes="USAF view on airspace control, explicitly addresses small UAS "
              "and loitering munitions. Primary URL is the verified Wayback "
              "snapshot (2024); .mil URL kept as mirror for US-egress runs.",
    ),
    Source(
        id="us_afdoctrine_smartbook",
        title="Air Force Doctrine Smart Book",
        issuer="US Air Force (LeMay Center)",
        doc_type=DocType.REFERENCE,
        url="https://web.archive.org/web/2024/https://www.doctrine.af.mil/Portals/61/documents/SmartBook/AFDoctrineSmartBook.pdf",
        mirror_urls=["https://www.doctrine.af.mil/Portals/61/documents/SmartBook/AFDoctrineSmartBook.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=70,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_AIR_FORCE,
        notes="30 January 2025 edition. One-page synopses of every current "
              "AFDP. Excellent compact training signal — high terminology "
              "density per token. Primary URL is the verified Wayback "
              "snapshot (2024); .mil URL kept as mirror for US-egress runs.",
    ),
    Source(
        id="us_atp_3_52_4",
        title=("ATP 3-52.4 / MCRP 3-20F.10 / NTTP 6-02.9 / AFTTP 3-2.8, "
               "Multi-Service TTP for Air Control Communication"),
        issuer="US Army / USMC / USN / USAF (ALSA)",
        doc_type=DocType.TTP,
        url="https://info.publicintelligence.net/MTTP-AirspaceControl.pdf",
        mirror_urls=[
            "https://www.alssa.mil/Portals/9/Documents/mttps/acc_2024.pdf",
            "https://static.e-publishing.af.mil/production/1/lemay_center/publication/afttp3-2.8/afttp3-2.8.pdf",
        ],
        license=License.US_DISTRIBUTION_A,
        estimated_pages=210,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.US_AIR_FORCE,
        notes="Tactical C2 voice procedures including for UAS in the air "
              "picture. Primary URL is the publicintelligence.net mirror "
              "(2.4 MB application/pdf, verified May 2026). The 2.4 MB "
              "size suggests this may be the earlier 2021 edition rather "
              "than Oct 2024 — once US-egress is available, run a "
              "verification fetch against the alssa.mil URL to confirm "
              "edition currency.",
    ),

    # -------------------------------------------------------------------------
    # P0 — US POLICY / REFERENCE (autonomy directive, terminology)
    # -------------------------------------------------------------------------
    Source(
        id="us_dodd_3000_09",
        title="DoDD 3000.09, Autonomy in Weapon Systems",
        issuer="US DoD",
        doc_type=DocType.REFERENCE,
        url="https://web.archive.org/web/20260515013449/https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodd/300009p.pdf",
        mirror_urls=["https://www.esd.whs.mil/portals/54/documents/dd/issuances/dodd/300009p.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=30,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.GENERIC,
        notes="January 2023 edition (still current as of 2026). Defines "
              "autonomous and semi-autonomous weapon systems and required "
              "human judgment levels. Keystone reference for the project's "
              "ethics framing — must be cited in the paper's Limitations / "
              "Ethics section to position the work outside LAWS scope. "
              "Primary URL is the verified Wayback snapshot (May 2026); "
              ".mil URL kept as mirror for US-egress runs.",
    ),
    Source(
        id="us_dod_dictionary",
        title="DoD Dictionary of Military and Associated Terms",
        issuer="US Joint Staff",
        doc_type=DocType.REFERENCE,
        url="https://web.archive.org/web/2024/https://irp.fas.org/doddir/dod/dictionary.pdf",
        mirror_urls=["https://irp.fas.org/doddir/dod/dictionary.pdf"],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=380,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.GENERIC,
        notes="November 2021 public edition (last standalone PDF release). "
              "Terminology grounding — every doctrine doc cross-references "
              "this glossary. High-leverage for SLM vocabulary alignment. "
              "Primary URL is the verified Wayback snapshot of the FAS "
              "mirror (2024); FAS direct URL kept as mirror but currently "
              "hits Cloudflare WAF challenge from non-US IPs.",
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
        notes="Withdrawn Jun 2024 and folded into JDP 0-30 UK Air Power. "
              "Archived copy still hosted by gov.uk. UK-specific terminology, "
              "NATO classification mapping, and operational-level UAS "
              "employment — high doctrinal value despite supersession.",
    ),
    Source(
        id="uk_jdp_0_30",
        title="JDP 0-30, UK Air Power (3rd Edition)",
        issuer="UK MOD (DCDC)",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://assets.publishing.service.gov.uk/media/636baad0d3bf7f1649c4e36d/UK_Air_Power_JDP_0_30.pdf",
        license=License.UK_OGL,
        estimated_pages=120,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.UK_JDP,
        notes="November 2022, 3rd edition. Verified May 2026 — note actual "
              "title is 'UK Air Power' (not 'UK Air and Space Power' as "
              "earlier draft notes suggested). Current doctrinal successor "
              "to JDP 0-30.2.",
    ),

    # -------------------------------------------------------------------------
    # P0 — NATO (Allied Joint Publications, public release)
    # -------------------------------------------------------------------------
    Source(
        id="nato_ajp_3",
        title="AJP-3, Allied Joint Doctrine for the Conduct of Operations (Edition D, Version 1)",
        issuer="NATO Standardization Office",
        doc_type=DocType.ALLIED_DOCTRINE,
        url="https://assets.publishing.service.gov.uk/media/6964e72799fbdc498faecce2/AJP_3_Ed_D_V1-O.pdf",
        license=License.NATO_PUBLIC,
        estimated_pages=200,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.NATO,
        notes="NATO joint operations capstone, Edition D Version 1. NATO "
              "Unclassified releasable to public; mirrored via UK MOD on "
              "gov.uk.",
    ),
    Source(
        id="nato_ajp_3_3",
        title="AJP-3.3, Allied Joint Doctrine for Air and Space Operations (Edition C, Version 1)",
        issuer="NATO Standardization Office",
        doc_type=DocType.ALLIED_DOCTRINE,
        url="https://assets.publishing.service.gov.uk/media/698ef9fb7da91680ad7f4389/AJP_3_3_EdC_V1.pdf",
        license=License.NATO_PUBLIC,
        estimated_pages=140,
        priority=Priority.P0,
        parser_strategy=ParserStrategy.NATO,
        notes="Edition C Version 1. NATO Unclassified releasable to public; "
              "mirrored on UK MOD gov.uk (NSO portal rotates URLs). NATO "
              "capstone for air & space operations.",
    ),

    # -------------------------------------------------------------------------
    # P1 — NATO AIR / AIRSPACE (supplementary allied doctrine)
    # -------------------------------------------------------------------------
    Source(
        id="nato_ajp_3_3_5",
        title="AJP-3.3.5, Allied Joint Doctrine for Airspace Control (Edition C, Version 1)",
        issuer="NATO Standardization Office",
        doc_type=DocType.ALLIED_DOCTRINE,
        url="https://assets.publishing.service.gov.uk/media/65eb0ff562ff489bab87b366/AJP_3_3_5_EdC_Airspace.pdf",
        license=License.NATO_PUBLIC,
        estimated_pages=80,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.NATO,
        notes="March 2024 edition. NATO airspace control with UAS-relevant "
              "sections; mirrored on UK MOD gov.uk.",
    ),
    Source(
        id="nato_sto_en_sci_271_03",
        title="STO Educational Note STO-EN-SCI-271/03, UAS Control System (UCS) Standardization",
        issuer="NATO Science and Technology Organization",
        doc_type=DocType.REFERENCE,
        url="https://publications.sto.nato.int/publications/STO%20Educational%20Notes/STO-EN-SCI-271/EN-SCI-271-03.pdf",
        license=License.NATO_PUBLIC,
        estimated_pages=20,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.NATO,
        notes="Public proxy for STANAG 4586 (full STANAG NATO-restricted). "
              "Covers UCS lost-link state machine fundamentals — THE closest "
              "redistributable anchor for the project's lost-link target.",
    ),

    # -------------------------------------------------------------------------
    # P1 — US JOINT (supplementary, intelligence + EW)
    # -------------------------------------------------------------------------
    Source(
        id="us_jp_2_0",
        title="JP 2-0, Joint Intelligence",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://web.archive.org/web/20220524124356/https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp2_0.pdf",
        mirror_urls=[
            "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp2_0.pdf",
            "https://irp.fas.org/doddir/dod/jp2_0.pdf",
        ],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=140,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="Joint intelligence (pre-2024 baseline via Wayback May 2022); "
              "mission context, ISR — essential for situational awareness "
              "reasoning in lost-link scenarios. .mil URL kept as mirror "
              "for US-egress runs (will pull the 2024 revision).",
    ),
    Source(
        id="us_jp_3_13_1",
        title="JP 3-13.1, Electronic Warfare",
        issuer="US Joint Staff",
        doc_type=DocType.JOINT_DOCTRINE,
        url="https://www.usna.edu/Training/_files/documents/References/3C%20MQS%20References/2015-2016%203C%20MQS%20References/Joint%20Publication%203-13.1_Electronic%20Warfare_FEB2012.pdf",
        mirror_urls=[
            "https://edocs.nps.edu/dodpubs/topic/jointpubs/JP3/JP3_13_1_070125.pdf",
            "https://irp.fas.org/doddir/dod/jp3-13-1.pdf",
        ],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=140,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.US_JOINT,
        notes="February 2012 edition (latest public). Largely superseded by "
              "JP 3-85 in operational use but still cited; includes GPS/"
              "comms denial fundamentals relevant to lost-link causation. "
              "Primary URL is US Naval Academy training reference mirror "
              "(verified May 2026, 1.2 MB application/pdf); NPS edocs is "
              "secondary mirror (832 KB).",
    ),

    # -------------------------------------------------------------------------
    # P1 — CIVIL AVIATION LOST-LINK ANCHORS (FAA / EASA / ICAO)
    # The lost-link state machines are best documented in the civil regime.
    # These define what a 'doctrine-correct' lost-link action looks like
    # without crossing into military engagement guidance.
    # -------------------------------------------------------------------------
    Source(
        id="faa_jo_7110_65",
        title="FAA Order JO 7110.65, Air Traffic Control",
        issuer="US Federal Aviation Administration",
        doc_type=DocType.REFERENCE,
        url="https://www.faa.gov/documentLibrary/media/Order/7110.65BB_Bsc_w_Chg_1_and_2_dtd_1-22-26_Final.pdf",
        mirror_urls=[
            "https://www.faa.gov/documentLibrary/media/Order/7110.65BB_Basic_dtd_2-20-25.pdf",
            "https://sanjuan.vatcar.net/wp-content/uploads/2025/09/FAA-7110.65BB.pdf",
        ],
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=600,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.CIVIL_AVIATION,
        notes="FAA ATC handbook, version BB Basic with Changes 1 and 2 "
              "(dated 22 January 2026 — most current). Paragraph 5-2-6 "
              "covers UAS lost link / Code 7400 procedures — concrete "
              "state machine for ATC-side response to lost-link UAS. "
              "Direct stable PDF URL resolved May 2026 via FAA "
              "documentLibrary (9.6 MB application/pdf). Mirror is the "
              "Basic-only version (9 MB, pre-Change 1+2) and a VATSIM "
              "community redistribution.",
    ),
    Source(
        id="faa_notice_uas_lost_link",
        title="FAA Notice N JO 7110.724, UAS Lost Link Procedures",
        issuer="US Federal Aviation Administration",
        doc_type=DocType.REFERENCE,
        url="https://www.faa.gov/documentLibrary/media/Notice/N_JO_7110.724_5-2-9_UAS_Lost_Link.pdf",
        license=License.US_PUBLIC_DOMAIN,
        estimated_pages=8,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.CIVIL_AVIATION,
        notes="Direct ATC-side lost-link procedures for UAS. Small but very "
              "high signal density — every paragraph is lost-link relevant.",
    ),
    Source(
        id="easa_oso6_c2_link",
        title="EASA SAIL III Means of Compliance — OSO#06 (C2 Link Characteristics)",
        issuer="European Union Aviation Safety Agency",
        doc_type=DocType.REFERENCE,
        url="https://www.easa.europa.eu/en/downloads/139100/en",
        license=License.EASA_PUBLIC,
        estimated_pages=40,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.CIVIL_AVIATION,
        notes="SORA C2 link Means of Compliance under SAIL III. Very "
              "specific lost-link behavior requirements: C2 link "
              "monitoring, lost-link declaration thresholds, contingency "
              "transitions.",
    ),
    Source(
        id="easa_uas_ear",
        title="EASA Easy Access Rules for Unmanned Aircraft Systems (Reg 2019/947 + 2019/945)",
        issuer="European Union Aviation Safety Agency",
        doc_type=DocType.REFERENCE,
        url="https://www.easa.europa.eu/en/downloads/110913/en",
        license=License.EASA_PUBLIC,
        estimated_pages=468,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.CIVIL_AVIATION,
        notes="468 pages, ~6.3 MB. EASA EAR defines 'lost C2 link' as the "
              "exceptional condition triggering contingency procedures. "
              "Direct PDF URL resolved May 2026 from the EASA document "
              "library landing page.",
    ),
    Source(
        id="icao_rpas_conops",
        title="ICAO RPAS Concept of Operations for International IFR Operations",
        issuer="International Civil Aviation Organization",
        doc_type=DocType.REFERENCE,
        url="https://web.archive.org/web/20251118172855/https://www.icao.int/sites/default/files/left-menu-pdfs/ICAO%20RPAS%20CONOPS.pdf",
        mirror_urls=["https://www.icao.int/sites/default/files/left-menu-pdfs/ICAO%20RPAS%20CONOPS.pdf"],
        license=License.ICAO_PUBLIC,
        estimated_pages=25,
        priority=Priority.P1,
        parser_strategy=ParserStrategy.CIVIL_AVIATION,
        notes="ICAO public RPAS Concept of Operations — redistributable "
              "substitute for paywalled ICAO Doc 10019 (Manual on RPAS). "
              "Includes C2 link loss handling concepts at the ICAO "
              "framework level. Primary URL is the verified Wayback "
              "snapshot (Nov 2025); icao.int direct URL kept as mirror "
              "(returns 403 from non-US IPs).",
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
    print("By doc type:")
    for dt in DocType:
        n = sum(1 for s in SOURCES if s.doc_type == dt)
        if n:
            print(f"  {dt.value:<28} {n}")
    print()
    print("Missing URL (need manual research):")
    for s in needs_url():
        print(f"  [{s.priority.value}] {s.id} — {s.title}")
