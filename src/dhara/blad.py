"""Build the Dhara corpus from BLAD, the published Bangladeshi Acts dataset.

BLAD (arXiv 2607.17111, CC-BY on Hugging Face) already contains what a bdlaws
crawl would produce, so Dhara does not re-scrape it. What BLAD does *not*
contain, verified against the file rather than its dataset card:

  - No provision titles. The card advertises `section_title`; the actual records
    carry `section_content` alone. Titles are dense with legal terminology and
    help retrieval, so this is a real loss.
  - No provision numbers as a field. The number is embedded at the head of the
    text, in Bangla or ASCII digits, terminated by either danda, and sometimes
    behind a footnote marker such as `55[`.
  - `is_repealed` is present but null for all 1,484 acts, so repeal status has
    to be derived from the title here.
  - No chapter (অধ্যায়).

Those four gaps are why `configs/domains.yaml` acts still get a targeted bdlaws
pass; see DECISIONS.md.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Iterator

import yaml

from .normalize import light
from .schema import Chunk, to_ascii_digits

BLAD_JSON = pathlib.Path(
    "data/external/blad/Contextualized_Bangladesh_Legal_Acts.json"
)
DOMAINS_YAML = pathlib.Path("configs/domains.yaml")

# A provision opens with its number: "১।", "৪০৷", "2.", optionally preceded by a
# footnote marker like "55[" that the portal renders as a superscript.
PROVISION_HEAD = re.compile(r"^\s*(?:(\d+)\[)?\s*([০-৯0-9]+\s*[ক-হA-Za-z]{0,2})\s*[।৷.]")
# A record that opens with a bare sub-section marker is a continuation: BLAD
# sometimes emits "(1) ..." and "(2) ..." of one provision as separate records.
# Dropping them silently loses 153 provisions of the CrPC alone, so they are
# merged back into the provision above — sub-sections stay with their parent.
CONTINUATION = re.compile(r"^\s*(?:\d+\[)?\s*\(([০-৯0-9]+[ক-হa-z]?)\)")
# An omitted provision, marked with asterisks rather than the word বিলুপ্ত.
OMITTED_STARS = re.compile(r"^\s*(?:\d+\[)?\s*\*+\s*\]?\s*$")
FOOTNOTE_MARKER = re.compile(r"(\d+)\[")

# Excluded by rule, not oversight — see configs/domains.yaml `exclusions`.
EXCLUDE_TITLE = re.compile(r"সংশোধন|\(Amendment\)|\[রহিত\]|\[Repealed\]", re.I)
EXCLUDE_PROVISION = re.compile(r"\[বিলুপ্ত\]|\[Omitted\]")

MAX_WORDS = 400
SUB_WORDS = 250
SUB_OVERLAP = 50


def load_domains(path: pathlib.Path = DOMAINS_YAML) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["domains"]


def _act_id(source_url: str) -> str:
    match = re.search(r"act-print-(\d+)", source_url or "")
    return match.group(1) if match else ""


def build_act_index(domains: list[dict]) -> dict[str, dict]:
    """Map a matchable act-title fragment to its domain entry.

    Matching is on the Bangla title fragments written in configs/domains.yaml.
    Deliberately literal: an act nobody listed must not silently join the corpus,
    because `domain` and `risk_tier` drive how an answer is presented.
    """
    index: dict[str, dict] = {}
    for entry in domains:
        for act in entry.get("acts") or []:
            fragment = re.split(r"[,—(#]", str(act))[0].strip()
            if len(fragment) >= 6:
                index[fragment] = entry
    return index


def _loose(text: str) -> str:
    """Fold the differences that stop two spellings of one act title matching.

    bdlaws and BLAD disagree about hyphens and spacing in several titles —
    ভোক্তা-অধিকার versus ভোক্তা অধিকার is the one that bit first. Without this,
    a present act is reported as a coverage gap, which is a worse error than a
    missed match because it sends someone off to crawl something we already have.
    """
    return re.sub(r"[\s‌‍-]+", "", light(text))


def match_domain(title: str, index: dict[str, dict]) -> dict | None:
    loose_title = _loose(title)
    for fragment, entry in index.items():
        if _loose(fragment) in loose_title:
            return entry
    return None


def parse_provision(content: str) -> tuple[str, str, list[str]]:
    """Return (number_as_printed, body, footnote_markers).

    The number stays as printed for the citation; `to_ascii_digits` supplies the
    lookup form. An unnumbered provision returns an empty number and is dropped
    by the caller rather than guessed at.
    """
    markers = FOOTNOTE_MARKER.findall(content[:40])
    head = PROVISION_HEAD.match(content)
    if not head:
        return "", content.strip(), markers
    number = head.group(2)
    return number, content[head.end():].strip(), markers


def _sub_chunk(words: list[str]) -> list[list[str]]:
    if len(words) <= MAX_WORDS:
        return [words]
    out, start = [], 0
    while start < len(words):
        out.append(words[start : start + SUB_WORDS])
        start += SUB_WORDS - SUB_OVERLAP
    return out


def acts(path: pathlib.Path = BLAD_JSON) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["acts"]


# Acts outside configs/domains.yaml still enter the corpus when build(all_acts=True)
# is used. They carry domain "other" and a medium risk tier: unknown risk must not
# silently present as low, and must not flood every result with a legal-aid
# referral either, which would make the referral meaningless where it matters.
UNCLASSIFIED = {"id": "other", "risk_tier": "medium"}


def build(
    path: pathlib.Path = BLAD_JSON, all_acts: bool = False
) -> tuple[list[Chunk], dict[str, int]]:
    """Build the corpus.

    all_acts=False  -> only the everyday-life acts named in configs/domains.yaml
    all_acts=True   -> every act in BLAD, domain-labelled where known
    """
    index = build_act_index(load_domains())
    stats = {
        "acts_total": 0,
        "acts_matched": 0,
        "acts_excluded_amendment_or_repealed": 0,
        "acts_matched_but_empty": 0,
        "provisions_seen": 0,
        "provisions_dropped_omitted": 0,
        "provisions_dropped_unnumbered": 0,
        "provisions_merged_into_parent": 0,
        "provisions_dropped_preamble": 0,
        "chunks": 0,
    }
    chunks: list[Chunk] = []

    for act in acts(path):
        stats["acts_total"] += 1
        title = (act.get("act_title") or "").strip()
        entry = match_domain(title, index)
        if entry is None:
            if not all_acts:
                continue
            entry = UNCLASSIFIED
        if EXCLUDE_TITLE.search(title):
            stats["acts_excluded_amendment_or_repealed"] += 1
            continue
        stats["acts_matched"] += 1

        sections = act.get("sections") or []
        if not sections:
            stats["acts_matched_but_empty"] += 1
            continue

        act_id = _act_id(act.get("source_url", ""))
        slug = f"{entry['id']}_{act_id}"
        kind = "article" if "সংবিধান" in title else "section"

        # First pass: attach continuation records to the provision they belong to.
        merged: list[tuple[str, str, list[str]]] = []
        for section in sections:
            stats["provisions_seen"] += 1
            content = section.get("section_content") or ""
            if EXCLUDE_PROVISION.search(content) or OMITTED_STARS.match(content):
                stats["provisions_dropped_omitted"] += 1
                continue
            if CONTINUATION.match(content) and merged:
                number, body, markers = merged[-1]
                merged[-1] = (number, f"{body} {content.strip()}", markers)
                stats["provisions_merged_into_parent"] += 1
                continue
            number, body, markers = parse_provision(content)
            if not number or not body:
                # A preamble ("WHEREAS it is expedient...") is not a provision and
                # is excluded on purpose; anything else here is a parser miss.
                if re.match(r"\s*(WHEREAS|যেহেতু)", content):
                    stats["provisions_dropped_preamble"] += 1
                else:
                    stats["provisions_dropped_unnumbered"] += 1
                continue
            merged.append((number, body, markers))

        for number, body, markers in merged:
            number = re.sub(r"\s+", "", number)
            provision_id = f"{slug}_s{to_ascii_digits(number)}"
            pieces = _sub_chunk(light(body).split())
            for i, piece in enumerate(pieces):
                text_bn = " ".join(piece)
                chunks.append(
                    Chunk(
                        chunk_id=provision_id if len(pieces) == 1 else f"{provision_id}_p{i}",
                        provision_id=provision_id,
                        act_id=slug,
                        act_title_bn=title,
                        act_title_en=act.get("csv_metadata", {}).get(
                            "act_title_from_csv", ""
                        ),
                        act_year=int(act["act_year"])
                        if str(act.get("act_year", "")).isdigit()
                        else None,
                        act_no=str(act.get("act_no") or ""),
                        chapter_bn=None,  # BLAD has none; the bdlaws pass fills it
                        provision_kind=kind,
                        provision_no_bn=number,
                        provision_no_ascii=to_ascii_digits(number),
                        provision_title_bn=None,  # likewise
                        text_bn=text_bn,
                        text_raw=body if len(pieces) == 1 else text_bn,
                        domain=entry["id"],
                        risk_tier=entry["risk_tier"],
                        sub_idx=i,
                        n_sub=len(pieces),
                        source_url=act.get("source_url", ""),
                        source_dataset="blad",
                        crawl_date=(act.get("fetch_timestamp") or "")[:10],
                        n_words=len(piece),
                        language=act.get("language") or "unknown",
                        footnote_markers=markers,
                    )
                )
    stats["chunks"] = len(chunks)
    return chunks, stats


def missing_acts(path: pathlib.Path = BLAD_JSON) -> list[tuple[str, str, str]]:
    """Acts named in configs/domains.yaml that BLAD cannot supply.

    Either absent from BLAD entirely, or present with zero provisions. These are
    the targets of the bdlaws pass; returning them explicitly stops a silent
    coverage hole from reaching the corpus.
    """
    index = build_act_index(load_domains())
    present: dict[str, int] = {}
    for act in acts(path):
        title = _loose(act.get("act_title") or "")
        for fragment in index:
            if _loose(fragment) in title:
                present[fragment] = max(
                    present.get(fragment, 0), len(act.get("sections") or [])
                )
    gaps = []
    for fragment, entry in index.items():
        count = present.get(fragment)
        if count is None:
            gaps.append((entry["id"], fragment, "absent from BLAD"))
        elif count == 0:
            gaps.append((entry["id"], fragment, "present but zero provisions"))
    return sorted(gaps)
