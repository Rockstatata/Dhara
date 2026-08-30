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
# Three shapes beyond the plain "103." that bdlaws actually prints, each of which
# was silently costing whole provisions:
#   - a stray leading dot or tab, as in ".301." (Penal Code) and ".\t72."
#   - a hyphen before the letter suffix: "171-I.", "70-B.", "265-I.", "19-I."
#   - three-character suffixes: "44CCC.", "১৫ককক।"
# Widening the pattern recovers 13 provisions and changes no existing parse.
PROVISION_HEAD = re.compile(
    r"^\s*\.?\s*(?:(\d+)\[)?\s*([০-৯0-9]+\s*[-–]?\s*[ক-হA-Za-z]{0,3})\s*[।৷.]"
)
# A record that opens with a bare sub-section marker is a continuation: BLAD
# sometimes emits "(1) ..." and "(2) ..." of one provision as separate records.
# Dropping them silently loses 153 provisions of the CrPC alone, so they are
# merged back into the provision above — sub-sections stay with their parent.
CONTINUATION = re.compile(r"^\s*(?:\d+\[)?\s*\(([০-৯0-9]+[ক-হa-z]?)\)")
# An omitted provision, marked with asterisks rather than the word বিলুপ্ত.
OMITTED_STARS = re.compile(r"^\s*(?:\d+\[)?\s*\*+\s*\]?\s*$")
FOOTNOTE_MARKER = re.compile(r"(\d+)\[")

# Excluded by rule, not oversight — see configs/domains.yaml `exclusions`.
EXCLUDE_TITLE = re.compile(
    r"সংশোধন|ইনষ্টিটিউট|\(Amendment\)|\[রহিত\]|\[Repealed\]", re.I
)
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
    portal = _portal_titles()

    for act in acts(path):
        stats["acts_total"] += 1
        title = (act.get("act_title") or "").strip()
        entry = match_domain(title, index)
        if entry is None:
            if not all_acts:
                continue
            entry = UNCLASSIFIED
        # The portal carries the repeal marker that BLAD's title omits; without
        # this check a repealed act that BLAD happens to carry is indistinguishable
        # from a live one. See `repealed_on_portal`.
        if repealed_on_portal(_act_id(act.get("source_url", "")), portal):
            stats["acts_excluded_amendment_or_repealed"] += 1
            continue
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

# --------------------------------------------------------------------------
# The bdlaws pass
# --------------------------------------------------------------------------

# Three separate crawls feed this, all in the same record shape:
#   bdlaws_provisions.jsonl  the 203 acts BLAD carries with zero provisions
#   bdlaws_missing.jsonl     the 53 acts absent from BLAD entirely, mostly
#                            enacted after BLAD's July 2026 snapshot
#   bdlaws_titles.jsonl      provision titles for acts BLAD does supply
BDLAWS_JSONL = [
    pathlib.Path("data/interim/bdlaws_provisions.jsonl"),
    pathlib.Path("data/interim/bdlaws_missing.jsonl"),
]
# Act titles scraped from the portal's chronological index. Needed because acts
# absent from BLAD have no act-level metadata anywhere else.
ACT_TITLES = pathlib.Path("data/interim/bdlaws_act_titles.json")


def _bdlaws_rows(paths) -> dict[str, list[dict]]:
    """Load crawled provisions, keyed by act, with duplicates dropped.

    The crawl files overlap: an act can be fetched by one pass and re-fetched by
    a later one aimed at a different gap. Without this guard the same provision
    is emitted twice and produces two chunks sharing one `chunk_id` — which is
    the answer key the gold set points at, so a duplicate silently makes an
    annotation ambiguous. It happened on 2026-08-25 with act-1710, present in
    both `bdlaws_missing.jsonl` and a follow-up crawl, and it put 68 duplicated
    ids into the corpus. First occurrence wins; the pages are identical.
    """
    if isinstance(paths, pathlib.Path):
        paths = [paths]
    by_act: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["act_id"], row.get("section_id", ""))
                if key in seen:
                    continue
                seen.add(key)
                by_act.setdefault(row["act_id"], []).append(row)
    return by_act


def _portal_titles(path: pathlib.Path = ACT_TITLES) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def repealed_on_portal(act_id: str, portal: dict[str, str] | None = None) -> bool:
    """Is this act marked `[রহিত]` / `[Repealed]` in the bdlaws index?

    The repeal marker lives in the *portal's* title, not in BLAD's. BLAD stores
    কপিরাইট আইন, ২০০০ where the portal says কপিরাইট আইন, ২০০০ [রহিত] — so running
    `EXCLUDE_TITLE` over a BLAD title cannot see the marker, and every repealed
    act that BLAD happens to carry walked straight into the corpus. That put 4
    repealed acts and 316 chunks into the evaluation corpus and 215 acts and
    7,849 chunks into the full one, including two thirds of the cybercrime
    domain.

    Retrieving a repealed provision is the one failure this system is explicitly
    forbidden to make, so the check is done against the portal title regardless
    of which source supplied the text.
    """
    title = (portal if portal is not None else _portal_titles()).get(act_id, "")
    return bool(title) and bool(EXCLUDE_TITLE.search(title))


def _detect_language(text: str) -> str:
    """BLAD marks these acts `unknown`, so language is measured from the text."""
    bn = sum(1 for c in text if "ঀ" <= c <= "৿")
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    if bn > en:
        return "bengali"
    return "english" if en else "unknown"


def build_from_bdlaws(
    path=BDLAWS_JSONL,
    blad_path: pathlib.Path = BLAD_JSON,
    all_acts: bool = False,
) -> tuple[list[Chunk], dict[str, int]]:
    """Chunks for the acts BLAD leaves empty, fetched from the portal.

    Act-level metadata still comes from BLAD — it carries the title, year and
    number for these acts even though it carries none of their provisions. The
    provision text and, crucially, the provision *title* come from bdlaws, which
    is the only source for titles at all.
    """
    index = build_act_index(load_domains())
    by_act = _bdlaws_rows(path)
    meta = {_act_id(a.get("source_url", "")): a for a in acts(blad_path)}
    portal = _portal_titles()
    stats = {"acts": 0, "provisions": 0, "with_title": 0, "unnumbered": 0, "chunks": 0}
    chunks: list[Chunk] = []

    for act_id, rows in by_act.items():
        info = meta.get(act_id, {})
        # Acts absent from BLAD have no metadata there, so the portal index is
        # the only source of a title — and without a title an act cannot be
        # matched to a domain or cited.
        title = (info.get("act_title") or portal.get(act_id) or "").strip()
        if not title:
            continue
        if repealed_on_portal(act_id, portal):
            continue
        entry = match_domain(title, index)
        if entry is None:
            if not all_acts:
                continue
            entry = UNCLASSIFIED
        if EXCLUDE_TITLE.search(title):
            continue
        stats["acts"] += 1
        slug = f"{entry['id']}_{act_id}"
        kind = "article" if "সংবিধান" in title else "section"

        for row in rows:
            stats["provisions"] += 1
            raw = row.get("text_raw") or ""
            if EXCLUDE_PROVISION.search(raw) or OMITTED_STARS.match(raw):
                continue
            number, body, markers = parse_provision(raw)
            if not number or not body:
                stats["unnumbered"] += 1
                continue
            prov_title = row.get("title_bn") or ""
            if prov_title:
                stats["with_title"] += 1
            language = _detect_language(raw)
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
                        act_title_en=info.get("csv_metadata", {}).get("act_title_from_csv", ""),
                        act_year=int(info["act_year"]) if str(info.get("act_year", "")).isdigit() else None,
                        act_no=str(info.get("act_no") or ""),
                        chapter_bn=None,
                        provision_kind=kind,
                        provision_no_bn=number,
                        provision_no_ascii=to_ascii_digits(number),
                        provision_title_bn=prov_title or None,
                        text_bn=text_bn,
                        text_raw=body if len(pieces) == 1 else text_bn,
                        domain=entry["id"],
                        risk_tier=entry["risk_tier"],
                        sub_idx=i,
                        n_sub=len(pieces),
                        source_url=row.get("source_url", ""),
                        source_dataset="bdlaws",
                        crawl_date=row.get("crawl_date", ""),
                        n_words=len(piece),
                        language=language,
                        footnote_markers=markers,
                    )
                )
    stats["chunks"] = len(chunks)
    return chunks, stats


# --------------------------------------------------------------------------
# Title enrichment
# --------------------------------------------------------------------------

# A list rather than one path: the title crawl is resumable per act but writes
# its output file from scratch, so a follow-up run that fills a gap has to go to
# a new file or it would truncate the first run's work. `_fix` holds the six acts
# a dropped connection cost the main run.
BDLAWS_TITLES = [
    pathlib.Path("data/interim/bdlaws_titles_all.jsonl"),
    pathlib.Path("data/interim/bdlaws_titles.jsonl"),
    pathlib.Path("data/interim/bdlaws_titles_fix.jsonl"),
]


def dedupe_chunk_ids(chunks: list[Chunk]) -> tuple[list[Chunk], dict[str, int]]:
    """Make `chunk_id` unique, because it is the answer key.

    A gold answer points at a `chunk_id`. Two chunks sharing one makes that
    answer ambiguous and nothing raises an error, so this runs on every build.

    Collisions arrive two ways and deserve opposite treatment:

      - **Identical text.** The same provision emitted twice, usually because a
        source file overlaps another. The second copy carries no information and
        is dropped.
      - **Different text.** Two genuinely different provisions whose numbers
        parsed to the same value — an amendment clause and a substantive section
        in the same Act, most often. Dropping one would lose real law, so the
        later ones are suffixed `_d2`, `_d3`. An ugly id costs nothing; a missing
        provision costs a citizen an answer.
    """
    stats = {"exact_duplicates_dropped": 0, "collisions_renamed": 0}
    seen: dict[str, str] = {}          # chunk_id -> text_bn of the first seen
    counts: dict[str, int] = {}
    out: list[Chunk] = []
    for chunk in chunks:
        first = seen.get(chunk.chunk_id)
        if first is None:
            seen[chunk.chunk_id] = chunk.text_bn
            out.append(chunk)
            continue
        if first == chunk.text_bn:
            stats["exact_duplicates_dropped"] += 1
            continue
        counts[chunk.chunk_id] = counts.get(chunk.chunk_id, 1) + 1
        chunk.chunk_id = f"{chunk.chunk_id}_d{counts[chunk.chunk_id]}"
        stats["collisions_renamed"] += 1
        seen[chunk.chunk_id] = chunk.text_bn
        out.append(chunk)
    return out, stats


def enrich_titles(
    chunks: list[Chunk], path=BDLAWS_TITLES
) -> tuple[list[Chunk], dict[str, int]]:
    """Attach provision titles crawled from bdlaws to BLAD-derived chunks.

    BLAD carries no provision titles at all. bdlaws puts them in `.txt-head`, so
    a separate pass fetches them for the acts BLAD *does* supply. Those chunks
    already exist, so this is a join rather than an append — matched on
    (portal act id, provision number in ASCII digits), which is the only pair
    stable across the two sources. The URL's section id is a global integer and
    is useless for matching.

    Titles matter more than their word count suggests: they are the densest
    concentration of formal legal terminology in the corpus, which is exactly the
    vocabulary the register gap is about.
    """
    stats = {"available": 0, "matched": 0, "already_had": 0}
    paths = [path] if isinstance(path, pathlib.Path) else list(path)

    lookup: dict[tuple[str, str], str] = {}
    for one in paths:
        if not one.exists():
            continue
        with one.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                title = (row.get("title_bn") or "").strip()
                if not title:
                    continue
                number, _body, _markers = parse_provision(row.get("text_raw") or "")
                if number:
                    stats["available"] += 1
                    lookup[(row["act_id"], to_ascii_digits(number))] = title

    for chunk in chunks:
        if chunk.provision_title_bn:
            stats["already_had"] += 1
            continue
        key = (chunk.act_id.split("_")[-1], chunk.provision_no_ascii)
        title = lookup.get(key)
        if title:
            chunk.provision_title_bn = title
            stats["matched"] += 1
    return chunks, stats
