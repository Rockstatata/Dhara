"""Fetch acts from the bdlaws portal, for what BLAD cannot supply.

Two jobs, both gaps found by measuring BLAD rather than trusting its card:

  1. **The 203 empty acts.** BLAD carries them with `language: "unknown"` and zero
     provisions. Among them are সাইবার নিরাপত্তা আইন ২০২৩, ডিজিটাল নিরাপত্তা আইন
     ২০১৮ and মানিলণ্ডারিং প্রতিরোধ আইন ২০০৯ — not obscure.
  2. **Provision titles.** BLAD has none at all. bdlaws puts them in `.txt-head`,
     and they are dense with the legal terminology that helps retrieval most.

Portal facts, all verified rather than assumed:

  - HTTP only. Port 443 refuses connections, so requests must not be upgraded.
  - No robots.txt; `/robots.txt` returns a styled HTML 404. No declared
    restriction, which is not the same as permission — hence the delay below.
  - Responses are sometimes UTF-16 with a BOM, so encoding is sniffed per page.
  - `?lang=` switches the interface chrome only, never the text of the law. An
    English act stays English under `lang=bn` and vice versa; there is no
    parallel bilingual version of any act.
  - Section ids in URLs are opaque global integers, not provision numbers.
"""

from __future__ import annotations

import pathlib
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Iterator

import requests
from bs4 import BeautifulSoup

BASE = "http://bdlaws.minlaw.gov.bd"
HEADERS = {
    "User-Agent": "Dhara-Research-Crawler/1.0 "
    "(student NLP project; sarwad038@gmail.com)"
}
DELAY = 1.5
TIMEOUT = 30
RAW = pathlib.Path("data/raw/bdlaws")

SECTION_HREF = re.compile(r"/act-(\d+)/section-(\d+)\.html")


@dataclass
class ScrapedProvision:
    act_id: str
    section_id: str
    title_bn: str
    text_raw: str
    source_url: str
    crawl_date: str = field(default_factory=lambda: date.today().isoformat())


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xfe\xff", b"\xff\xfe"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def fetch(url: str, dest: pathlib.Path) -> str:
    """Archive-first fetch. Re-running a crawl costs nothing and re-crawling is
    both slow and rude, so the archive is written before anything is parsed."""
    if dest.exists():
        return _decode(dest.read_bytes())
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    time.sleep(DELAY)
    return _decode(resp.content)


def section_ids(act_id: str) -> list[str]:
    html = fetch(f"{BASE}/act-{act_id}.html", RAW / act_id / "_index.html")
    found = {sid for aid, sid in SECTION_HREF.findall(html) if aid == act_id}
    return sorted(found, key=int)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def provision(act_id: str, section_id: str) -> ScrapedProvision | None:
    url = f"{BASE}/act-{act_id}/section-{section_id}.html"
    html = fetch(url, RAW / act_id / f"{section_id}.html")
    soup = BeautifulSoup(html, "lxml")
    head = soup.select_one(".txt-head")
    body = soup.select_one(".txt-details")
    if body is None:
        return None
    text = _clean(body.get_text(" "))
    if not text:
        return None
    return ScrapedProvision(
        act_id=act_id,
        section_id=section_id,
        title_bn=_clean(head.get_text(" ")) if head else "",
        text_raw=text,
        source_url=url,
    )


def act(act_id: str) -> Iterator[ScrapedProvision]:
    for section_id in section_ids(act_id):
        try:
            found = provision(act_id, section_id)
        except requests.HTTPError as exc:
            print(f"  !! act-{act_id}/section-{section_id}: {exc}")
            continue
        if found is not None:
            yield found


def acts(act_ids: list[str]) -> Iterator[ScrapedProvision]:
    for i, act_id in enumerate(act_ids, start=1):
        try:
            found = list(act(act_id))
        except Exception as exc:  # noqa: BLE001 — one bad act must not stop a long crawl
            print(f"  !! act-{act_id} failed: {type(exc).__name__}: {exc}")
            continue
        print(f"  [{i}/{len(act_ids)}] act-{act_id}: {len(found)} provisions")
        yield from found


def write_jsonl(items: Iterator[ScrapedProvision], dest: pathlib.Path) -> int:
    import json

    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            n += 1
    return n
