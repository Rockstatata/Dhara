"""Collect real citizen legal questions from public sources.

Only sources cleared in docs/research/question-sources.md are implemented here.
Deliberately absent, and not to be added:

  - Jagonews24  : robots.txt sets Content-Signal ai-train=no, an express
                  reservation of rights. This project fine-tunes models.
  - Quora       : robots.txt forbids using content to train AI models.
  - Reddit      : robots.txt is Disallow: / for all agents.
  - Facebook    : ToS forbids automated collection; the Groups Graph API
                  permissions were removed in April 2024. Manual reading and
                  hand-rewriting by a person is the only route, and it happens
                  outside this file.

Conduct rules, from the project spec and from what each site's robots.txt says:
raw responses are archived before parsing, every fetch checks the archive first
so a re-run costs nothing, and DELAY seconds pass between live requests.

Nothing here redistributes source text. The archive stays in data/raw/, which is
git-ignored, and the released dataset carries only paraphrases (see DECISIONS.md,
"Mined questions: verbatim kept local, paraphrase released").
"""

from __future__ import annotations

import json
import pathlib
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Iterator

import requests
from bs4 import BeautifulSoup

UA = "Dhara-Research-Crawler/1.0 (student NLP project; sarwad038@gmail.com)"
HEADERS = {"User-Agent": UA}
DELAY = 1.5
TIMEOUT = 30

RAW = pathlib.Path("data/raw/questions")


@dataclass
class MinedItem:
    """One fetched article, before questions are separated out of it."""

    source: str
    source_url: str
    title: str
    published: str
    text: str
    crawl_date: str = field(default_factory=lambda: date.today().isoformat())


def _archive_path(source: str, key: str, suffix: str) -> pathlib.Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:120]
    return RAW / source / f"{safe}{suffix}"


def fetch(url: str, dest: pathlib.Path, params: dict | None = None) -> str:
    """Fetch with archive-first resumption and a polite delay."""
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")
    time.sleep(DELAY)
    return resp.text


def _strip_html(html: str) -> str:
    text = BeautifulSoup(html, "lxml").get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Prothom Alo — পাঠকের উকিল and পাঠকের প্রশ্ন: আইন
#
# The listing page is JS-rendered, but Quintype exposes a public JSON API that
# returns the same content, so requests alone is enough. robots.txt allows the
# two paths used here; only /api/auth/, /api/comments/get_comments_json,
# /shell.html and /story/*/element/ are disallowed.
# --------------------------------------------------------------------------

PA_BASE = "https://www.prothomalo.com"
# The paper has run its reader legal column under several names over the years.
# Each is a separate search; the headline filter below drops the diet, mind and
# editor columns that the broader queries also return.
PA_QUERIES = {
    "prothomalo_pathoker_ukil": '"পাঠকের উকিল"',
    "prothomalo_pathoker_proshno": '"পাঠকের প্রশ্ন" আইন',
    "prothomalo_aini_poramorsho": '"আইনি পরামর্শ"',
    "prothomalo_pathoker_proshno_all": '"পাঠকের প্রশ্ন"',
}
# Headlines that the broader queries return but that are not law columns.
PA_LAW_HEADLINE = re.compile(r"আইন|উকিল|আইনজীবী|আদালত|মামলা")


def prothomalo_slugs(source: str, query: str, limit: int = 200) -> list[dict]:
    raw = fetch(
        f"{PA_BASE}/api/v1/advanced-search",
        _archive_path(source, "_listing", ".json"),
        params={"fields": "headline,url,slug,published-at", "q": query, "limit": limit},
    )
    items = json.loads(raw).get("items", [])
    return [i for i in items if PA_LAW_HEADLINE.search(i.get("headline") or "")]


def prothomalo(source: str, query: str) -> Iterator[MinedItem]:
    for item in prothomalo_slugs(source, query):
        slug = item.get("slug")
        if not slug:
            continue
        raw = fetch(
            f"{PA_BASE}/api/v1/stories-by-slug",
            _archive_path(source, slug, ".json"),
            params={"slug": slug},
        )
        story = json.loads(raw).get("story", {})
        parts = [
            el.get("text", "")
            for card in story.get("cards", [])
            for el in card.get("story-elements", [])
            if el.get("type") == "text"
        ]
        yield MinedItem(
            source=source,
            source_url=f"{PA_BASE}/{slug}",
            title=story.get("headline") or item.get("headline") or "",
            published=str(story.get("published-at") or item.get("published-at") or ""),
            text=_strip_html(" ".join(parts)),
        )


# --------------------------------------------------------------------------
# Ajker Patrika — আইনি পরামর্শ
#
# Server-rendered HTML; the listing hrefs and the full article body are both in
# the initial payload. One reader question per article, and the answer is marked
# with a literal "উত্তর:" — the cleanest separation of any source found.
# The same content is served at /women/legal-advice and /lifestyle/legal-advice;
# only the first is walked, so nothing is double-counted.
# --------------------------------------------------------------------------

AP_BASE = "https://www.ajkerpatrika.com"
AP_SOURCE = "ajkerpatrika_aini_poramorsho"
AP_HREF = re.compile(r"/women/legal-advice/[A-Za-z0-9-]+")


def ajkerpatrika(max_pages: int = 8) -> Iterator[MinedItem]:
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        html = fetch(
            f"{AP_BASE}/women/legal-advice",
            _archive_path(AP_SOURCE, f"_listing_p{page}", ".html"),
            params={"page": page},
        )
        hrefs = set(AP_HREF.findall(html)) - seen
        if not hrefs:
            break
        seen |= hrefs
        for href in sorted(hrefs):
            page_html = fetch(
                AP_BASE + href, _archive_path(AP_SOURCE, href, ".html")
            )
            soup = BeautifulSoup(page_html, "lxml")
            h1 = soup.find("h1")
            body = " ".join(
                p.get_text(" ") for p in soup.find_all("p")
            )
            yield MinedItem(
                source=AP_SOURCE,
                source_url=AP_BASE + href,
                title=h1.get_text(" ").strip() if h1 else "",
                published="",
                text=re.sub(r"\s+", " ", body).strip(),
            )


# --------------------------------------------------------------------------
# Lawyers Club Bangladesh — দৈনন্দিন জীবনে আইন
#
# Open WordPress REST API, and robots.txt is "Disallow:" with an empty value,
# which permits everything. This is the only source with real land coverage
# (নামজারি, খতিয়ান, অগ্রক্রয়, বাটোয়ারা), which every newspaper column misses.
# These are editorial explainers rather than citizen letters, so they enter the
# pool as candidate queries, never as evidence of citizen register.
# --------------------------------------------------------------------------

LCB_BASE = "https://www.lawyersclubbangladesh.com"
LCB_SOURCE = "lawyersclub_daily_life"
# Category ids from the site's own taxonomy endpoint. "দৈনন্দিন জীবনে আইন" is the
# everyday-law column and carries the land coverage nothing else has;
# "নারী ও শিশু" is the only source of any kind for the women_children domain.
# Category 757 "নারী ও শিশু" was tried and dropped: all 277 posts are court and
# policy *news* ("হাইকোর্টের নির্দেশ", "কমিটি গঠন"), not citizen questions, and it
# yielded zero usable items. Recorded here so nobody re-adds it hoping otherwise.
# The women_children domain therefore has no mined source and must be filled by
# annotators reframing questions from the family and criminal columns.
LCB_CATEGORIES = {
    "lawyersclub_daily_life": 756,
}


def lawyersclub(
    source: str = LCB_SOURCE, cat_id: int = 756, per_page: int = 100
) -> Iterator[MinedItem]:
    for page in range(1, 10):
        try:
            raw = fetch(
                f"{LCB_BASE}/wp-json/wp/v2/posts",
                _archive_path(source, f"_posts_p{page}", ".json"),
                params={"categories": cat_id, "per_page": per_page, "page": page},
            )
        except requests.HTTPError as exc:
            # WordPress answers a page past the last one with 400
            # rest_post_invalid_page_number rather than an empty list.
            if exc.response is not None and exc.response.status_code == 400:
                break
            raise
        posts = json.loads(raw)
        if not posts:
            break
        for post in posts:
            yield MinedItem(
                source=source,
                source_url=post.get("link", ""),
                title=_strip_html(post.get("title", {}).get("rendered", "")),
                published=post.get("date", ""),
                text=_strip_html(post.get("content", {}).get("rendered", "")),
            )


# --------------------------------------------------------------------------
# The Daily Star — Your Advocate
#
# English, and on several articles only the lawyer's reply is rendered while the
# reader's query is not in the HTML. Kept because it is the only source with
# labour, tax and consumer coverage, but every item needs a human check that a
# question is actually present. Stock Drupal robots.txt; nothing relevant is
# disallowed. Pagination stops at page 3 — deeper pages return a constant set of
# sidebar links that are not real results.
# --------------------------------------------------------------------------

DS_BASE = "https://www.thedailystar.net"
DS_SOURCE = "dailystar_your_advocate"
DS_HREF = re.compile(r"/law-our-rights/your-advocate/[A-Za-z0-9/-]+")


def dailystar(max_pages: int = 4) -> Iterator[MinedItem]:
    seen: set[str] = set()
    for page in range(0, max_pages):
        html = fetch(
            f"{DS_BASE}/ds/law-our-rights/your-advocate",
            _archive_path(DS_SOURCE, f"_listing_p{page}", ".html"),
            params={"page": page},
        )
        hrefs = set(DS_HREF.findall(html)) - seen
        if not hrefs:
            break
        seen |= hrefs
        for href in sorted(hrefs):
            page_html = fetch(DS_BASE + href, _archive_path(DS_SOURCE, href, ".html"))
            soup = BeautifulSoup(page_html, "lxml")
            h1 = soup.find("h1")
            body = " ".join(p.get_text(" ") for p in soup.find_all("p"))
            yield MinedItem(
                source=DS_SOURCE,
                source_url=DS_BASE + href,
                title=h1.get_text(" ").strip() if h1 else "",
                published="",
                text=re.sub(r"\s+", " ", body).strip(),
            )


SOURCES = {
    **{
        name: (lambda n=name, q=query: prothomalo(n, q))
        for name, query in PA_QUERIES.items()
    },
    "ajkerpatrika_aini_poramorsho": ajkerpatrika,
    **{
        name: (lambda n=name, c=cat: lawyersclub(n, c))
        for name, cat in LCB_CATEGORIES.items()
    },
    "dailystar_your_advocate": dailystar,
}


def mine(names: list[str] | None = None) -> Iterator[MinedItem]:
    """Yield items from each source, keeping going if one source breaks.

    A newspaper changing its markup should cost us that newspaper, not the whole
    run — everything already archived stays usable.
    """
    for name in names or list(SOURCES):
        try:
            yield from SOURCES[name]()
        except Exception as exc:  # noqa: BLE001 — one bad source must not stop the rest
            print(f"  !! {name} failed: {type(exc).__name__}: {exc}")


def write_jsonl(items: Iterator[MinedItem], dest: pathlib.Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            n += 1
    return n
