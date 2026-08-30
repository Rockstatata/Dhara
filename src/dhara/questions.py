"""Split mined articles into individual citizen questions, and strip PII.

Each source lays its questions out differently, so there is a rule per source and
a conservative fallback. Nothing here is expected to be perfect: the output is a
*candidate* pool that a human reads during annotation. What matters is that the
failures are visible, so every record carries `split_confidence` and a human
sees the low ones first.

PII removal happens here rather than at annotation time. The archived raw HTML in
data/raw/ keeps the original for provenance, and that directory is git-ignored.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import asdict, dataclass
from typing import Iterator

# --------------------------------------------------------------------------
# PII
# --------------------------------------------------------------------------

# Bangladeshi mobile numbers in either numeral system, plus generic long digits.
PHONE = re.compile(r"[০-৯0-9]{11}|[০-৯0-9]{4}[- ][০-৯0-9]{6,7}")
# NID: 10, 13 or 17 digits.
NID = re.compile(r"\b[০-৯0-9]{13}\b|\b[০-৯0-9]{17}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

# A reader's sign-off: a short name, a comma, a short place, optionally "থেকে",
# at the very end of the question. This is where nearly all the PII lives.
SIGNATURE = re.compile(
    r"\s*(?:নাম প্রকাশে অনিচ্ছুক|[^।?*\n]{2,40})\s*,\s*[^।?*\n]{2,30}?"
    r"(?:\s*থেকে)?\s*[।.]?\s*$"
)
# Standing boilerplate that is not part of anyone's question.
BOILERPLATE = re.compile(
    r"(পাঠকের উকিল|নকশা|দৈনিক প্রথম আলো|কারওয়ান বাজার|সিএ ভবন"
    r"|খামের ওপর লিখুন|ই-মেইল|পরামর্শ দিয়েছেন|উত্তরণ)"
)


def strip_pii(text: str) -> str:
    text = EMAIL.sub(" ", text)
    text = NID.sub(" ", text)
    text = PHONE.sub(" ", text)
    text = SIGNATURE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class Question:
    qid: str
    question_bn: str
    source: str
    source_url: str
    article_title: str
    n_words: int
    split_confidence: str  # high | medium | low
    # Filled by annotators, not here.
    relevant_chunk_ids: list[str] | None = None
    register: str | None = None
    domain: str | None = None
    paraphrase: str | None = None


# --------------------------------------------------------------------------
# Per-source splitting
# --------------------------------------------------------------------------

ANSWER_MARKER = re.compile(r"উত্তর\s*[:：]")

MIN_WORDS = 8
MAX_WORDS = 220

SENTINEL = "\x00"

# Prothom Alo's column has been migrated between CMSes and each migration left a
# different marker between one reader's letter and the lawyer's reply. Four
# appear in the archive, and some articles mix two:
#
#   Private Use Area chars, mostly U+F032 and U+F06C — Wingdings bullets that
#     lost their font in the migration. This is the majority case, and it is why
#     81 of 92 articles look like one undifferentiated 600-word blob until you
#     look at the codepoints.
#   '*' bullets on both letter and reply.
#   A bare ASCII '2', which is U+F032 downgraded a second time.
#   Explicit প্রশ্ন: / উত্তর: labels.
#
# Normalising all of them to one sentinel is far more robust than branching.
#
# The ASCII '2' rule is safe: Bangla prose writes numbers in Bangla numerals, so
# a lone Latin '2' immediately before Bangla script is the artefact, not a count.
BOUNDARY = re.compile(
    r"[-]+\s*"
    r"|প্রশ্ন\s*[:：]"
    r"|উত্তর\s*[:：]"
    r"|\s\*\s*"
    r"|(?<=\s)2\s*(?=[ঀ-৿])"
)


def _looks_like_question(seg: str) -> bool:
    return "?" in seg and len(seg.split()) >= MIN_WORDS


def _by_boundaries(text: str) -> list[tuple[str, str]]:
    """Split on any letter/reply boundary marker, keep the segments that ask.

    Replies almost never contain a question mark, so selecting on '?' separates
    letters from answers without relying on strict alternation — which does not
    hold in the older migrated articles.
    """
    segments = BOUNDARY.sub(SENTINEL, text).split(SENTINEL)
    if len(segments) < 3:
        return []
    return [(s.strip(), "high") for s in segments[1:] if _looks_like_question(s)]


def _by_answer_marker(text: str) -> list[tuple[str, str]]:
    """Ajker Patrika: one question, then a literal 'উত্তর:'."""
    parts = ANSWER_MARKER.split(text, maxsplit=1)
    if len(parts) != 2:
        return []
    return [(parts[0].strip(), "high")] if _looks_like_question(parts[0]) else []


def _by_question_runs(text: str) -> list[tuple[str, str]]:
    """Fallback: a run of sentences ending at a question mark.

    Used where an article has no structural markers at all. Noisy on purpose —
    it over-captures rather than dropping real questions, and the records are
    labelled low so a human reads them first.
    """
    out: list[tuple[str, str]] = []
    for run in re.findall(r"[^।?]*(?:।[^।?]*){0,4}\?", text):
        if _looks_like_question(run):
            out.append((run.strip(), "low"))
    return out


# Matched by prefix, so adding another Prothom Alo column or Lawyers Club
# category needs no change here.
SPLITTERS = {
    "prothomalo": (_by_boundaries, _by_question_runs),
    "ajkerpatrika": (_by_answer_marker, _by_boundaries, _by_question_runs),
    "dailystar": (_by_question_runs,),
}


def _splitters_for(source: str):
    for prefix, funcs in SPLITTERS.items():
        if source.startswith(prefix):
            return funcs
    return (_by_question_runs,)


# Lawyers Club posts are editorial explainers, not letters. Where the headline is
# itself a plain-Bangla question it is a usable query; the body never is. These
# enter the pool as queries and are never evidence of citizen register, so they
# are marked and can be excluded from any register-based analysis.
TITLE_IS_QUESTION = re.compile(r"[?？]\s*$|কীভাবে|কিভাবে|কী করবেন|করণীয়|কি করবেন")


def split_article(article: dict, index: int) -> Iterator[Question]:
    source = article["source"]
    text = article.get("text", "")

    if source.startswith("lawyersclub"):
        title = article.get("title", "").strip()
        if TITLE_IS_QUESTION.search(title) and len(title.split()) >= 4:
            candidates = [(title, "medium")]
        else:
            candidates = []
    else:
        candidates = []
        for splitter in _splitters_for(source):
            candidates = splitter(text)
            if candidates:
                break

    seen: set[str] = set()
    for n, (raw, confidence) in enumerate(candidates, start=1):
        cleaned = strip_pii(BOILERPLATE.sub(" ", raw))
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" *।")
        words = cleaned.split()
        if not (MIN_WORDS <= len(words) <= MAX_WORDS):
            continue
        key = " ".join(words[:12])
        if key in seen:
            continue
        seen.add(key)
        yield Question(
            qid=f"{source.split('_')[0][:4]}_{index:04d}_{n:02d}",
            question_bn=cleaned,
            source=source,
            source_url=article.get("source_url", ""),
            article_title=article.get("title", ""),
            n_words=len(words),
            split_confidence=confidence,
        )


def split_all(articles: list[dict]) -> list[Question]:
    """Split every article, dropping questions already seen in another source.

    Overlapping searches surface the same letter more than once, and a duplicate
    in the gold set would be counted twice in every metric.
    """
    out: list[Question] = []
    seen: set[str] = set()
    for i, article in enumerate(articles):
        for q in split_article(article, i):
            key = " ".join(q.question_bn.split()[:12])
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
    return out


def write_jsonl(questions: list[Question], dest: pathlib.Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
    return len(questions)


# --------------------------------------------------------------------------
# The lawyer's reply
# --------------------------------------------------------------------------
#
# Every mined question was answered in print by a practising advocate, and that
# reply is already sitting in `mined_articles.jsonl` — the splitter discarded it
# because it selects segments containing a question mark and a reply rarely has
# one.
#
# The reply is worth recovering for one specific reason: it is written in the
# register the statute uses. A reader writes "কো-অপারেটিভের কাছে টাকা আটকে আছে"
# and the advocate answers "দেওয়ানি আইনের আওতায় মানি মোকদ্দমা করতে পারেন". The
# second phrasing is the bridge across the register gap that this whole project
# is about, and for labelling purposes it is a far stronger retrieval query than
# the question alone.
#
# It is NOT a provision label. Advocates name a remedy or an Act, rarely a
# section, and sometimes — as with the affidavit question — they give practical
# advice citing no law at all. Treat it as strong evidence for a human, never as
# the answer itself.
#
# Copyright: the reply is the newspaper's text, exactly like the question. It
# lives in `data/interim/`, which is git-ignored, and never enters the released
# dataset.


def _replies_by_boundaries(text: str) -> list[str]:
    """The reply following each segment `_by_boundaries` would have kept.

    Deliberately mirrors `_by_boundaries` so the nth reply belongs to the nth
    question and the `qid` numbering in `split_article` lines up. Returns an
    empty string where no reply followed, rather than shifting the list.
    """
    segments = BOUNDARY.sub(SENTINEL, text).split(SENTINEL)
    if len(segments) < 3:
        return []
    body = segments[1:]
    replies = []
    for i, segment in enumerate(body):
        if not _looks_like_question(segment):
            continue
        nxt = body[i + 1] if i + 1 < len(body) else ""
        replies.append("" if _looks_like_question(nxt) else nxt.strip())
    return replies


def replies_for_article(article: dict, index: int) -> dict[str, str]:
    """`{qid: reply}` for one article, keyed to match `split_article`."""
    source = article["source"]
    if source.startswith("lawyersclub"):
        return {}
    replies = _replies_by_boundaries(article.get("text", ""))
    prefix = source.split("_")[0][:4]
    out = {}
    for n, reply in enumerate(replies, start=1):
        cleaned = strip_pii(BOILERPLATE.sub(" ", reply))
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" *।")
        if cleaned:
            out[f"{prefix}_{index:04d}_{n:02d}"] = cleaned
    return out
