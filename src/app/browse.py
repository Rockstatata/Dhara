"""Read-only data for the two documentation tabs (corpus browser, gold Q&A browser).

Loaded once at module import, same as `Dhara.load()` -- these are large files
(39,484 chunks, 552 gold rows) and re-reading them per request would make the
demo feel frozen, the same trap `service.py` avoids for the encoder.

Never used by the search tab's ranking path -- this is presentation only, over
`text_raw` (the law as printed), never `text_bn` (what models consume).
"""

from __future__ import annotations

import json
import pathlib
import re

from dhara import schema

CORPUS_FILE = pathlib.Path("data/processed/corpus_v1.jsonl")
GOLD_FILE = pathlib.Path("data/processed/gold_verified_v3.jsonl")

_DIGITS = re.compile(r"\d+")


def _sort_key(provision_no_ascii: str) -> tuple:
    # "103" before "12A" before "2" -- numeric-first, then whatever text follows.
    m = _DIGITS.match(provision_no_ascii or "")
    if m:
        return (0, int(m.group()), provision_no_ascii)
    return (1, 0, provision_no_ascii or "")


def build_act_index() -> tuple[dict, list[tuple[str, str]]]:
    """Returns (act_id -> {title, year, act_no, provisions: [...]}, dropdown choices).

    Sub-chunks of a split provision (sub_idx/n_sub) are rejoined into one entry
    so a long section reads as one continuous provision, matching how a reader
    of the actual law would expect to see it -- not as N arbitrary fragments.
    """
    acts: dict[str, dict] = {}
    provisions: dict[str, dict] = {}  # provision_id -> assembled entry

    for chunk in schema.read_jsonl(CORPUS_FILE):
        act = acts.setdefault(chunk.act_id, {
            "title_bn": chunk.act_title_bn, "title_en": chunk.act_title_en,
            "year": chunk.act_year, "act_no": chunk.act_no, "provision_ids": [],
        })
        entry = provisions.get(chunk.provision_id)
        if entry is None:
            entry = {
                "provision_id": chunk.provision_id, "kind": chunk.provision_kind,
                "no_bn": chunk.provision_no_bn, "no_ascii": chunk.provision_no_ascii,
                "title_bn": chunk.provision_title_bn or "", "domain": chunk.domain,
                "risk_tier": chunk.risk_tier, "source_url": chunk.source_url,
                "parts": {},
            }
            provisions[chunk.provision_id] = entry
            act["provision_ids"].append(chunk.provision_id)
        entry["parts"][chunk.sub_idx] = chunk.text_raw or chunk.text_bn

    for entry in provisions.values():
        entry["text"] = "\n\n".join(entry["parts"][i] for i in sorted(entry["parts"]))
        del entry["parts"]

    for act in acts.values():
        act["provisions"] = sorted(
            (provisions[pid] for pid in act["provision_ids"]),
            key=lambda p: _sort_key(p["no_ascii"]),
        )
        del act["provision_ids"]

    choices = sorted(
        ((f"{a['title_bn'] or a['title_en']} ({a['year'] or '—'}) [{len(a['provisions'])} ধারা]", act_id)
         for act_id, a in acts.items()),
        key=lambda c: c[0],
    )
    return acts, choices


def render_act_markdown(act: dict) -> str:
    lines = [f"# {act['title_bn'] or act['title_en']}"]
    meta = []
    if act.get("title_en") and act.get("title_en") != act.get("title_bn"):
        meta.append(act["title_en"])
    if act.get("year"):
        meta.append(str(act["year"]))
    if act.get("act_no"):
        meta.append(f"Act No. {act['act_no']}")
    if meta:
        lines.append("*" + " · ".join(meta) + "*")
    lines.append(f"\n**{len(act['provisions'])} provisions**\n\n---")
    kind_label = {"article": "অনুচ্ছেদ", "section": "ধারা"}
    for p in act["provisions"]:
        label = kind_label.get(p["kind"], p["kind"])
        title = f" — {p['title_bn']}" if p["title_bn"] else ""
        lines.append(f"### {label} {p['no_bn']}{title}")
        if p.get("risk_tier") == "high":
            lines.append("`উচ্চ-ঝুঁকি ডোমেইন`")
        lines.append(p["text"])
        lines.append("\n---")
    return "\n\n".join(lines)


def build_gold_table() -> tuple[list[list], list[str]]:
    """Rows for the gold Q&A browser: real citizen questions with their
    adjudicated answer provision(s), resolved to a citation via the corpus."""
    provision_by_id: dict[str, schema.Chunk] = {}
    for chunk in schema.read_jsonl(CORPUS_FILE):
        provision_by_id.setdefault(chunk.provision_id, chunk)

    rows = []
    with GOLD_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            pids = r.get("relevant_provision_ids") or []
            citations = [provision_by_id[p].citation() for p in pids if p in provision_by_id]
            rows.append([
                r.get("domain", ""),
                "মিলিত" if r.get("answerable") else "উত্তরহীন (abstention gold)",
                r.get("register", ""),
                r.get("question_bn", ""),
                "; ".join(citations) or "—",
                (r.get("answer") or "")[:220],
                r.get("source", ""),
            ])
    columns = ["ডোমেইন", "উত্তরযোগ্য", "রেজিস্টার", "প্রশ্ন (বাংলা)", "উত্তর — ধারা", "উত্তরের সংক্ষিপ্তসার", "উৎস"]
    return rows, columns
