"""Score hand-authored provision-summary batches and write them as dataset rows.

    python scripts/70_build_provision_summaries.py --batch 1

Input is a `data/processed/summaries_v1_batch<NNN>.py` module holding a `BATCH`
list of `(chunk_id, summary_bn)` pairs. Summaries are written by hand, one per
provision, against that provision's actual text - this script does not generate
anything, it measures each row and writes the JSONL plus a per-batch report.

This is the summarization-dataset counterpart to
`scripts/65_build_authored_questions.py`, and it is deliberately conservative
about what it checks automatically: sentence-count, factual completeness, and
hallucination cannot be verified by a script, only by a human reading the
summary against the source. What this script CAN check without a human -
length band, script majority, a best-effort number-preservation heuristic, and
duplicate-summary hygiene - it checks, and everything else is left at
`review_status: needs_human_check` rather than silently assumed correct.

Rows are labelled `label_source: llm_authored_v1`, `verified: false`. That
distinction is load-bearing (DECISIONS.md 2026-09-18): an LLM-authored summary
honestly labelled as such is fine, one passed off as human-verified is not -
the same rule this project already applies to authored retrieval questions and
to the gold set.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pathlib
import re
from typing import Any

import sys

sys.stdout.reconfigure(encoding="utf-8")

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
BATCH_DIR = pathlib.Path("data/processed")
OUT_DIR = pathlib.Path("data/processed/summaries_v1")

# "2-4 Bangla sentences" approximated as a word-count band rather than literal
# sentence-splitting (Bangla sentence boundaries are the same danda-token
# question normalize.py already handles elsewhere in this repo). Band picked
# from a handful of the batch's own summaries written to genuinely read as
# 2-4 sentences, not reverse-engineered to make a target pass.
SUMMARY_MIN_WORDS = 15
SUMMARY_MAX_WORDS = 95

BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
DIGIT_RE = re.compile(r"\d+")
# Footnote/amendment markers (`22[`, `95[* * *]`) and sub-section labels
# (`(1)`, `(২)`) are structural noise, not facts a summary needs to repeat -
# stripped before counting "numbers in the source" so the preservation
# heuristic tracks amounts/dates/counts, not BLAD's footnote apparatus.
FOOTNOTE_MARKER_RE = re.compile(r"\d+\[")
SUBSECTION_LABEL_RE = re.compile(r"\(\s*\d+\s*\)")

# Bangla legal prose commonly spells small counts as words rather than
# digits, especially in flowing summary sentences ("অনধিক ছয় মাস" rather
# than "৬ মাস") even when the source statute itself writes both ("৬(ছয়)").
# A purely-digit heuristic calls that a dropped number when it was not -
# mapped here so "same numbers on both sides" survives normal Bangla prose,
# not just digit-for-digit transcription. Deliberately small (frequent
# penalty/count words only): this is a warning heuristic, not a parser.
NUMBER_WORDS = {
    "একজন": "1", "একটি": "1", "একটা": "1", "একবছর": "1",
    "দুই": "2", "তিন": "3", "চার": "4", "পাঁচ": "5",
    "ছয়": "6", "সাত": "7", "আট": "8", "নয়": "9", "দশ": "10",
    "এগারো": "11", "এগার": "11", "বারো": "12", "তেরো": "13",
    "চৌদ্দ": "14", "পনেরো": "15", "পনের": "15", "ষোলো": "16", "সতেরো": "17",
    "আঠারো": "18", "আঠার": "18", "উনিশ": "19", "বিশ": "20", "ত্রিশ": "30",
    "চল্লিশ": "40", "পঁয়তাল্লিশ": "45", "পঞ্চাশ": "50", "ষাট": "60",
    "সত্তর": "70", "আশি": "80", "নব্বই": "90", "একশ": "100", "একশো": "100",
    "দুইশ": "200", "তিনশ": "300", "পাঁচশ": "500", "হাজার": "1000",
    "লক্ষ": "100000", "লাখ": "100000",
}
# Word-boundary anchored (Bangla is space-delimited like Latin scripts) so a
# number word matches only as a standalone word, not as a substring of an
# unrelated word - "দ্বারা" ("by/through") contains the letters of "বার"
# ("occasion"/12) but is not the number 12, and bare "এক"/"বার" are common
# word-fragments inside other words, which is why they are excluded or only
# included as fixed compounds ("একজন", "বারো") above.
NUMBER_WORD_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(w) for w in NUMBER_WORDS) + r")(?!\w)"
)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bangla_majority(text: str) -> bool:
    bangla = sum("ঀ" <= char <= "৿" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    return bangla >= latin and bangla > 0


def extract_numbers(text: str) -> set[str]:
    normalized = text.translate(BANGLA_DIGITS)
    normalized = FOOTNOTE_MARKER_RE.sub(" ", normalized)
    normalized = SUBSECTION_LABEL_RE.sub(" ", normalized)
    digits = set(DIGIT_RE.findall(normalized))
    words = {NUMBER_WORDS[match] for match in NUMBER_WORD_RE.findall(text)}
    return digits | words


def load_batch(path: pathlib.Path) -> list[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location("summary_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BATCH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()

    corpus_rows = read_jsonl(args.corpus)
    corpus = {row["chunk_id"]: row for row in corpus_rows}

    batch_path = BATCH_DIR / f"summaries_v1_batch{args.batch:03d}.py"
    batch = load_batch(batch_path)

    seen_summaries: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    kills: collections.Counter = collections.Counter()

    for index, (chunk_id, summary) in enumerate(batch, 1):
        summary = " ".join(summary.split())
        document = corpus.get(chunk_id)
        failures: list[str] = []

        if document is None:
            failures.append("unknown_chunk_id")
            kills["unknown_chunk_id"] += 1
            rows.append({
                "id": f"summary_v1_b{args.batch:03d}_{index:04d}",
                "chunk_id": chunk_id,
                "summary_bn": summary,
                "gate_failures": failures,
                "eligible_for_dataset": False,
                "batch": args.batch,
            })
            continue

        source_text = document.get("text_raw") or document.get("text_bn") or ""
        source_title = document.get("provision_title_bn") or ""
        word_count = len(summary.split())

        if not bangla_majority(summary):
            failures.append("not_bangla_majority")
        if not (SUMMARY_MIN_WORDS <= word_count <= SUMMARY_MAX_WORDS):
            failures.append("length_outside_band")
        normalized_summary = " ".join(summary.casefold().split())
        if normalized_summary in seen_summaries and seen_summaries[normalized_summary] != chunk_id:
            failures.append(f"duplicate_of_{seen_summaries[normalized_summary]}")
        seen_summaries.setdefault(normalized_summary, chunk_id)

        source_numbers = extract_numbers(source_text)
        summary_numbers = extract_numbers(summary)
        numbers_preserved = (not source_numbers) or bool(source_numbers & summary_numbers)
        if source_numbers and not numbers_preserved:
            kills["numbers_dropped_warning"] += 1  # warning only, not a hard gate

        row = {
            "id": f"summary_v1_b{args.batch:03d}_{index:04d}",
            "chunk_id": chunk_id,
            "provision_id": document["provision_id"],
            "act_id": document["act_id"],
            "act_title_bn": document.get("act_title_bn"),
            "domain": document["domain"],
            "source_language": document.get("language"),
            "provision_title_bn": source_title,
            "source_text": source_text,
            "summary_bn": summary,
            "source_word_count": len(source_text.split()),
            "summary_word_count": word_count,
            "source_numbers": sorted(source_numbers),
            "summary_numbers": sorted(summary_numbers),
            "numbers_preserved": numbers_preserved,
            "label_source": "llm_authored_v1",
            "review_status": "needs_human_check",
            "verified": False,
            "split": None,
            "gate_failures": failures,
            "eligible_for_dataset": not failures,
            "batch": args.batch,
        }
        rows.append(row)
        for failure in failures:
            kills[failure] += 1

    out_path = args.out_dir / f"batch{args.batch:03d}.jsonl"
    write_jsonl(out_path, rows)

    eligible = [row for row in rows if row["eligible_for_dataset"]]
    report = {
        "batch": args.batch,
        "attempted": len(rows),
        "eligible": len(eligible),
        "yield": round(len(eligible) / len(rows), 4) if rows else 0.0,
        "domains": collections.Counter(row.get("domain") for row in eligible).most_common(),
        "source_languages": collections.Counter(row.get("source_language") for row in eligible).most_common(),
        "numbers_preserved_rate": round(
            sum(1 for row in eligible if row.get("numbers_preserved")) / len(eligible), 4
        ) if eligible else None,
        "kills_per_gate": dict(kills),
        "failing_rows": [
            {"id": row["id"], "chunk_id": row["chunk_id"], "gates": row["gate_failures"]}
            for row in rows if not row["eligible_for_dataset"]
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
