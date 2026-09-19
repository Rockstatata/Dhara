"""Build the Pass-2 sheets: paraphrase all questions so they can be released.

    python scripts/30_make_paraphrase_sheets.py --annotators Iftiaq Sarwad
    python scripts/30_make_paraphrase_sheets.py --annotators A B --qids data/processed/test_split_v1.json

Writes data/annotation/paraphrase_<name>.csv, one row per question.
Existing rows are retained by qid so rerunning the script never destroys completed
paraphrases; newly assigned questions are appended to each annotator's sheet.

## Why the dataset cannot be released without this

Every mined question is copied verbatim from a newspaper legal-advice column, so
the text belongs to the newspaper. CLAUDE.md's rule: the verbatim text stays
local and git-ignored, and the *paraphrase* ships -- same register, same facts,
different wording -- alongside the `source_url` and the provision label.

`paraphrase` is empty on all 794 round-one rows, so today the releasable dataset
is question ids, URLs and labels: reproducing the benchmark would mean
re-scraping a newspaper. For a paper whose contribution is the benchmark, that is
a weak release.

Scope is all 552 questions: the 200 frozen test questions and the 352 training
questions. With four annotators this is 138 questions each.

## What a good paraphrase preserves

**Register.** These are citizen letters, not legal prose. A paraphrase that
tidies «আমার স্বামী মুখে তালাক দিয়েছে» into formal legal Bangla destroys the
exact property the project measures -- the gap between how people write and how
statutes are written. Keep it colloquial, keep the run-on sentences.

**Length and detail.** The corpus analysis rests on real questions being long
narratives (median 74 words). A paraphrase compressed to one line would make the
released set a different distribution from the one all the results describe.
Keep the background, the timeline, the irrelevant asides.

**Every legal fact.** Amounts, dates, ages, relationships, who did what to whom.
Change the words, never the facts.

**No personal data.** Names, phone numbers, NID numbers, addresses and employer
names come out and are replaced with generic stand-ins. This is the one place a
paraphrase should *lose* information.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

GOLD = pathlib.Path("data/processed/gold_test_v1.jsonl")
SPLIT = pathlib.Path("data/processed/test_split_v1.json")
OUTDIR = pathlib.Path("data/annotation")

COLUMNS = ["qid", "annotator", "n_words", "question_bn", "paraphrase", "pii_removed", "notes"]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_existing(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--gold", type=pathlib.Path, default=GOLD)
    ap.add_argument("--qids", type=pathlib.Path, default=SPLIT)
    ap.add_argument("--outdir", type=pathlib.Path, default=OUTDIR)
    args = ap.parse_args()

    by_qid: dict[str, dict] = {}
    for row in read_jsonl(args.gold):
        by_qid.setdefault(row["qid"], row)

    split = json.loads(args.qids.read_text(encoding="utf-8"))
    wanted = split["test_qids"] + split["train_qids"]
    questions = [by_qid[q] for q in wanted if q in by_qid]
    missing = [qid for qid in wanted if qid not in by_qid]
    if missing:
        ap.error(f"{len(missing)} qids are absent from {args.gold}: {missing[:5]}")
    if len(set(wanted)) != len(wanted):
        ap.error(f"duplicate qids in {args.qids}")
    print(f"{len(questions)} questions to paraphrase")

    # Round-robin, and deliberately NOT the same split as the verification
    # sheets: a second person reading the question is a free extra check on
    # whether the verified label makes sense, and `notes` is where that surfaces.
    sheets: dict[str, list[dict]] = {name: [] for name in args.annotators}
    for i, row in enumerate(questions):
        name = args.annotators[(i + 1) % len(args.annotators)]
        text = " ".join(row["question_bn"].split())
        sheets[name].append({
            "qid": row["qid"],
            "annotator": name,
            "n_words": len(text.split()),
            "question_bn": text,
            "paraphrase": "",
            "pii_removed": "",
            "notes": "",
        })

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, assigned_rows in sheets.items():
        dest = args.outdir / f"paraphrase_{name}.csv"
        existing_rows = read_existing(dest)
        existing_by_qid = {row["qid"]: row for row in existing_rows}
        rows = [existing_by_qid.get(row["qid"], row) for row in assigned_rows]
        retained = sum(row["qid"] in existing_by_qid for row in assigned_rows)
        added = len(rows) - retained
        with dest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        median = sorted(int(r["n_words"]) for r in rows)[len(rows) // 2]
        print(
            f"wrote {dest}  ({len(rows)} questions: {retained} retained, "
            f"{added} added; median {median} words)"
        )

    print("\nThese sheets contain the verbatim newspaper text. They stay local:")
    print("  data/annotation/ is for working files; only the paraphrase column is released.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
