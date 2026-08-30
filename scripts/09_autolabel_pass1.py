"""Fill Pass 1 of the annotation sheets with machine proposals.

    python scripts/09_autolabel_pass1.py
    python scripts/09_autolabel_pass1.py --dry-run
    python scripts/09_autolabel_pass1.py --limit 20

Writes `answer`, `answer_source`, `register`, `domain` and `confidence` into every
sheet, plus a `pass1_auto` column marking the row as machine-filled. Humans then
verify and correct rather than author from scratch, and Pass 2 — the paraphrase
and the formal counterpart — stays entirely human.

**What this is and is not.** These are proposals from a lexical retriever. They
are not adjudicated answers, and the difference matters for two specific numbers
the paper reports:

  - A retriever cannot output "no provision answers this". It always returns its
    best match, so it can never propose `none` or `unanswerable`. Those labels
    exist to calibrate when the system should abstain, and every one of them has
    to come from a human noticing that nothing fits. Rows where this script is
    least sure are exactly where that is most likely, which is why `confidence`
    is derived from the score margin rather than set to a flat value.
  - If a retriever's proposals survive into the gold set unchanged, the
    evaluation compares a retriever against a retriever. `pass1_auto` exists so
    that the fraction of machine labels a human changed is measurable and
    reportable; if that fraction is near zero, the labels were rubber-stamped and
    the evaluation should be described accordingly.

Neither caveat is a reason not to run this — a verified proposal is much faster
than a blank cell, and the corpus is large enough now that most questions do have
an answer in it. They are reasons to keep `pass1_auto` and report against it.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.normalize import aggressive, content_tokens  # noqa: E402
from dhara.retrievers.bm25 import BM25Retriever  # noqa: E402

SHEETS = pathlib.Path("data/annotation")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")

MARKER = re.compile(r"^\s*\d+\.\s*<([^>\s]+)>")

# Vocabulary a citizen does not use unless they have talked to a lawyer, read a
# form, or looked the term up. Taken from the examples in the annotation guide;
# the guide's own tie-break applies, so one such word is not enough — the
# threshold below wants two.
FORMAL_MARKERS = {
    "উচ্ছেদ", "নামজারি", "অগ্রক্রয়", "জামিন", "রিট", "অভিভাবকত্ব", "আইনগত",
    "প্রতিকার", "এখতিয়ার", "মোকদ্দমা", "বাদী", "বিবাদী", "ডিক্রি", "আপিল",
    "তামাদি", "খতিয়ান", "দলিল", "রেজিস্ট্রি", "হেবা", "ওয়ারিশ", "দেনমোহর",
    "ভরণপোষণ", "অর্শ", "নিষেধাজ্ঞা", "সালিশ", "অধিগ্রহণ", "বিধান", "ধারা",
    "উপধারা", "অনুচ্ছেদ", "প্রজ্ঞাপন", "তফসিল", "রহিত", "বিলুপ্ত",
}
FORMAL_THRESHOLD = 2

N_PROPOSE = 3          # never more than the guideline's cap of three
N_RETRIEVE = 40        # over-fetch, then collapse sub-chunks to their provision


def register_of(question: str) -> str:
    """`formal` only when the asker used the vocabulary of law, not of trouble."""
    tokens = set(aggressive(question).split())
    return "formal" if len(tokens & FORMAL_MARKERS) >= FORMAL_THRESHOLD else "colloquial"


def confidence_of(scores: list[float]) -> str:
    """`3` / `2` / `1`, from how far the top hit stands above its rivals.

    A retriever that is barely preferring its winner is a retriever that has not
    found the answer, and those are the rows most likely to want `none`. The
    margin is the only honest signal available without a human, so it is what
    drives the flag that sends a row to a person first.
    """
    if not scores:
        return "1"
    if len(scores) == 1:
        return "2"
    top, second = scores[0], scores[1]
    if top <= 0:
        return "1"
    margin = (top - second) / top
    if top >= 12 and margin >= 0.25:
        return "3"
    if top >= 6 and margin >= 0.10:
        return "2"
    return "1"


def citation_line(chunk: schema.Chunk) -> str:
    """`<chunk_id> [Act title, ধারা N] title — snippet`, matching the format the
    candidate list already uses.

    A bare chunk_id like `other_850_s34` tells a human nothing — they cannot see
    which Act it is without leaving the spreadsheet to look it up. This was
    filed as a real usability complaint against the first version of this
    script's output: the `answer` cell was unreadable while the `candidates`
    cell right next to it was not, for no reason other than this function not
    existing yet. `08_merge_gold.py`'s `resolve()` extracts the `<chunk_id>`
    marker and ignores the rest, so the readable text costs nothing at parse
    time.
    """
    kind = "অনুচ্ছেদ" if chunk.provision_kind == "article" else "ধারা"
    head = f"[{chunk.act_title_bn or chunk.act_title_en}, {kind} {chunk.provision_no_bn}]"
    title = chunk.provision_title_bn
    snippet = " ".join(chunk.text_bn.split())[:100]
    body = f"{title} — {snippet}" if title else snippet
    return f"<{chunk.chunk_id}> {head} {body}"


def propose(retriever, by_id, question, candidate_ids):
    """Return (chunk_ids, source, domain, confidence).

    Candidates already shown on the sheet are preferred when they survive
    retrieval, so a proposal the annotator can see in front of them is not
    replaced by an identical-scoring one they would have to go look up.
    """
    hits, seen = [], set()
    for chunk_id, score in retriever.search(question, k=N_RETRIEVE):
        chunk = by_id[chunk_id]
        if chunk.provision_id in seen:
            continue
        seen.add(chunk.provision_id)
        hits.append((chunk_id, score))
        if len(hits) >= N_PROPOSE:
            break
    if not hits:
        return [], "none", "", "1"
    ids = [c for c, _ in hits]
    scores = [s for _, s in hits]
    on_sheet = set(candidate_ids)
    source = "candidate" if ids[0] in on_sheet else "own_search"
    return ids, source, by_id[ids[0]].domain, confidence_of(scores)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheets", type=pathlib.Path, default=SHEETS)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows per sheet")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument(
        "--overwrite", action="store_true",
        help="refill rows a human has already answered (default: leave them alone)",
    )
    args = ap.parse_args()

    chunks = list(schema.read_jsonl(args.corpus))
    by_id = {c.chunk_id: c for c in chunks}
    retriever = BM25Retriever()
    retriever.index(chunks)
    print(f"indexed {len(chunks)} chunks\n")

    totals = Counter()
    for path in sorted(args.sheets.glob("annotate_*.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        if not rows:
            continue
        columns = list(rows[0].keys())
        if "pass1_auto" not in columns:
            columns.append("pass1_auto")

        filled = skipped = 0
        for n, row in enumerate(rows):
            if args.limit and n >= args.limit:
                break
            if row.get("answer", "").strip() and not args.overwrite:
                skipped += 1
                continue
            candidate_ids = MARKER.findall(row.get("candidates", ""))
            ids, source, domain, conf = propose(
                retriever, by_id, row["question_bn"], candidate_ids
            )
            row["answer"] = (
                "\n".join(citation_line(by_id[cid]) for cid in ids) if ids else "none"
            )
            row["answer_source"] = source
            row["register"] = register_of(row["question_bn"])
            row["domain"] = domain or "other"
            row["confidence"] = conf
            row["pass1_auto"] = "1"
            filled += 1
            totals[f"confidence_{conf}"] += 1
            totals[f"source_{source}"] += 1
            totals[f"register_{row['register']}"] += 1

        if not args.dry_run:
            with path.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                for row in rows:
                    row.setdefault("pass1_auto", "")
                    writer.writerow(row)
        print(f"  {path.name:26s} filled {filled:4d}  left alone {skipped:4d}")

    print("\n" + "-" * 62)
    for key in sorted(totals):
        print(f"  {key:26s} {totals[key]:5d}")
    low = totals["confidence_1"]
    total = sum(v for k, v in totals.items() if k.startswith("confidence_"))
    if total:
        print(
            f"\n  {low}/{total} rows ({100 * low // total}%) came out at confidence 1 — "
            f"the retriever barely preferred its own answer.\n"
            f"  Those are where `none` and `unanswerable` are hiding, and they are "
            f"the rows to verify first."
        )
    if args.dry_run:
        print("\n  dry run — nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
