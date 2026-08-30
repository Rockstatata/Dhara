"""Build the per-annotator TSV sheets, with BM25 candidates already filled in.

    python scripts/06_make_annotation_sheets.py --annotators Adiba Rafi Nusrat Tanvir

One CSV per annotator, ready to import as a tab of a shared Google Sheet. Written
with the csv module and QUOTE_ALL, so the commas and quotation marks that fill
Bangla legal text cannot shift a column, and the ten candidates can sit in one
cell as real newlines instead of a literal backslash-n nobody can read.

Design decisions that live here rather than in the guide:

  - **Candidates come from BM25, never from a dense model.** That biases the gold
    set toward provisions a lexical retriever can already find, which makes this
    project's own thesis *harder* to prove. Conservative in the right direction.
  - **Some questions get no candidate list at all.** They measure how often the
    BM25 list simply missed the answer, and that miss rate is reported.
  - Everyone annotates the same calibration block first, so disagreement surfaces
    before 500 labels have been produced from four private interpretations.
  - The overlap block is double-annotated to yield a real inter-annotator κ.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.retrievers.bm25 import BM25Retriever  # noqa: E402

QUESTIONS = pathlib.Path("data/interim/questions_pool.jsonl")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
OUTDIR = pathlib.Path("data/annotation")

N_CALIBRATION = 20
N_OVERLAP = 150
N_UNAIDED = 15
N_PAIRED = 100
N_CANDIDATES = 10

# Column order is the order an annotator fills them in, left to right. Everything
# left of `answer` is written by this script and must not be hand-edited;
# everything from `answer` rightwards is the annotator's work.
#
# `annotator` is stamped into every row rather than left implicit in the filename,
# so that when the four sheets are concatenated authorship survives the merge and
# each person's contribution stays separately countable in the report.
#
# `answer_source` exists because the candidate list is BM25, and BM25 cannot reach
# an English provision from a Bangla question at all. Without this column an
# answer the annotator found by their own search is indistinguishable from one the
# candidate list supplied, and the BM25 miss rate — a headline number here —
# cannot be computed.
COLUMNS = [
    "qid", "annotator", "block", "annotation_mode", "question_bn", "candidates",
    "answer", "answer_source", "register", "domain", "confidence", "notes",
    "paraphrase", "formal_version",
]


def candidate_block(retriever: BM25Retriever, by_id: dict, question: str) -> str:
    """Ten numbered candidates in one cell, newline-separated.

    One cell rather than ten columns: the annotator reads top to bottom and types
    a number, and the sheet stays narrow enough to use on a laptop.

    Two details that decide whether these ten lines are usable:

    **Deduplicated by provision, not by chunk.** A long provision is split into
    overlapping sub-chunks that share one citation, and BM25 happily returns three
    of them. Three lines reading `ধারা ২` would spend a third of the annotator's
    ten slots re-offering one provision they already rejected. The best-scoring
    sub-chunk represents its provision and the rest are skipped, so ten lines mean
    ten genuinely different answers.

    **Title *and* a body snippet.** Titles are the densest formal legal vocabulary
    in the corpus, which is why they were worth crawling — but the most common
    ones are `সংজ্ঞা` and `প্রয়োগ`, which tell an annotator nothing at all. The
    title says which provision this is; the snippet says whether it answers the
    question.
    """
    lines = []
    seen: set[str] = set()
    # Over-fetch: after collapsing sub-chunks to their parent provision we still
    # want N_CANDIDATES distinct provisions to show.
    for chunk_id, _score in retriever.search(question, k=N_CANDIDATES * 4):
        chunk = by_id[chunk_id]
        if chunk.provision_id in seen:
            continue
        seen.add(chunk.provision_id)
        kind = "অনুচ্ছেদ" if chunk.provision_kind == "article" else "ধারা"
        snippet = " ".join(chunk.text_bn.split())[:110]
        head = f"[{chunk.act_title_bn[:44]}, {kind} {chunk.provision_no_bn}]"
        title = chunk.provision_title_bn
        body = f"{title} — {snippet}" if title else snippet
        lines.append(f"{len(lines) + 1}. <{chunk_id}> {head} {body}")
        if len(lines) == N_CANDIDATES:
            break
    return "\n".join(lines) if lines else "(no candidates)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--questions", type=pathlib.Path, default=QUESTIONS)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--outdir", type=pathlib.Path, default=OUTDIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    chunks = list(schema.read_jsonl(args.corpus))
    by_id = {c.chunk_id: c for c in chunks}
    retriever = BM25Retriever()
    retriever.index(chunks)
    print(f"indexed {len(chunks)} chunks")

    pool = [json.loads(line) for line in args.questions.open(encoding="utf-8")]
    # High-confidence splits first: an annotator's first hour should not be spent
    # on the questions the parser was least sure it had separated correctly.
    pool.sort(key=lambda q: {"high": 0, "medium": 1, "low": 2}[q["split_confidence"]])
    rng = random.Random(args.seed)

    calibration = pool[:N_CALIBRATION]
    rest = pool[N_CALIBRATION:]
    rng.shuffle(rest)

    overlap = rest[:N_OVERLAP]
    single = rest[N_OVERLAP:]
    unaided = {q["qid"] for q in rng.sample(rest, min(N_UNAIDED, len(rest)))}
    paired = {q["qid"] for q in rng.sample(rest, min(N_PAIRED, len(rest)))}

    people = args.annotators
    assigned: dict[str, list[tuple[dict, str]]] = {p: [] for p in people}
    for person in people:
        assigned[person] += [(q, "CALIBRATION") for q in calibration]
    # Each overlap question goes to exactly two people, rotating through pairs.
    for i, q in enumerate(overlap):
        a = people[i % len(people)]
        b = people[(i + 1) % len(people)]
        assigned[a].append((q, "OVERLAP"))
        assigned[b].append((q, "OVERLAP"))
    for i, q in enumerate(single):
        assigned[people[i % len(people)]].append((q, "MAIN"))

    args.outdir.mkdir(parents=True, exist_ok=True)
    for person, rows in assigned.items():
        path = args.outdir / f"annotate_{person}.csv"
        # utf-8-sig: without the BOM, Excel opens Bangla as mojibake, and someone
        # will open one of these in Excel no matter what the guide says.
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
            writer.writerow(COLUMNS)
            for q, block in rows:
                flags = [block]
                if q["qid"] in paired:
                    flags.append("PAIR")
                if q["qid"] in unaided:
                    flags.append("UNAIDED")
                cands = (
                    "(deliberately blank — search the corpus yourself)"
                    if q["qid"] in unaided
                    else candidate_block(retriever, by_id, q["question_bn"])
                )
                mode = "unaided" if q["qid"] in unaided else "assisted"
                writer.writerow(
                    [q["qid"], person, "+".join(flags), mode,
                     q["question_bn"], cands]
                    + [""] * (len(COLUMNS) - 6)
                )
        counts = Counter(b for _, b in rows)
        print(
            f"  {person:10s} {len(rows):4d} rows  "
            f"(calibration {counts['CALIBRATION']}, overlap {counts['OVERLAP']}, "
            f"main {counts['MAIN']})  -> {path}"
        )

    print(
        f"\n{len(pool)} questions | {len(calibration)} calibration (everyone), "
        f"{len(overlap)} double-annotated for κ, {len(unaided)} unaided controls, "
        f"{len(paired)} marked for a formal counterpart"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
