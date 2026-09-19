"""Build blind re-adjudication sheets, with no machine answer pre-filled.

    python scripts/25_make_verification_sheets.py --annotators Iftiaq Sarwad
    python scripts/25_make_verification_sheets.py --annotators A B --n-single 60 --n-double 40

Writes data/annotation/verify_<name>.csv per annotator, plus
data/annotation/verification_key_v1.jsonl — the existing machine labels, kept
OUT of the sheets so nobody can see them while annotating.

## Why a second annotation round is needed

`scripts/24_gold_audit.py` measured what came back from round one:

  - 794 of 794 rows are machine-filled (`pass1_auto = 1`);
  - 35 rows carry any recorded human action (20 `own_search` in the unaided
    block, 15 `candidate`);
  - `verified` is empty on every row, and no note was written by a person;
  - across 170 double-annotated questions and 270 annotator pairs, **zero**
    answers differ.

Zero disagreement over 270 pairs is not agreement, it is the signature of sheets
distributed with identical machine answers already in the `answer` column and
returned untouched. Inter-annotator kappa cannot be computed from that, and the
gold set cannot honestly be described as adjudicated.

Round one is not wasted — a verified machine proposal is a legitimate and much
cheaper way to build a gold set than authoring from a blank cell. What is
missing is the verification, and the point of this script is to make the second
pass measurable in a way the first was not.

## What makes these sheets different

**No `answer` column is pre-filled.** The annotator reads the question, reads
the ten BM25 candidates, and writes an answer or `none`. What round one proposed
is written to a separate key file this script keeps out of the sheets.

**Two blocks, two different measurements.**

  - `VERIFY` — each question goes to exactly one annotator. Comparing these
    answers against the key measures how often the machine proposal was right,
    which is the number that decides whether the existing 552-question
    evaluation stands as-is, stands with a caveat, or has to be rebuilt.
  - `DOUBLE` — each question goes to two annotators who never see each other's
    sheet. This is the block that yields a real inter-annotator kappa, and it
    only works because the answer cell starts empty.

**`prior_verdict` is deliberately absent too.** Telling an annotator that the
machine called a question unanswerable would anchor exactly the judgement the
abstention threshold is calibrated on.

## Stratified by triage band, not uniform

With `--triage results/runs/label_triage.json` the sample is drawn evenly from
the three bands `scripts/27_label_triage.py` assigns, instead of uniformly at
random. The bands come from using the advocate's published reply as a second,
independent query for the same information need:

| band | meaning | share of the 330 triaged |
|---|---|---|
| `good` | labelled provision in the top 10 for that query | 47.6% |
| `middle` | ranks 11-200 | 28.2% |
| `suspect` | not in the top 200 at all | 24.2% |

Uniform sampling would spend most of the annotators' effort on labels that are
already corroborated. Even sampling gives an accuracy estimate *per band*, and
the band sizes are known, so a weighted average reconstructs the overall error
rate with far more precision for the same number of judgements. It also front-
loads the labels most likely to need fixing.

Sample sizes default to 100 single + 60 double. At 60 double-annotated questions
a kappa is reportable, and 100 single is enough to put a +/-10 point interval on
the machine-proposal accuracy — enough to tell "mostly right" from "coin flip",
which is the decision at hand.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.retrievers.bm25 import BM25Retriever  # noqa: E402

GOLD = pathlib.Path("data/processed/gold_test_v1.jsonl")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
OUTDIR = pathlib.Path("data/annotation")
KEY = OUTDIR / "verification_key_v1.jsonl"

N_CANDIDATES = 10

# Same left-to-right fill order as round one, minus everything the machine used
# to fill. `answer` is the first column the annotator touches.
COLUMNS = [
    "qid", "annotator", "block", "question_bn", "candidates",
    "answer", "answer_source", "verdict", "confidence", "notes",
]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def candidate_block(retriever: BM25Retriever, by_id: dict, question: str) -> str:
    """Ten distinct provisions, one per line, exactly as round one presented them.

    Kept identical to `scripts/06_make_annotation_sheets.py` on purpose: if the
    second pass showed a different candidate list, a disagreement with round one
    could not be attributed to the annotator rather than to the list.
    """
    lines: list[str] = []
    seen: set[str] = set()
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--gold", type=pathlib.Path, default=GOLD)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--outdir", type=pathlib.Path, default=OUTDIR)
    ap.add_argument("--n-single", type=int, default=100, help="questions annotated once")
    ap.add_argument("--n-double", type=int, default=60, help="questions annotated twice")
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--qids", type=pathlib.Path, default=None,
                    help="test_split_v1.json — restrict the pool to one of its qid lists")
    ap.add_argument("--prefix", default="verify",
                    help="output filename prefix, e.g. `verify2` for wave 2")
    ap.add_argument("--qids-key", default="test_qids", choices=["test_qids", "train_qids"],
                    help="which list to take from --qids. `train_qids` builds wave 2: the "
                         "352 questions that become real training pairs once verified")
    ap.add_argument("--triage", type=pathlib.Path, default=None,
                    help="results/runs/label_triage.json — stratify the sample evenly "
                         "across its bands instead of sampling uniformly")
    args = ap.parse_args()

    if len(args.annotators) < 2 and args.n_double:
        ap.error("--annotators needs at least two names when --n-double > 0")

    gold = read_jsonl(args.gold)
    # One row per question. Round one's duplicates are identical anyway, which is
    # the finding that prompted this script.
    by_qid: dict[str, dict] = {}
    for row in gold:
        by_qid.setdefault(row["qid"], row)
    questions = sorted(by_qid.values(), key=lambda r: r["qid"])
    print(f"gold: {len(gold)} rows over {len(questions)} distinct questions")

    if args.qids:
        # Verifying the frozen test set in full beats sampling it: a test set with
        # unmeasured label errors is the problem this round exists to remove, and
        # an error rate estimated from a sample still leaves every individual
        # label unverified.
        wanted = set(json.loads(args.qids.read_text(encoding="utf-8"))[args.qids_key])
        questions = [q for q in questions if q["qid"] in wanted]
        print(f"restricted to {args.qids_key}: {len(questions)} questions (from {args.qids})")
        assert questions, "no gold question matched the qid list"


    rng = random.Random(args.seed)
    need = args.n_single + args.n_double

    if args.triage and args.triage.exists():
        # Even draw per band. The bands have very different sizes, so this is a
        # deliberately unrepresentative sample -- the point is to estimate accuracy
        # *within* each band and weight by the known band sizes afterwards, which
        # is more precise per judgement than a uniform sample of the same size.
        triage = json.loads(args.triage.read_text(encoding="utf-8"))
        band_of = {r["qid"]: r["band"] for r in triage["rows"]}
        by_band: dict[str, list[dict]] = {}
        for row in questions:
            by_band.setdefault(band_of.get(row["qid"], "untriaged"), []).append(row)
        for rows_ in by_band.values():
            rng.shuffle(rows_)

        order = ["suspect", "middle", "good", "untriaged"]
        per_band = need // len([b for b in order if by_band.get(b)])
        pool = []
        for band in order:
            pool.extend(by_band.get(band, [])[:per_band])
        # Top up from whatever is left if a band was too small to fill its share.
        if len(pool) < need:
            chosen = {r["qid"] for r in pool}
            for band in order:
                for row in by_band.get(band, []):
                    if row["qid"] not in chosen:
                        pool.append(row)
                        chosen.add(row["qid"])
                    if len(pool) >= need:
                        break
                if len(pool) >= need:
                    break
        rng.shuffle(pool)
        counts = collections.Counter(band_of.get(r["qid"], "untriaged") for r in pool[:need])
        print(f"stratified by triage band: {dict(counts)}")
    else:
        pool = questions[:]
        rng.shuffle(pool)
        if args.triage:
            print(f"note: {args.triage} not found — sampling uniformly instead")

    assert len(pool) >= need, f"only {len(pool)} questions available, need {need}"
    single = pool[: args.n_single]
    double = pool[args.n_single: need]

    chunks = list(schema.read_jsonl(args.corpus))
    by_id = {c.chunk_id: c for c in chunks}
    retriever = BM25Retriever()
    retriever.index(chunks)
    print(f"BM25 indexed over {len(chunks)} chunks")

    # Assign: single questions round-robin, double questions to two different
    # annotators each so the pairs are spread rather than always the same two.
    sheets: dict[str, list[dict]] = {name: [] for name in args.annotators}
    candidates_cache: dict[str, str] = {}

    def candidates_for(row: dict) -> str:
        if row["qid"] not in candidates_cache:
            candidates_cache[row["qid"]] = candidate_block(retriever, by_id, row["question_bn"])
        return candidates_cache[row["qid"]]

    def emit(row: dict, annotator: str, block: str) -> None:
        sheets[annotator].append({
            "qid": row["qid"],
            "annotator": annotator,
            "block": block,
            "question_bn": row["question_bn"],
            "candidates": candidates_for(row),
            "answer": "", "answer_source": "", "verdict": "",
            "confidence": "", "notes": "",
        })

    for i, row in enumerate(single):
        emit(row, args.annotators[i % len(args.annotators)], "VERIFY")

    n = len(args.annotators)
    for i, row in enumerate(double):
        first = args.annotators[i % n]
        second = args.annotators[(i + 1 + (i // n)) % n]
        if second == first:                     # only possible when n == 1, guarded above
            second = args.annotators[(i + 1) % n]
        emit(row, first, "DOUBLE")
        emit(row, second, "DOUBLE")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in sheets.items():
        rng.shuffle(rows)                       # VERIFY and DOUBLE interleaved, so the
        dest = args.outdir / f"{args.prefix}_{name}.csv"   # double-annotated ones are not obvious
        with dest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        blocks = {b: sum(1 for r in rows if r["block"] == b) for b in ("VERIFY", "DOUBLE")}
        print(f"wrote {dest}  ({len(rows)} rows: {blocks})")

    # The key never enters a sheet. It is what round two is scored against. The
    # triage band travels with it so accuracy can be reported per band and then
    # weighted by the band sizes, which is the whole point of stratifying.
    band_lookup = {}
    if args.triage and args.triage.exists():
        band_lookup = {r["qid"]: r["band"]
                       for r in json.loads(args.triage.read_text(encoding="utf-8"))["rows"]}

    key_path = KEY if args.prefix == "verify" else KEY.with_name(f"{args.prefix}_key_v1.jsonl")
    with key_path.open("w", encoding="utf-8") as fh:
        for row in single + double:
            fh.write(json.dumps({
                "qid": row["qid"],
                "prior_answer_provision_ids": row.get("relevant_provision_ids") or [],
                "prior_verdict": row.get("verdict"),
                "prior_annotator": row.get("annotator"),
                "block": "VERIFY" if row in single else "DOUBLE",
                "triage_band": band_lookup.get(row["qid"]),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {key_path}  ({len(single) + len(double)} rows) — do NOT share this with annotators")

    print("\nWhat to tell the annotators:")
    print("  - Answer from the question and the ten candidates. Write chunk ids, comma-separated.")
    print("  - If the candidates do not contain the answer, search the corpus yourself and")
    print("    write `own_search` in answer_source. If nothing in the law answers it, write")
    print("    `none` in answer and `unanswerable` in verdict.")
    print("  - Do not consult round one's sheets, and do not compare sheets with each other.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
