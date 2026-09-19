"""Rank gold labels by how likely they are to be wrong, using the lawyer's reply.

    python scripts/27_label_triage.py
    python scripts/27_label_triage.py --limit 50        # smoke test

Writes results/runs/label_triage.json and results/tables/label_triage.csv, the
second sorted worst-first so a human re-checks the most suspicious labels first.

## Why this exists

Eyeballing said some gold labels look wrong. Eyeballing cannot say *how many*,
and the number matters: at a 5% error rate the evaluation stands with a footnote,
at 30% every reported figure is noise around an unknown centre.

Two facts make an automatic triage possible without new human work:

  - Every mined question came from a newspaper legal-advice column, so a
    practising advocate's published reply exists for 491 of the gold rows.
  - That reply is written in formal legal register — «দেওয়ানি আইনের আওতায় মানি
    মোকদ্দমা করতে পারেন» — which is the vocabulary of the statute rather than of
    the citizen. `scripts/11_extract_answers.py` recovered these for exactly this
    reason.

So the reply is a second, independent query for the same information need, and a
much easier one for a retriever than the citizen's letter. If the labelled
provision is nowhere near the top when the *advocate's own answer* is the query,
the label is suspect.

## What the signals mean, and what they do not

`lawyer_rank` — rank of the labelled provision when the lawyer's reply is the
query. Good labels should rank well; this is the primary signal.

`lawyer_cosine` — similarity between the reply and the labelled provision text,
which catches the case where everything ranks badly because the reply is short.

**This is triage, not adjudication.** A label can be correct and still score
badly here: advocates often answer with practical steps and cite no law at all
("go to the police station, file a GD"), and a reply about procedure will not
resemble the substantive provision that governs the right. The output is a
priority order for human re-checking, and the only honest way to turn it into an
error rate is to have a person adjudicate a sample from each band — which is what
`scripts/25_make_verification_sheets.py` produces.

Nothing here writes a label. The gold set is not modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402

INDEX = pathlib.Path("models/index_bge_m3_zeroshot_v1")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
SHEETS = pathlib.Path("data/annotation")
LAWYER = pathlib.Path("data/interim/lawyer_answers.jsonl")
OUT_RUN = pathlib.Path("results/runs/label_triage.json")
OUT_TABLE = pathlib.Path("results/tables/label_triage.csv")

# Bands are reported separately rather than collapsed into one "suspect" count,
# because the middle band is exactly where a heuristic is least trustworthy.
GOOD_RANK = 10
SUSPECT_RANK = 200


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_lawyer_answers() -> dict[str, str]:
    """Prefer the interim file; fall back to the column in the sheets."""
    answers: dict[str, str] = {}
    if LAWYER.exists():
        for row in read_jsonl(LAWYER):
            text = (row.get("answer") or row.get("lawyer_answer") or "").strip()
            if text:
                answers[row["qid"]] = text
    if not answers:
        for path in sorted(SHEETS.glob("annotate_*.csv")):
            with path.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    text = (row.get("lawyer_answer") or "").strip()
                    if text:
                        answers.setdefault(row["qid"], text)
    return answers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=pathlib.Path, default=INDEX)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--probes", type=pathlib.Path, default=PROBES)
    ap.add_argument("--limit", type=int, default=0, help="triage only the first N (smoke test)")
    args = ap.parse_args()

    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    C = np.load(args.index / "embeddings.npy").astype(np.float32)
    cid = json.loads((args.index / "chunk_ids.json").read_text(encoding="utf-8"))

    chunks = {c.chunk_id: c for c in schema.read_jsonl(args.corpus)}
    cprov = np.array([chunks[c].provision_id for c in cid])

    probes = read_jsonl(args.probes)
    answers = load_lawyer_answers()
    todo = [p for p in probes if answers.get(p["qid"])]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(probes)} probe questions, {len(todo)} with a lawyer's published reply")
    if not todo:
        raise SystemExit("no lawyer answers found — nothing to triage")

    from sentence_transformers import SentenceTransformer

    print(f"encoding {len(todo)} replies with {manifest['checkpoint']} (CPU)")
    model = SentenceTransformer(manifest["checkpoint"])
    model.max_seq_length = manifest["max_seq_length"]
    prefix = manifest.get("query_prefix", "")
    Q = model.encode([prefix + answers[p["qid"]] for p in todo], batch_size=8,
                     normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=True).astype(np.float32)

    rows = []
    for i, probe in enumerate(todo):
        scores = C @ Q[i]
        order = np.argsort(-scores)[:SUSPECT_RANK]
        gold = set(probe["gold_provision_ids"])
        rank = next((r for r, j in enumerate(order, 1) if cprov[j] in gold), None)

        gold_idx = [k for k, c in enumerate(cid) if cprov[k] in gold]
        cosine = float(max(scores[j] for j in gold_idx)) if gold_idx else None

        band = ("good" if rank and rank <= GOOD_RANK
                else "middle" if rank else "suspect")
        first = chunks.get(probe["gold_chunk_ids"][0])
        rows.append({
            "qid": probe["qid"],
            "band": band,
            "lawyer_rank": rank,
            "lawyer_cosine": round(cosine, 4) if cosine is not None else None,
            "lang_tag": probe["lang_tag"],
            "labelled_act": (first.act_title_bn or "")[:60] if first else "",
            "labelled_provision": (first.provision_title_bn or "")[:60] if first else "",
            "question": " ".join(probe["question_bn"].split())[:120],
            "lawyer_answer": " ".join(answers[probe["qid"]].split())[:120],
        })

    bands = {b: [r for r in rows if r["band"] == b] for b in ("good", "middle", "suspect")}
    cosines = [r["lawyer_cosine"] for r in rows if r["lawyer_cosine"] is not None]

    print(f"\ntriage over {len(rows)} labels, using the advocate's reply as the query")
    for name, band in bands.items():
        share = len(band) / len(rows)
        print(f"  {name:8s} {len(band):4d}  ({share:.1%})", end="")
        if band:
            band_cos = [r['lawyer_cosine'] for r in band if r['lawyer_cosine'] is not None]
            if band_cos:
                print(f"   median cosine {sorted(band_cos)[len(band_cos)//2]:.3f}")
            else:
                print()
        else:
            print()
    if cosines:
        print(f"  overall median cosine to the labelled provision: "
              f"{sorted(cosines)[len(cosines)//2]:.3f}")

    print("\nWorst 10 — re-check these first:")
    order = sorted(rows, key=lambda r: (r["lawyer_cosine"] if r["lawyer_cosine"] is not None else -1))
    for r in order[:10]:
        print(f"  {r['qid']}  cos={r['lawyer_cosine']}  rank={r['lawyer_rank']}")
        print(f"      Q: {r['question'][:88]}")
        print(f"      labelled: [{r['labelled_act'][:40]} | {r['labelled_provision'][:36]}]")

    report = {
        "run_id": "label_triage",
        "index": str(args.index),
        "n_labels_triaged": len(rows),
        "n_probes": len(probes),
        "coverage": round(len(rows) / len(probes), 4),
        "bands": {b: len(v) for b, v in bands.items()},
        "band_shares": {b: round(len(v) / len(rows), 4) for b, v in bands.items()},
        "median_cosine": round(sorted(cosines)[len(cosines) // 2], 4) if cosines else None,
        "thresholds": {"good_rank": GOOD_RANK, "suspect_rank": SUSPECT_RANK},
        "note": (
            "Triage only. A correct label can land in 'suspect' when the advocate answered "
            "with practical steps and cited no law. Turn this into an error rate by having a "
            "human adjudicate a sample from each band, not by counting the bands."
        ),
        "rows": rows,
    }
    OUT_RUN.parent.mkdir(parents=True, exist_ok=True)
    OUT_RUN.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", newline="", encoding="utf-8") as fh:
        fields = ["qid", "band", "lawyer_rank", "lawyer_cosine", "lang_tag",
                  "labelled_act", "labelled_provision", "question", "lawyer_answer"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(order)

    print(f"\nwrote {OUT_RUN}")
    print(f"wrote {OUT_TABLE}  (sorted worst-first)")


if __name__ == "__main__":
    main()
