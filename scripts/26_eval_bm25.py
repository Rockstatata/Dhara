"""Tune BM25 on dev, score it on the probe set, emit runs in script 13's schema.

    python scripts/26_eval_bm25.py                      # tune, then score tuned + default
    python scripts/26_eval_bm25.py --no-tune            # score the default 1.5/0.75 only
    python scripts/26_eval_bm25.py --dev-sample 100     # faster sweep

Writes results/runs/bm25_default.json, results/runs/bm25_tuned.json and the
matching recall tables.

## Why this exists

The lexical floor was being quoted at R@10 = 0.067 from a scratchpad helper that
never wrote a run artifact, with `k1`/`b` left at library defaults. Two problems,
both fatal for a paper:

  - **No `results/runs/*.json`, so no `per_query`.** Every other rung can be
    compared with a paired bootstrap; the baseline could not be compared with
    anything. A number quoted from a deleted scratch script is not reproducible.
  - **Untuned.** `src/dhara/retrievers/bm25.py` says it in its own header: "an
    untuned BM25 that loses to our fine-tuned encoder would prove nothing". A
    reviewer's first question about a neural-beats-lexical claim is whether the
    lexical side was tuned, and "we used the defaults" ends the discussion badly.

## What is tuned on what

`k1` and `b` are swept on the **synthetic dev split**, never on the probe
questions. The dev questions are template-generated and their labels are the
hand-anchored provisions, so they are a legitimate tuning surface that shares no
question with the evaluation set. This keeps the gold-isolation rule intact: the
probe set is touched exactly once per configuration, after the choice is frozen.

Tuning on dev has a known cost worth stating. Dev questions are synthetic and the
probe questions are real, so a hyperparameter that suits one may not suit the
other. The alternative — tuning on the probe set — would inflate the baseline in
a way that flatters this project's own thesis in reverse, and is not an option.

Both the tuned and the default configuration are scored and both are written out.
If tuning turns out not to help, that is a result, and it is one this script can
show rather than assert.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.metrics import CUTOFFS, first_hit, summarize  # noqa: E402
from dhara.retrievers.bm25 import BM25Retriever  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
DEV = pathlib.Path("data/processed/dev_anchor_v1.jsonl")
OUT_RUNS = pathlib.Path("results/runs")
OUT_TABLES = pathlib.Path("results/tables")
GROUPS = ("all", "english", "bengali", "mixed")

K1_GRID = (0.9, 1.2, 1.5, 1.8)
B_GRID = (0.4, 0.6, 0.75, 0.9)
DEPTH = 100


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_questions(retriever: BM25Retriever, questions: list[dict],
                    provision_of: dict[str, str], depth: int = DEPTH) -> list[dict]:
    """One row per question: the rank of the first gold provision, or None."""
    per_query = []
    for q in questions:
        gold = set(q["gold_provision_ids"])
        hits = retriever.search(q["question_bn"], k=depth)
        ranked = [provision_of[cid] for cid, _ in hits]
        per_query.append({
            "qid": q["qid"],
            "lang_tag": q.get("lang_tag"),
            "annotation_mode": q.get("annotation_mode"),
            "block": q.get("block"),
            "rank": first_hit(ranked, gold),
            "top_score": float(hits[0][1]) if hits else None,
            "gold_score_max": None,
            "retrieved": [cid for cid, _ in hits[:20]],
        })
    return per_query


def recall_at(per_query: list[dict], k: int) -> float:
    hits = sum(1 for r in per_query if r["rank"] is not None and r["rank"] <= k)
    return hits / len(per_query) if per_query else 0.0


def write_run(run_id: str, retriever: BM25Retriever, per_query: list[dict],
              probes: list[dict], tuning: dict | None, out_runs: pathlib.Path,
              corpus_path: pathlib.Path = CORPUS, probes_path: pathlib.Path = PROBES) -> pathlib.Path:
    def ranks_for(group: str):
        return [r["rank"] for r in per_query if group == "all" or r["lang_tag"] == group]

    metrics = {g: summarize(ranks_for(g)) for g in GROUPS}
    run = {
        "run_id": run_id,
        "retriever": "bm25",
        "checkpoint": None,
        "fine_tuned": False,
        "k1": retriever.k1,
        "b": retriever.b,
        "use_title": retriever.use_title,
        "corpus": str(corpus_path),
        "probes": str(probes_path),
        "n": len(per_query),
        "depth": DEPTH,
        "level": "provision",
        "cutoffs": list(CUTOFFS),
        "by_lang_count": dict(Counter(p.get("lang_tag") for p in probes)),
        "metrics": metrics,
        "tuning": tuning,
        "normalization": "aggressive (content_tokens) — BM25 is a lexical model and "
                         "gets the aggressive level per CLAUDE.md §4.5",
        "per_query": per_query,
    }
    out_runs.mkdir(parents=True, exist_ok=True)
    dest = out_runs / f"{run_id}.json"
    dest.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    table = OUT_TABLES / f"{run_id}_recall.csv"
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "n"] + [f"R@{k}" for k in CUTOFFS] + ["MRR@10"])
        for g in GROUPS:
            m = metrics[g]
            writer.writerow([g, m["n"]] + [m[f"R@{k}"] for k in CUTOFFS] + [m["MRR@10"]])

    print(f"\nwrote {dest}")
    print(f"wrote {table}")
    for g in GROUPS:
        m = metrics[g]
        print(f"  {g:8s} n={m['n']:4d} " + "  ".join(f"R@{k}={m[f'R@{k}']:.3f}" for k in CUTOFFS))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--probes", type=pathlib.Path, default=PROBES)
    ap.add_argument("--dev", type=pathlib.Path, default=DEV)
    ap.add_argument("--dev-sample", type=int, default=150,
                    help="dev questions per configuration; the sweep re-indexes 39k chunks "
                         "for every (k1, b), so this is the cost knob")
    ap.add_argument("--tune-at", type=int, default=10, help="cutoff the sweep optimises")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--k1", type=float, default=None,
                    help="skip the dev sweep and tune-score directly at this k1 "
                         "(requires --b too) -- for reusing an already-known-good "
                         "value on a rescore, since tuning is dev-based and does "
                         "not depend on the probe/gold set")
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--tag", default="",
                    help="suffix appended to run_id/table filenames, e.g. '_v2', "
                         "so a rescore never overwrites the frozen v1 runs")
    ap.add_argument("--out", type=pathlib.Path, default=OUT_RUNS)
    args = ap.parse_args()

    chunks = list(schema.read_jsonl(args.corpus))
    provision_of = {c.chunk_id: c.provision_id for c in chunks}
    probes = read_jsonl(args.probes)
    print(f"corpus {len(chunks)} chunks | probes {len(probes)} questions")

    tuning = None
    best = (1.5, 0.75)

    if args.k1 is not None and args.b is not None:
        best = (args.k1, args.b)
        tuning = {
            "tuned_on": "reused from a prior sweep (--k1/--b passed explicitly); "
                        "tuning is dev-based and does not depend on the probe/gold set",
            "selected": {"k1": best[0], "b": best[1]},
        }
        print(f"skipping the sweep, scoring directly at k1={best[0]}, b={best[1]}")
    elif not args.no_tune:
        dev_rows = read_jsonl(args.dev)
        # One entry per dev question, with every anchored provision as gold.
        by_question: dict[str, dict] = {}
        for row in dev_rows:
            entry = by_question.setdefault(row["question_bn"], {
                "qid": row["qid"], "question_bn": row["question_bn"],
                "lang_tag": row.get("lang_tag"), "gold_provision_ids": set(),
            })
            entry["gold_provision_ids"].update(row["gold_provision_ids"])
        dev = [{**e, "gold_provision_ids": sorted(e["gold_provision_ids"])}
               for e in by_question.values()]
        dev.sort(key=lambda r: r["qid"])
        dev = dev[: args.dev_sample]
        print(f"tuning on {len(dev)} dev questions from {args.dev.name} "
              f"(synthetic; shares no question with the probe set)")

        results = []
        for k1, b in itertools.product(K1_GRID, B_GRID):
            t0 = time.time()
            retriever = BM25Retriever(k1=k1, b=b)
            retriever.index(chunks)
            per_query = score_questions(retriever, dev, provision_of, depth=args.tune_at)
            score = recall_at(per_query, args.tune_at)
            results.append({"k1": k1, "b": b, f"dev_R@{args.tune_at}": round(score, 4)})
            print(f"  k1={k1:<4} b={b:<5} dev R@{args.tune_at}={score:.4f}  "
                  f"({time.time()-t0:.0f}s)")

        results.sort(key=lambda r: -r[f"dev_R@{args.tune_at}"])
        best = (results[0]["k1"], results[0]["b"])
        tuning = {
            "tuned_on": str(args.dev),
            "dev_questions": len(dev),
            "objective": f"R@{args.tune_at}",
            "grid": {"k1": list(K1_GRID), "b": list(B_GRID)},
            "results": results,
            "selected": {"k1": best[0], "b": best[1]},
            "note": "Tuned on the synthetic dev split, never on the probe questions.",
        }
        print(f"\nbest on dev: k1={best[0]}, b={best[1]} "
              f"(R@{args.tune_at}={results[0][f'dev_R@{args.tune_at}']:.4f})")

        with (OUT_TABLES / "bm25_tuning.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["k1", "b", f"dev_R@{args.tune_at}"])
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {OUT_TABLES / 'bm25_tuning.csv'}")

    print("\nscoring the DEFAULT configuration on the probe set (k1=1.5, b=0.75)")
    default = BM25Retriever(k1=1.5, b=0.75)
    default.index(chunks)
    write_run(f"bm25_default{args.tag}", default, score_questions(default, probes, provision_of),
              probes, None, args.out, corpus_path=args.corpus, probes_path=args.probes)

    if tuning and best != (1.5, 0.75):
        print(f"\nscoring the TUNED configuration on the probe set (k1={best[0]}, b={best[1]})")
        tuned = BM25Retriever(k1=best[0], b=best[1])
        tuned.index(chunks)
        write_run(f"bm25_tuned{args.tag}", tuned, score_questions(tuned, probes, provision_of),
                  probes, tuning, args.out, corpus_path=args.corpus, probes_path=args.probes)
    elif tuning:
        print("\nthe sweep selected the library defaults; bm25_default IS the tuned run.")
        run_path = args.out / f"bm25_default{args.tag}.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["tuning"] = tuning
        run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  recorded the sweep in {run_path}")

    print("\nCompare against any other rung with the paired bootstrap:")
    print("  python scripts/18_compare_runs.py --a results/runs/bge_m3_zeroshot.json \\")
    print("      --b results/runs/bm25_tuned.json")


if __name__ == "__main__":
    main()
