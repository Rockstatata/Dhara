"""Score a reranked candidate list, emitting the same run schema as script 13.

    python scripts/21_eval_rerank.py \
        --rerank data/processed/rerank_bge_v2m3_top100.jsonl \
        --run-id bge_m3_acttitle_rerank

Writes results/runs/<run-id>.json and results/tables/<run-id>_recall.csv.

## Why this is a separate script from 13

`13_eval_dense_index.py` scores an *index* — a matrix of embeddings it can
search itself. A reranker does not produce an index; it produces an ordering
over a candidate list that some earlier stage retrieved. The artifact is a
per-query ranked list of chunk ids, so it needs its own reader.

Everything downstream stays identical on purpose. The output schema here is the
same as 13's, field for field, so `18_compare_runs.py` can take a reranked run
and a dense run as its two arguments without knowing or caring that one came
from a cross-encoder. The paired bootstrap is the only way any of these rungs
get compared, and it must not need a special case per rung.

## The ceiling this stage is working against

A reranker can only reorder what retrieval handed it. On the zero-shot dense
run, 30.3% of questions have their gold provision in ranks 11-100 (recoverable)
but 30.4% are not in the top 200 at all (unrecoverable at any rerank depth).
That second number is the hard ceiling: reranking the top 100 caps R@10 at
0.627 even if the reranker were perfect. `recoverable_ceiling` in the output
records that bound alongside the achieved number, so nobody reads a rerank
result without also seeing what it could not have fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.metrics import CUTOFFS, first_hit, summarize  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
GROUPS = ("all", "english", "bengali", "mixed")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", type=pathlib.Path, required=True,
                    help="JSONL: {qid, ranked_chunk_ids, scores, candidate_source}")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--probes", type=pathlib.Path, default=PROBES)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/runs"))
    args = ap.parse_args()

    probes = {p["qid"]: p for p in read_jsonl(args.probes)}
    reranked = {r["qid"]: r for r in read_jsonl(args.rerank)}

    missing = set(probes) - set(reranked)
    assert not missing, (
        f"{len(missing)} probe questions have no reranked list, e.g. {sorted(missing)[:3]}. "
        "Scoring a partial run against the full probe set would silently understate recall."
    )

    provision_of = {}
    for chunk in schema.read_jsonl(args.corpus):
        provision_of[chunk.chunk_id] = chunk.provision_id

    meta = next(iter(reranked.values()))
    per_query = []
    for qid, probe in probes.items():
        row = reranked[qid]
        gold = set(probe["gold_provision_ids"])
        ranked_ids = row["ranked_chunk_ids"]
        unknown = [c for c in ranked_ids if c not in provision_of]
        assert not unknown, f"{qid}: {len(unknown)} reranked chunk ids not in corpus, e.g. {unknown[:3]}"

        rank = first_hit((provision_of[c] for c in ranked_ids), gold)
        per_query.append({
            "qid": qid,
            "lang_tag": probe["lang_tag"],
            "annotation_mode": probe.get("annotation_mode"),
            "block": probe.get("block"),
            "rank": rank,
            "top_score": float(row["scores"][0]) if row.get("scores") else None,
            "gold_score_max": None,  # cross-encoder scores only the candidates it saw
            "retrieved": ranked_ids[:20],
            "candidate_rank_before": row.get("rank_before"),
        })

    def ranks_for(group: str) -> list:
        return [r["rank"] for r in per_query if group == "all" or r["lang_tag"] == group]

    metrics = {g: summarize(ranks_for(g)) for g in GROUPS}

    # What the candidate list made possible at all, independent of the reranker.
    depth = len(meta.get("ranked_chunk_ids", []))
    reachable = sum(1 for r in per_query if r["rank"] is not None)
    ceiling = round(reachable / len(per_query), 4)

    run = {
        "run_id": args.run_id,
        "retriever": "rerank",
        "checkpoint": meta.get("reranker_checkpoint"),
        "fine_tuned": meta.get("reranker_fine_tuned", False),
        "candidate_source": meta.get("candidate_source"),
        "rerank_depth": depth,
        "corpus": str(args.corpus),
        "probes": str(args.probes),
        "n": len(per_query),
        "depth": depth,
        "level": "provision",
        "cutoffs": list(CUTOFFS),
        "by_lang_count": dict(Counter(p["lang_tag"] for p in probes.values())),
        "metrics": metrics,
        "recoverable_ceiling": ceiling,
        "recoverable_ceiling_note": (
            "Fraction of questions whose gold provision appears anywhere in the candidate "
            "list. The reranker cannot exceed this at any cutoff; the remainder was lost "
            "by the retrieval stage, not by reranking."
        ),
        "per_query": per_query,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{args.run_id}.json"
    dest.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    table = pathlib.Path("results/tables") / f"{args.run_id}_recall.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "n"] + [f"R@{k}" for k in CUTOFFS] + ["MRR@10"])
        for g in GROUPS:
            m = metrics[g]
            writer.writerow([g, m["n"]] + [m[f"R@{k}"] for k in CUTOFFS] + [m["MRR@10"]])

    print(f"wrote {dest}")
    print(f"wrote {table}")
    print(f"  candidate source : {meta.get('candidate_source')} (depth {depth})")
    print(f"  reranker         : {meta.get('reranker_checkpoint')}")
    print(f"  recoverable ceiling (gold present in candidates): {ceiling:.3f}")
    for g in GROUPS:
        m = metrics[g]
        print(f"  {g:8s} n={m['n']:4d} " + "  ".join(f"R@{k}={m[f'R@{k}']:.3f}" for k in CUTOFFS))


if __name__ == "__main__":
    main()
