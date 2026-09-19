"""Score a cached dense index against the probe set and emit a run JSON.

    python scripts/13_eval_dense_index.py --index models/index_bge_m3_zeroshot_v1 --run-id bge_m3_zeroshot

The embeddings were produced on a T4 by notebooks/colab_bge_m3_eval.ipynb, which
printed aggregate recall and nothing else. This script re-derives every number
from the saved vectors on CPU in seconds, and — the reason it exists — writes
`per_query` alongside them. No number reaches the report except through a
results/runs/*.json emitted here, and a run JSON without per_query cannot be
error-analysed or bootstrapped later without re-running the GPU job.

It never re-embeds. Queries come from the cached `probe_query.npy`, whose row
order is asserted against probe_questions.jsonl, so a stale index fails loudly
instead of quietly scoring the wrong questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.metrics import CUTOFFS, first_hit, summarize  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
DEPTH = 200  # ranks kept per query; also the deepest cutoff we can honestly report


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def chunk_to_provision(corpus_path: pathlib.Path) -> dict[str, str]:
    """chunk_id -> provision_id. Read streaming; the corpus is ~138 MB."""
    mapping = {}
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["chunk_id"]] = row["provision_id"]
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=pathlib.Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--probes", type=pathlib.Path, default=PROBES)
    ap.add_argument("--query-cache", default="probe_query.npy",
                    help="filename within --index to read cached query vectors from, "
                         "e.g. probe_query_v2.npy for a rescore that must not touch "
                         "the v1 cache")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/runs"))
    args = ap.parse_args()

    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    corpus_emb = np.load(args.index / "embeddings.npy").astype(np.float32)
    chunk_ids = json.loads((args.index / "chunk_ids.json").read_text(encoding="utf-8"))
    query_emb = np.load(args.index / args.query_cache).astype(np.float32)
    probes = read_jsonl(args.probes)

    assert len(chunk_ids) == corpus_emb.shape[0], "chunk_ids and embeddings disagree"
    assert len(probes) == query_emb.shape[0], (
        f"{len(probes)} probe questions but {query_emb.shape[0]} cached query vectors — "
        "the index is stale; re-run the notebook"
    )

    provision_of = chunk_to_provision(args.corpus)
    missing = [c for c in chunk_ids if c not in provision_of]
    assert not missing, f"{len(missing)} indexed chunks are not in the corpus, e.g. {missing[:3]}"
    provisions = np.array([provision_of[c] for c in chunk_ids])

    per_query = []
    for i, probe in enumerate(probes):
        scores = corpus_emb @ query_emb[i]
        top = np.argpartition(scores, -DEPTH)[-DEPTH:]
        top = top[np.argsort(scores[top])[::-1]]
        gold = set(probe["gold_provision_ids"])
        rank = first_hit((provisions[j] for j in top), gold)
        per_query.append(
            {
                "qid": probe["qid"],
                "lang_tag": probe["lang_tag"],
                "annotation_mode": probe.get("annotation_mode"),
                "block": probe.get("block"),
                "rank": rank,
                "top_score": float(scores[top[0]]),
                "gold_score_max": float(
                    max((scores[j] for j in np.where(np.isin(provisions, list(gold)))[0]), default=float("nan"))
                ),
                "retrieved": [chunk_ids[j] for j in top[:20]],
            }
        )

    def slice_ranks(tag: str | None) -> list:
        return [r["rank"] for r in per_query if tag is None or r["lang_tag"] == tag]

    groups = ["all", "english", "bengali", "mixed"]
    metrics = {g: summarize(slice_ranks(None if g == "all" else g)) for g in groups}

    run = {
        "run_id": args.run_id,
        "retriever": "biencoder",
        "checkpoint": manifest["checkpoint"],
        "fine_tuned": manifest.get("fine_tuned", False),
        "index": str(args.index),
        "manifest": manifest,
        "corpus": str(args.corpus),
        "probes": str(args.probes),
        "n": len(probes),
        "depth": DEPTH,
        "level": "provision",
        "cutoffs": list(CUTOFFS),
        "by_lang_count": dict(Counter(p["lang_tag"] for p in probes)),
        "metrics": metrics,
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
        for g in groups:
            m = metrics[g]
            writer.writerow([g, m["n"]] + [m[f"R@{k}"] for k in CUTOFFS] + [m["MRR@10"]])

    print(f"wrote {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
    print(f"wrote {table}")
    for g in groups:
        m = metrics[g]
        print(f"  {g:8s} n={m['n']:4d} " + "  ".join(f"R@{k}={m[f'R@{k}']:.3f}" for k in CUTOFFS))


if __name__ == "__main__":
    main()
