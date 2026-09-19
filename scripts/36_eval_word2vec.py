"""Score the self-trained Word2Vec retrieval rung (proposal §8.2, Rung 2).

    python scripts/36_eval_word2vec.py

Mean-pooled centre-word vectors, cosine similarity, over
models/word2vec_legal.kv (scripts/10_train_word2vec.py). This rung never
existed before the v2 relabeling, so it is scored once, directly against the
verified gold (data/processed/probe_questions_v2.jsonl) -- there is no v1
number to reproduce or compare against.

Expected and fine to report honestly (proposal §8.2, risk #6 in
docs/Dhara_Proposal.md): weak absolute recall. The corpus is small for
Word2Vec (a few hundred thousand tokens); this is a stated corpus-size
finding, not a bug to chase.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.metrics import CUTOFFS, first_hit, summarize  # noqa: E402
from dhara.retrievers.word2vec import Word2VecRetriever  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
PROBES = pathlib.Path("data/processed/probe_questions_v2.jsonl")
OUT_RUNS = pathlib.Path("results/runs")
OUT_TABLES = pathlib.Path("results/tables")
GROUPS = ("all", "english", "bengali", "mixed")
DEPTH = 100
RUN_ID = "word2vec_v2"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    chunks = list(schema.read_jsonl(CORPUS))
    provision_of = {c.chunk_id: c.provision_id for c in chunks}
    probes = read_jsonl(PROBES)
    print(f"corpus {len(chunks)} chunks | probes {len(probes)} questions")

    retriever = Word2VecRetriever()
    print(f"loaded word2vec vocab={len(retriever.vocab):,} dim={retriever.dim}")
    retriever.index(chunks)

    per_query = []
    for q in probes:
        gold = set(q["gold_provision_ids"])
        hits = retriever.search(q["question_bn"], k=DEPTH)
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

    def ranks_for(group: str):
        return [r["rank"] for r in per_query if group == "all" or r["lang_tag"] == group]

    metrics = {g: summarize(ranks_for(g)) for g in GROUPS}
    run = {
        "run_id": RUN_ID,
        "retriever": "word2vec",
        "checkpoint": "models/word2vec_legal.kv",
        "fine_tuned": False,
        "corpus": str(CORPUS),
        "probes": str(PROBES),
        "n": len(per_query),
        "depth": DEPTH,
        "level": "provision",
        "cutoffs": list(CUTOFFS),
        "by_lang_count": dict(Counter(p.get("lang_tag") for p in probes)),
        "metrics": metrics,
        "normalization": "aggressive() -- matches training tokenization "
                          "(scripts/10_train_word2vec.py), per CLAUDE.md §4.5",
        "per_query": per_query,
    }
    OUT_RUNS.mkdir(parents=True, exist_ok=True)
    dest = OUT_RUNS / f"{RUN_ID}.json"
    dest.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    table = OUT_TABLES / f"{RUN_ID}_recall.csv"
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


if __name__ == "__main__":
    main()
