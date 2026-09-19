"""Pool dev_retrieval_v4 + test_retrieval_v4 into one eval set.

    python scripts/72_pool_eval_v4.py

Splitting the 71 provision-disjoint-from-train human rows into two 35-row
halves amplifies noise without adding information: each half is dominated by
a handful of correlated same-Act clusters (dev: criminal_procedure_75 x5,
other_751 x5; test: criminal_procedure_11 x5, money_recovery_46 x4), so a
single cluster landing on one side or the other can swing that side's R@10
by 10+ points. The apparent dev-vs-test "collapse" this session flagged is
that artifact, not a labeling defect (chunk_ids, repeal status, and text
were all audited clean against corpus_v1.jsonl).

Pooling to 71 rows halves that instability without discarding any row.
This does NOT restore a held-out test: use it for iterative comparison with
scripts/18_compare_runs.py's paired bootstrap CI, and treat any single
model's number as an estimate with a wide interval, not a fact.
"""

from __future__ import annotations

import json
import pathlib

DEV = pathlib.Path("data/processed/dev_retrieval_v4.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v4.jsonl")
OUT = pathlib.Path("data/processed/eval_retrieval_v4.jsonl")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    dev = read_jsonl(DEV)
    test = read_jsonl(TEST)
    pooled = dev + test
    assert len({r["qid"] for r in pooled}) == len(pooled), "duplicate qid across dev/test"
    with OUT.open("w", encoding="utf-8") as handle:
        for row in pooled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(pooled)} rows ({len(dev)} dev + {len(test)} test)")


if __name__ == "__main__":
    main()
