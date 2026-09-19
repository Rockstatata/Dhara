"""Merge approved + human_train_v4 + authored into train_retrieval_v6.jsonl.

    python scripts/70_merge_pool_v6.py

Same policy as `66_merge_authored_into_pool.py` (provision-level and
normalized-question exclusion against dev/test), but against the v4
dev/test (`dev_retrieval_v4.jsonl` / `test_retrieval_v4.jsonl`, built by
`68_build_human_aware_retrieval_splits_v4.py`), which already enforces this
same provision-level invariant on the human split itself. Because the human
train rows already can't share a provision with the new dev/test, this
merge should drop ~0 rows to leakage -- unlike the v5 merge, which dropped
199 human rows it had never been told to protect at split time.

`hard_negative_chunk_ids` stays `[]` for every row here; mine them with
`scripts/71_mine_negatives_v6.py` before this pool goes near a GPU.
"""

from __future__ import annotations

import collections
import json
import pathlib

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
APPROVED = pathlib.Path("data/processed/train_retrieval_v3.jsonl")
HUMAN_TRAIN = pathlib.Path("data/processed/human_train_v4.jsonl")
AUTHORED = pathlib.Path("data/processed/authored_v1/merged_passing.jsonl")
DEV = pathlib.Path("data/processed/dev_retrieval_v4.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v4.jsonl")
OUT = pathlib.Path("data/processed/train_retrieval_v6.jsonl")
REPORT = pathlib.Path("results/runs/merge_train_retrieval_v6.json")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_q(text: str) -> str:
    return " ".join(text.casefold().split())


def main() -> None:
    corpus = {row["chunk_id"]: row for row in read_jsonl(CORPUS)}
    approved = read_jsonl(APPROVED)
    human_train = read_jsonl(HUMAN_TRAIN)
    authored = read_jsonl(AUTHORED)
    dev, test = read_jsonl(DEV), read_jsonl(TEST)

    held_out_provisions = {
        corpus[cid]["provision_id"]
        for row in [*dev, *test] for cid in row["positive_chunk_ids"] if cid in corpus
    }
    blocked_questions = {norm_q(row["question"]) for row in [*dev, *test]}

    pool = [*approved, *human_train, *authored]

    kept: list[dict] = []
    dropped_leak = dropped_dup_q = 0
    seen_q: dict[str, str] = {}
    for row in pool:
        pid = row.get("provision_id")
        if pid is None:
            cid0 = row["positive_chunk_ids"][0]
            pid = corpus.get(cid0, {}).get("provision_id")
        qn = norm_q(row["question"])
        if pid in held_out_provisions or qn in blocked_questions:
            dropped_leak += 1
            continue
        if qn in seen_q and seen_q[qn] != pid:
            dropped_dup_q += 1
            continue
        seen_q[qn] = pid
        kept.append(row)

    write_jsonl(OUT, kept)

    report = {
        "input_approved_rows": len(approved),
        "input_human_train_rows": len(human_train),
        "input_authored_rows": len(authored),
        "pool_before_filters": len(pool),
        "kept": len(kept),
        "dropped_leak_vs_dev_test": dropped_leak,
        "dropped_cross_provision_duplicate": dropped_dup_q,
        "distinct_questions": len({norm_q(r["question"]) for r in kept}),
        "distinct_provisions": len({r.get("provision_id") or corpus[r["positive_chunk_ids"][0]]["provision_id"] for r in kept}),
        "by_source": dict(collections.Counter(r.get("source") or "human_approved_title_pair" for r in kept).most_common()),
        "next_step": "mine hard negatives with scripts/71_mine_negatives_v6.py",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
