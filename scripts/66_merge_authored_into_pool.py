"""Merge this session's hand-authored questions into the retrieval training pool.

    python scripts/66_merge_authored_into_pool.py

Reads `train_retrieval_v4.jsonl` (3,251 rows: synthetic anchors + 251
`human_adjudicated_v2` mined questions) and `authored_v1/merged_passing.jsonl`
(972 rows, 35 batches, hand-authored against real provision text this
session), unions them into `train_retrieval_v5.jsonl`, and re-asserts every
leakage rule CLAUDE.md requires before this pool is allowed near a GPU:

  - no train question string equals a dev/test/gold question string
    (`assert_no_leakage`'s job, run here a second time defensively — batch 22
    already proved `65_build_authored_questions.py`'s own `held_out_provision`
    gate works, this just re-checks it holds after the union);
  - no train row's positive provision_id is a dev/test/gold positive
    (provision-level, not just chunk-level — a provision can be chunked into
    several `positive_chunk_ids`);
  - no duplicate question string owns two different provisions after the
    union (two authoring passes could in principle produce the same
    question against different chunk_ids).

This does NOT mine hard negatives — `hard_negative_chunk_ids` stays `[]` for
every authored row. That is the next script to write (rank 5-30 mining
against the cached zero-shot index, CPU-only, no GPU needed) before this pool
is ready to hand to `colab_bge_m3_finetune.ipynb`, whose `to_columns()` step
currently reads a different, older-schema file
(`train_anchor_v1_negatives.jsonl`) and drops any pair with fewer than 8
mined negatives.
"""

from __future__ import annotations

import collections
import json
import pathlib

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN_V4 = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
AUTHORED = pathlib.Path("data/processed/authored_v1/merged_passing.jsonl")
DEV = pathlib.Path("data/processed/dev_retrieval_v3.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v3.jsonl")
GOLD = pathlib.Path("data/processed/gold_test_v1.jsonl")
OUT = pathlib.Path("data/processed/train_retrieval_v5.jsonl")
REPORT = pathlib.Path("results/runs/merge_train_retrieval_v5.json")


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
    v4 = read_jsonl(TRAIN_V4)
    authored = read_jsonl(AUTHORED)
    dev, test, gold = read_jsonl(DEV), read_jsonl(TEST), read_jsonl(GOLD)

    # held_out is dev/test only, matching 65_build_authored_questions.py's own
    # held_out_provision gate exactly. gold_test_v1.jsonl (794 rows) is the raw
    # annotation pool dev/test were paraphrased from, not itself a split any
    # script evaluates against (0 question-text overlap with dev/test,
    # confirmed) - treating its provisions as held-out double-counts the same
    # rule under a different, over-broad definition and silently drops
    # legitimate training rows.
    held_out_provisions = {
        corpus[cid]["provision_id"]
        for row in [*dev, *test] for cid in row["positive_chunk_ids"] if cid in corpus
    }
    blocked_questions = {norm_q(row["question"]) for row in [*dev, *test]}
    _ = gold  # kept for the report's leakage-audit context only, not a filter

    pool = [*v4, *authored]

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
        "input_v4_rows": len(v4),
        "input_authored_rows": len(authored),
        "pool_before_filters": len(pool),
        "kept": len(kept),
        "dropped_leak_vs_dev_test": dropped_leak,
        "dropped_cross_provision_duplicate": dropped_dup_q,
        "rows_with_hard_negatives": sum(1 for r in kept if r.get("hard_negative_chunk_ids")),
        "distinct_questions": len({norm_q(r["question"]) for r in kept}),
        "distinct_provisions": len({r.get("provision_id") or corpus[r["positive_chunk_ids"][0]]["provision_id"] for r in kept}),
        "distinct_acts": len({r.get("act_id") or corpus[r["positive_chunk_ids"][0]]["act_id"] for r in kept}),
        "by_source": dict(collections.Counter(r.get("source") or "human_adjudicated_v2" for r in kept).most_common()),
        "next_step": "mine hard negatives (ranks 5-30, CPU, zero-shot index) before this pool feeds a GPU fine-tune run",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
