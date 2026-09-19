"""Balanced-mix retrieval training pool: quality-filtered, size-capped synthetic.

    python scripts/73_build_balanced_pool_v7.py

Same policy as `70_merge_pool_v6.py` (provision-level + normalized-question
exclusion against dev/test v4), but the synthetic side is no longer "dump
everything": v6's pool was 2,685 approved + 942 authored + 408 human = 4,035
rows, human share 10%. The classification experiment this session
(DECISIONS.md 2026-09-19) found that a ~92%-synthetic training mix measurably
hurt a lexical classifier by shifting its feature statistics toward
synthetic phrasing. Retrieval's risk is smaller (it trains on semantic
embeddings, not bag-of-words, and already curriculum-stages rather than
dumping everything into one flat set) but untested -- this pool lets that
be tested directly instead of assumed.

'Quality' selection, same definition used for the classification balanced
mix: LOWEST content_overlap for approved rows (this project's premise is
that real questions have near-zero lexical overlap with their answer, so a
HIGH-overlap approved pair is the easy/shortcut-able one), and
overlap_gate_waived=False for authored rows (passed the lexical-overlap
quality gate cleanly).
"""

from __future__ import annotations

import collections
import json
import pathlib

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
APPROVED = pathlib.Path("data/processed/train_retrieval_v3.jsonl")
AUTHORED = pathlib.Path("data/processed/authored_v1/merged_passing.jsonl")
HUMAN_TRAIN = pathlib.Path("data/processed/human_train_v4.jsonl")
DEV = pathlib.Path("data/processed/dev_retrieval_v4.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v4.jsonl")
OUT = pathlib.Path("data/processed/train_retrieval_v7.jsonl")
REPORT = pathlib.Path("results/runs/merge_train_retrieval_v7.json")

N_APPROVED = 600
N_AUTHORED = 600


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
    approved_all = sorted(read_jsonl(APPROVED), key=lambda r: r.get("content_overlap", 1.0))
    approved = approved_all[:N_APPROVED]
    authored_all = [r for r in read_jsonl(AUTHORED) if not r.get("overlap_gate_waived")]
    authored = authored_all[:N_AUTHORED]
    human_train = read_jsonl(HUMAN_TRAIN)
    dev, test = read_jsonl(DEV), read_jsonl(TEST)

    held_out_provisions = {
        corpus[cid]["provision_id"]
        for row in [*dev, *test] for cid in row["positive_chunk_ids"] if cid in corpus
    }
    blocked_questions = {norm_q(row["question"]) for row in [*dev, *test]}

    pool = [*human_train, *approved, *authored]

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
        "input_human_rows": len(human_train),
        "input_approved_pool": len(approved_all), "approved_selected": len(approved),
        "input_authored_pool": len(authored_all), "authored_selected": len(authored),
        "pool_before_filters": len(pool),
        "kept": len(kept),
        "dropped_leak_vs_dev_test": dropped_leak,
        "dropped_cross_provision_duplicate": dropped_dup_q,
        "human_share_pct": round(100 * len(human_train) / len(kept), 1),
        "distinct_provisions": len({r.get("provision_id") or corpus[r["positive_chunk_ids"][0]]["provision_id"] for r in kept}),
        "by_source": dict(collections.Counter(r.get("source") or "human_approved_title_pair" for r in kept).most_common()),
        "next_step": "mine hard negatives with scripts/74_mine_negatives_v7.py",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
