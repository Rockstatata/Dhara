"""Build the three retrieval files consumed by colab_retrieval_finetune_v2.

    python scripts/48_build_retrieval_colab_splits.py

Training is the explicitly-labelled silver title-paraphrase set produced by
``47_curate_retrieval_training.py``.  Development and test are a deterministic,
approximately balanced, stratified partition of the 480 answerable
human-adjudicated v2 questions.  The split unit is the question: related legal
questions can legitimately cite a shared provision, but no held-out provision
appears anywhere in training.  A provision-disjoint dev/test split is not viable
for this resource because multi-label questions form one 430-question connected
component through shared supporting provisions.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
from typing import Any


SILVER = pathlib.Path("data/processed/retrieval_silver_title_candidates_v1.jsonl")
GOLD = pathlib.Path("data/processed/gold_verified_v2.jsonl")
TRAIN_OUT = pathlib.Path("data/processed/train_retrieval_v2.jsonl")
DEV_OUT = pathlib.Path("data/processed/dev_retrieval_v2.jsonl")
TEST_OUT = pathlib.Path("data/processed/test_retrieval_v2.jsonl")
AUDIT_OUT = pathlib.Path("results/runs/retrieval_colab_splits_v2.json")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": row["qid"],
        "question": row["question_bn"],
        "positive_chunk_ids": row["relevant_chunk_ids"],
        "hard_negative_chunk_ids": [],
        "domain": row["domain"],
        "register": row["register"],
        "source": "human_adjudicated_v2",
        "annotation_mode": row["annotation_mode"],
        "answer_source": row["answer_source"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace these derived, non-final files")
    args = parser.parse_args()
    outputs = (TRAIN_OUT, DEV_OUT, TEST_OUT, AUDIT_OUT)
    exists = [path for path in outputs if path.exists()]
    if exists and not args.force:
        parser.error("derived split files already exist; use --force to regenerate: "
                     + ", ".join(str(path) for path in exists))

    train = read_jsonl(SILVER)
    gold = [row for row in read_jsonl(GOLD) if row.get("answerable") and row.get("relevant_chunk_ids")]
    assert len(gold) == 480, f"expected 480 answerable v2 rows; got {len(gold)}"

    # Stratify by fields later used in error slices, then alternate a stable
    # hash order.  This gives a 240/240 question-disjoint split without relying
    # on Python's process-randomized hash().
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in gold:
        buckets[(row["domain"], row["register"])].append(row)
    dev, test = [], []
    for key in sorted(buckets):
        rows = sorted(
            buckets[key],
            key=lambda row: hashlib.sha1(f"retrieval-v2:{row['qid']}".encode("utf-8")).hexdigest(),
        )
        dev.extend(to_eval_row(row) for row in rows[::2])
        test.extend(to_eval_row(row) for row in rows[1::2])

    train_provisions = {row["provision_id"] for row in train}
    dev_chunks = {chunk for row in dev for chunk in row["positive_chunk_ids"]}
    test_chunks = {chunk for row in test for chunk in row["positive_chunk_ids"]}
    assert not ({row["qid"] for row in dev} & {row["qid"] for row in test})
    # Gold is represented at provision-level in the silver artifact; recover it
    # from ids by dropping the final paragraph suffix where present.
    def provision_id(chunk_id: str) -> str:
        return chunk_id.rsplit("_p", 1)[0] if "_p" in chunk_id else chunk_id
    held_out_provisions = {provision_id(chunk) for chunk in dev_chunks | test_chunks}
    assert not (train_provisions & held_out_provisions), "silver train includes a held-out gold provision"

    for path, rows in ((TRAIN_OUT, train), (DEV_OUT, dev), (TEST_OUT, test)):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, rows)

    audit = {
        "train": {"rows": len(train), "provisions": len(train_provisions), "source": "silver_auto_curated"},
        "dev": {"rows": len(dev), "source": "human_adjudicated_v2"},
        "test": {"rows": len(test), "source": "human_adjudicated_v2"},
        "invariants": {
            "question_qid_disjoint": True,
            "held_out_gold_provision_excluded_from_train": True,
        },
        "interpretation": "Use this for a silver-supervision experiment. It does not convert generated labels into reviewed labels.",
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
