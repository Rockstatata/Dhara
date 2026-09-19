"""Build leakage-controlled retrieval splits that train on natural human queries.

    python scripts/51_build_human_aware_retrieval_splits_v3.py

The 480 answerable, human-adjudicated questions are the only natural-query
resource in the project.  Reserving all of them for evaluation leaves the model
to learn only short, title-shaped synthetic questions.  This script uses half
for training and reserves one quarter each for development and the frozen test.

The split unit is a normalized question, not an individual row.  Consequently
an exact repeated question cannot occur in more than one split.  Provisions may
appear in both the human training and human evaluation sets: this evaluates
generalization to new formulations of a known legal provision.  The evaluation
files must never be used for optimization or hard-negative mining.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
from typing import Any


APPROVED = pathlib.Path("data/processed/train_retrieval_v3.jsonl")
GOLD = pathlib.Path("data/processed/gold_verified_v2.jsonl")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN_OUT = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
DEV_OUT = pathlib.Path("data/processed/dev_retrieval_v3.jsonl")
TEST_OUT = pathlib.Path("data/processed/test_retrieval_v3.jsonl")
AUDIT_OUT = pathlib.Path("results/runs/retrieval_human_aware_splits_v3.json")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.casefold()).strip()


def eval_row(row: dict[str, Any]) -> dict[str, Any]:
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


def train_row(row: dict[str, Any], chunk_to_row: dict[str, dict[str, Any]]) -> dict[str, Any]:
    positives = row["relevant_chunk_ids"]
    first = chunk_to_row[positives[0]]
    return {
        "qid": row["qid"],
        "question": row["question_bn"],
        "positive_chunk_ids": positives,
        "hard_negative_chunk_ids": [],
        "provision_id": first["provision_id"],
        "act_id": first["act_id"],
        "domain": row["domain"],
        "lang_tag": "bengali" if any("\u0980" <= char <= "\u09ff" for char in row["question_bn"]) else "english",
        "source": "human_adjudicated_v2",
        "annotation_mode": row["annotation_mode"],
        "answer_source": row["answer_source"],
        "label_source": "human_adjudicated_provision",
        "review_status": "approved",
        "eligible_for_headline_training": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace these derived split artifacts")
    args = parser.parse_args()
    outputs = (TRAIN_OUT, DEV_OUT, TEST_OUT, AUDIT_OUT)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        parser.error("outputs exist; these are versioned artifacts: " + ", ".join(map(str, existing)))

    approved = read_jsonl(APPROVED)
    gold = [row for row in read_jsonl(GOLD) if row.get("answerable") and row.get("relevant_chunk_ids")]
    corpus = read_jsonl(CORPUS)
    chunk_to_row = {row["chunk_id"]: row for row in corpus}
    assert len(gold) == 480, f"expected 480 answerable human rows, found {len(gold)}"
    assert all(chunk in chunk_to_row for row in gold for chunk in row["relevant_chunk_ids"])

    # Normalize-group before stratifying: repeated wording is indivisible.
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in gold:
        groups[normalize_question(row["question_bn"])].append(row)
    bucketed: dict[tuple[str, str], list[list[dict[str, Any]]]] = collections.defaultdict(list)
    for group in groups.values():
        key = (group[0]["domain"], group[0]["register"])
        bucketed[key].append(group)

    human_train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    # Stable 2:1:1 assignment within each diagnostic stratum.
    for key in sorted(bucketed):
        ordered = sorted(
            bucketed[key],
            key=lambda group: hashlib.sha1(
                f"human-aware-v3:{normalize_question(group[0]['question_bn'])}".encode("utf-8")
            ).hexdigest(),
        )
        for index, group in enumerate(ordered):
            target = human_train if index % 4 in (0, 1) else dev if index % 4 == 2 else test
            target.extend(group)

    human_train_rows = [train_row(row, chunk_to_row) for row in human_train]
    dev_rows = [eval_row(row) for row in dev]
    test_rows = [eval_row(row) for row in test]
    train_rows = [*approved, *human_train_rows]
    split_sets = {
        "train": {normalize_question(row["question"]) for row in train_rows},
        "dev": {normalize_question(row["question"]) for row in dev_rows},
        "test": {normalize_question(row["question"]) for row in test_rows},
    }
    assert not (split_sets["train"] & split_sets["dev"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["dev"] & split_sets["test"])
    assert not ({row["qid"] for row in train_rows} & {row["qid"] for row in dev_rows + test_rows})
    assert len({row["qid"] for row in train_rows}) == len(train_rows)

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(DEV_OUT, dev_rows)
    write_jsonl(TEST_OUT, test_rows)
    audit = {
        "version": "v3",
        "split_policy": "normalized-question-grouped, stratified 50/25/25 human split",
        "approved_title_pairs_in_train": len(approved),
        "human_adjudicated": {"train": len(human_train_rows), "dev": len(dev_rows), "test": len(test_rows)},
        "training_rows": len(train_rows),
        "train_source_counts": dict(
            collections.Counter(row.get("source", "human_approved_title_pair") for row in train_rows)
        ),
        "exact_normalized_question_overlap": {"train_dev": 0, "train_test": 0, "dev_test": 0},
        "provision_overlap_policy": "allowed for human formulation generalization; never use dev/test questions to tune",
        "frozen_test": str(TEST_OUT),
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
