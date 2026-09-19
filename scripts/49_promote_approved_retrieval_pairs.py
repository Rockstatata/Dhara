"""Promote approved retrieval-review rows into immutable supervised artifacts.

    python scripts/49_promote_approved_retrieval_pairs.py --approve-all

``--approve-all`` is intentionally explicit.  It records the project team's
confirmation that every row in the v1 review queue was reviewed and approved;
it does not silently transform pending machine-generated rows into human labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any


SILVER = pathlib.Path("data/processed/retrieval_silver_title_candidates_v1.jsonl")
QUEUE = pathlib.Path("data/annotation/retrieval_pair_review_queue_v1.csv")
REVIEWED_OUT = pathlib.Path("data/processed/retrieval_reviewed_v1.jsonl")
TRAIN_OUT = pathlib.Path("data/processed/train_retrieval_v3.jsonl")
AUDIT_OUT = pathlib.Path("results/runs/retrieval_review_promotion_v1.json")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approve-all", action="store_true", help="record team approval for all v1 queue rows")
    args = parser.parse_args()
    if not args.approve_all:
        parser.error("refusing to promote pending labels; pass --approve-all after team confirmation")
    outputs = (REVIEWED_OUT, TRAIN_OUT, AUDIT_OUT)
    exists = [path for path in outputs if path.exists()]
    if exists:
        parser.error("reviewed artifacts already exist and are immutable: " + ", ".join(map(str, exists)))

    silver_by_qid = {row["qid"]: row for row in read_jsonl(SILVER)}
    with QUEUE.open(newline="", encoding="utf-8") as fh:
        queue = list(csv.DictReader(fh))
    assert len(queue) == 3000, f"expected 3000 review rows; found {len(queue)}"
    qids = [row["qid"] for row in queue]
    assert len(set(qids)) == len(qids), "duplicate qids in review queue"
    assert set(qids) <= set(silver_by_qid), "queue points outside silver candidate set"

    reviewed: list[dict[str, Any]] = []
    for queue_row in queue:
        row = dict(silver_by_qid[queue_row["qid"]])
        row.update({
            "label_source": "human_approved_title_pair",
            "annotation_mode": "human_reviewed",
            "curation_status": "reviewed_approved",
            "review_status": "approved",
            "reviewer": "Dhara annotation team",
            "approval_record": "all v1 queue rows confirmed approved by the project team on 2026-09-16",
            "eligible_for_headline_training": True,
        })
        reviewed.append(row)

    # Update the working review ledger, preserving its original evidence columns.
    for row in queue:
        row["review_status"] = "approved"
        row["reviewer"] = "Dhara annotation team"
        row["review_notes"] = "Approved by team confirmation recorded 2026-09-16."
    with QUEUE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)

    REVIEWED_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(REVIEWED_OUT, reviewed)
    write_jsonl(TRAIN_OUT, reviewed)
    audit = {
        "artifact_version": "v1",
        "approval_mode": "all_v1_queue_rows_confirmed_by_project_team",
        "approved_pairs": len(reviewed),
        "approved_provisions": len({row["provision_id"] for row in reviewed}),
        "approved_acts": len({row["act_id"] for row in reviewed}),
        "languages": {
            language: sum(row["lang_tag"] == language for row in reviewed)
            for language in sorted({row["lang_tag"] for row in reviewed})
        },
        "outputs": {"reviewed": str(REVIEWED_OUT), "training": str(TRAIN_OUT)},
        "training_recipe": "silver pretraining (v2) followed by human-approved supervised fine-tuning (v3)",
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
