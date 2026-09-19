"""Merge validated v2 augmentation shards into a supervised train v5.

    python scripts/63_promote_diverse_augmentation_v2.py \
      --validated-dir data/processed/augmentation_validated_v3 \
      --approve-project-synthetic

Every shard must already have passed script 62.  This adds the two aggregate
gates script 58 did not have, both of which the v1 attempt would have failed:

*   **Length shape.**  The v1 shard produced a median of ~25 words whatever
    length was requested, against a human median of 68.  Overlap gates cannot
    see that, so a corpus of uniformly short questions could have been promoted.
    The promoted set's median and p75 must now sit inside a band around the
    human distribution, and its interquartile spread must be at least half the
    human spread - a set that is all one length fails even if every row is
    individually in range.
*   **Human audit sample.**  Constrained decoding (script 61) can make a
    question non-copying by making it vague, which no lexical gate detects.  A
    50-row sample is written out for review; promotion is a project decision and
    this file is the evidence for it.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
from typing import Any

BASE_TRAIN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
OUT = pathlib.Path("data/processed/train_retrieval_v5.jsonl")
AUDIT = pathlib.Path("results/runs/retrieval_diverse_augmentation_promotion_v2.json")
SAMPLE = pathlib.Path("data/annotation/augmentation_v2_audit_sample.csv")

MIN_ROWS = 1000
AUDIT_SAMPLE_ROWS = 50
SEED = 20260917


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def quantile(values: list[float], fraction: float) -> float:
    return sorted(values)[round((len(values) - 1) * fraction)]


def deciles(values: list[int]) -> list[int]:
    ordered = sorted(values)
    return [ordered[round((len(ordered) - 1) * i / 10)] for i in range(11)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-dir", type=pathlib.Path, required=True)
    parser.add_argument("--base-train", type=pathlib.Path, default=BASE_TRAIN)
    parser.add_argument("--output", type=pathlib.Path, default=OUT)
    parser.add_argument("--audit", type=pathlib.Path, default=AUDIT)
    parser.add_argument("--sample", type=pathlib.Path, default=SAMPLE)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--approve-project-synthetic", action="store_true")
    args = parser.parse_args()
    if not args.approve_project_synthetic:
        parser.error("explicit project approval is required to promote generated synthetic rows")
    if args.output.exists() or args.audit.exists():
        raise SystemExit("v5 promotion artifacts already exist; create a new version rather than overwrite")

    files = sorted(args.validated_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit("no validated shards found")
    augmented = [row for path in files for row in read_jsonl(path)]
    if len(augmented) < args.min_rows:
        raise SystemExit(
            f"only {len(augmented)} validated augmentations; do not promote a thin generated layer"
        )

    base = read_jsonl(args.base_train)
    human = [row for row in base if row.get("source") == "human_adjudicated_v2"]
    human_lengths = [len(row["question"].split()) for row in human]
    lengths = [int(row["question_word_count"]) for row in augmented]

    overlaps = sorted(float(row["content_overlap"]) for row in augmented)
    assert quantile(overlaps, .5) <= .10, "aggregate median overlap gate failed"
    assert quantile(overlaps, .9) <= .20, "aggregate p90 overlap gate failed"
    assert all(float(row["title_content_overlap"]) == 0.0 for row in augmented)
    assert len({row["qid"] for row in augmented}) == len(augmented)

    human_median, human_p75 = quantile(human_lengths, .5), quantile(human_lengths, .75)
    human_spread = quantile(human_lengths, .75) - quantile(human_lengths, .25)
    median, p75 = quantile(lengths, .5), quantile(lengths, .75)
    spread = quantile(lengths, .75) - quantile(lengths, .25)
    assert 0.5 * human_median <= median <= 1.5 * human_median, (
        f"length median {median} is outside half-to-1.5x the human median {human_median}"
    )
    assert p75 >= 0.6 * human_p75, f"length p75 {p75} collapsed against the human p75 {human_p75}"
    assert spread >= 0.5 * human_spread, (
        f"interquartile spread {spread} is under half the human spread {human_spread}; "
        "the generator is producing one length, not a distribution"
    )

    normalized_base = {" ".join(row["question"].casefold().split()) for row in base}
    assert not (normalized_base & {" ".join(row["question"].casefold().split()) for row in augmented})

    rng = random.Random(SEED)
    sample = rng.sample(augmented, min(AUDIT_SAMPLE_ROWS, len(augmented)))
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    with args.sample.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "qid", "provision_id", "domain", "question_word_count", "content_overlap",
            "question", "reads_like_a_citizen_wrote_it", "answerable_from_this_provision", "notes",
        ])
        writer.writeheader()
        for row in sample:
            writer.writerow({
                "qid": row["qid"], "provision_id": row["provision_id"], "domain": row["domain"],
                "question_word_count": row["question_word_count"], "content_overlap": row["content_overlap"],
                "question": row["question"],
                "reads_like_a_citizen_wrote_it": "", "answerable_from_this_provision": "", "notes": "",
            })

    merged = [*base, *augmented]
    audit = {
        "base_human_and_approved_rows": len(base),
        "promoted_llm_augmentations": len(augmented),
        "training_rows": len(merged),
        "augmentation_overlap": {"median": quantile(overlaps, .5), "p90": quantile(overlaps, .9)},
        "augmentation_word_deciles": deciles(lengths),
        "human_word_deciles": deciles(human_lengths),
        "length_gate": {"median": median, "human_median": human_median,
                        "p75": p75, "human_p75": human_p75,
                        "iqr": spread, "human_iqr": human_spread},
        "constrained_decoding_rows": sum(1 for row in augmented if row.get("constrained_decoding")),
        "human_audit_sample": str(args.sample),
        "human_audit_status": "PENDING - review the sample before citing any result trained on this file",
        "approval_record": "project approval requested with --approve-project-synthetic",
        "inputs": [str(path) for path in files],
    }
    write_jsonl(args.output, merged)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
