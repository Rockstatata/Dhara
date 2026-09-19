"""Merge validated augmentation shards into the final supervised train v5.

    python scripts/58_promote_diverse_augmentation_v1.py \
      --validated-dir data/processed/augmentation_validated_v1 \
      --approve-project-synthetic

Every shard must already have passed script 57. This command applies the
aggregate human-profile gate before producing an immutable training artifact.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

BASE_TRAIN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
OUT = pathlib.Path("data/processed/train_retrieval_v5.jsonl")
AUDIT = pathlib.Path("results/runs/retrieval_diverse_augmentation_promotion_v1.json")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def quantile(values: list[float], fraction: float) -> float:
    return values[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-dir", type=pathlib.Path, required=True)
    parser.add_argument("--base-train", type=pathlib.Path, default=BASE_TRAIN)
    parser.add_argument("--output", type=pathlib.Path, default=OUT)
    parser.add_argument("--audit", type=pathlib.Path, default=AUDIT)
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
    if len(augmented) < 1000:
        raise SystemExit("fewer than 1,000 validated augmentations; do not promote a thin generated layer")
    overlaps = sorted(float(row["content_overlap"]) for row in augmented)
    assert quantile(overlaps, .5) <= .10, "aggregate median overlap gate failed"
    assert quantile(overlaps, .9) <= .20, "aggregate p90 overlap gate failed"
    assert all(float(row["title_content_overlap"]) == 0.0 for row in augmented)
    assert len({row["qid"] for row in augmented}) == len(augmented)
    base = read_jsonl(args.base_train)
    normalized_base = {" ".join(row["question"].casefold().split()) for row in base}
    assert not (normalized_base & {" ".join(row["question"].casefold().split()) for row in augmented})
    merged = [*base, *augmented]
    audit = {
        "base_human_and_approved_rows": len(base),
        "promoted_llm_augmentations": len(augmented),
        "training_rows": len(merged),
        "augmentation_overlap": {"median": quantile(overlaps, .5), "p90": quantile(overlaps, .9)},
        "approval_record": "project approval requested with --approve-project-synthetic",
        "inputs": [str(path) for path in files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, merged)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
