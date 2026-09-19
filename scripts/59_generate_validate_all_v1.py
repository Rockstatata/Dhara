"""Run every augmentation shard unattended on the SSH GPU.

This is intentionally a thin orchestrator: generation remains resumable and
the validator is run immediately for each shard. Existing files are skipped,
so an interrupted run can simply be started again.

    python scripts/59_generate_validate_all_v1.py \
      --ledger data/processed/diverse_augmentation_prompt_ledger_v1.jsonl \
      --corpus data/processed/corpus_v1.jsonl \
      --train data/processed/train_retrieval_v4.jsonl \
      --dev data/processed/dev_retrieval_v3.jsonl \
      --test data/processed/test_retrieval_v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


def count_jsonl(path: pathlib.Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--train", type=pathlib.Path, required=True)
    parser.add_argument("--dev", type=pathlib.Path, required=True)
    parser.add_argument("--test", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, default=pathlib.Path("data/processed"))
    parser.add_argument("--validated-dir", type=pathlib.Path, default=pathlib.Path("data/processed/augmentation_validated_v2"))
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.shard_size <= 0 or args.start < 0:
        parser.error("--shard-size must be positive and --start non-negative")

    with args.ledger.open(encoding="utf-8") as handle:
        total = sum(bool(line.strip()) for line in handle)
    end = total if args.limit is None else min(total, args.start + args.limit)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.validated_dir.mkdir(parents=True, exist_ok=True)
    generator = pathlib.Path(__file__).with_name("56_generate_diverse_queries_v1.py")
    validator = pathlib.Path(__file__).with_name("57_validate_diverse_queries_v1.py")
    summary: list[dict[str, Any]] = []

    for offset in range(args.start, end, args.shard_size):
        shard = offset // args.shard_size
        raw = args.raw_dir / f"augmentation_raw_v2_part{shard:03d}.jsonl"
        validated = args.validated_dir / f"part{shard:03d}.jsonl"
        if not raw.exists():
            subprocess.run([
                sys.executable, str(generator), "--input", str(args.ledger),
                "--output", str(raw), "--start", str(offset),
                "--limit", str(min(args.shard_size, end - offset)),
                "--model", args.model, "--batch-size", str(args.batch_size),
            ], check=True)
        if not validated.exists():
            subprocess.run([
                sys.executable, str(validator), "--raw", str(raw),
                "--output", str(validated), "--corpus", str(args.corpus),
                "--train", str(args.train), "--dev", str(args.dev),
                "--test", str(args.test),
            ], check=True)
        record = {"shard": shard, "start": offset, "raw": str(raw),
                  "validated": str(validated), "raw_rows": count_jsonl(raw),
                  "validated_rows": count_jsonl(validated)}
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    print(json.dumps({"total_ledger_rows": total, "processed_shards": len(summary),
                      "raw_rows": sum(r["raw_rows"] for r in summary),
                      "validated_rows": sum(r["validated_rows"] for r in summary)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
