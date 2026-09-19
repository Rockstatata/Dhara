"""Fast release gate for human-like synthetic retrieval supervision.

    python scripts/55_check_diverse_training_quality_v1.py --candidate PATH

The current deterministic coverage file is intentionally expected to fail. The
same command is the promotion gate for the LLM-generated replacement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


HUMAN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile(values: list[float], fraction: float) -> float:
    return values[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--human", type=pathlib.Path, default=HUMAN)
    args = parser.parse_args()
    human = [row for row in read_jsonl(args.human) if row.get("source") == "human_adjudicated_v2"]
    candidate = read_jsonl(args.candidate)
    assert human and candidate
    human_lengths = sorted(len(row["question"].split()) for row in human)
    candidate_lengths = sorted(len(row["question"].split()) for row in candidate)
    overlaps = sorted(float(row["content_overlap"]) for row in candidate)
    title_overlaps = [float(row.get("title_content_overlap", 0.0)) for row in candidate]
    report = {
        "candidate_rows": len(candidate),
        "candidate_overlap": {"median": quantile(overlaps, .5), "p90": quantile(overlaps, .9)},
        "candidate_length": {"p05": quantile(candidate_lengths, .05), "p95": quantile(candidate_lengths, .95)},
        "human_length": {"p05": quantile(human_lengths, .05), "p95": quantile(human_lengths, .95)},
        "maximum_title_overlap": max(title_overlaps),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert report["candidate_overlap"]["median"] <= 0.10, "median content overlap exceeds human-like gate"
    assert report["candidate_overlap"]["p90"] <= 0.20, "p90 content overlap exceeds human-like gate"
    assert report["candidate_length"]["p05"] <= report["human_length"]["p05"], "candidate lacks short human forms"
    assert report["candidate_length"]["p95"] >= report["human_length"]["p95"], "candidate lacks long human narratives"
    assert report["maximum_title_overlap"] == 0.0, "section-title copying detected"
    print("PASS: candidate is eligible for supervised-training promotion")


if __name__ == "__main__":
    main()
