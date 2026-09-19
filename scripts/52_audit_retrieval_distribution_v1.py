"""Audit whether every retrieval supervision source resembles human queries.

    python scripts/52_audit_retrieval_distribution_v1.py

This is a release gate, not a score-producing experiment.  It records source
distributions and refuses to present a synthetic layer as final fine-tuning data
when its lexical or length profile diverges from natural human questions.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.synth import content_overlap  # noqa: E402


CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
COVERAGE = pathlib.Path("data/processed/pretrain_retrieval_coverage_v1.jsonl")
OUT = pathlib.Path("results/runs/retrieval_distribution_audit_v1.json")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile(values: list[float], fraction: float) -> float:
    return values[round((len(values) - 1) * fraction)]


def summarize(
    rows: list[dict[str, Any]],
    overlap: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    lengths = sorted(len(row["question"].split()) for row in rows)
    overlaps = sorted(overlap(row) for row in rows)
    return {
        "rows": len(rows),
        "word_count": {
            "minimum": lengths[0], "p05": quantile(lengths, .05),
            "median": quantile(lengths, .5), "p90": quantile(lengths, .9),
            "p95": quantile(lengths, .95), "maximum": lengths[-1],
        },
        "content_overlap": {"median": round(quantile(overlaps, .5), 4), "p90": round(quantile(overlaps, .9), 4)},
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"immutable audit already exists: {OUT}")
    corpus = {row["chunk_id"]: row for row in read_jsonl(CORPUS)}
    train = read_jsonl(TRAIN)
    human = [row for row in train if row.get("source") == "human_adjudicated_v2"]
    approved = [row for row in train if row.get("label_source") == "human_approved_title_pair"]
    coverage = read_jsonl(COVERAGE)
    assert human and approved and coverage

    def positive_overlap(row: dict[str, Any]) -> float:
        document = corpus[row["positive_chunk_ids"][0]]
        text = f"{document.get('provision_title_bn') or ''} {document.get('text_raw') or document.get('text_bn') or ''}"
        return content_overlap(row["question"], text)

    report = {
        "purpose": "compare every training source against the natural human query distribution",
        "sources": {
            "natural_human": summarize(human, positive_overlap),
            "approved_title_pairs": summarize(approved, positive_overlap),
            "synthetic_coverage_pretrain": summarize(coverage, lambda row: row["content_overlap"]),
        },
        "coverage_title_overlap_gate": {
            "maximum": max(row["title_content_overlap"] for row in coverage),
            "passed": all(row["title_overlap_gate"] == "pass" for row in coverage),
        },
        "decision": {
            "coverage_pretrain_allowed": True,
            "coverage_final_finetune_allowed": False,
            "reason": "The coverage layer is structurally broad but must not become final supervision until a diverse LLM-generated layer passes human-profile gates.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
