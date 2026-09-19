"""Create a traceable high-coverage *candidate* set for retrieval training.

    python scripts/47_curate_retrieval_training.py

The script deliberately does not call generated examples ``reviewed``.  It
extracts the comparatively safe ``tier_b`` provision-title paraphrases, removes
repealed chunks and every provision used by the frozen v2 gold set, and writes:

* a silver pretraining set, suitable for a clearly-labelled first training stage;
* a 3,000-row, provision-balanced CSV review queue; and
* an audit report documenting exactly what was included and excluded.

Rows become eligible for the headline supervised training set only after a
reviewer marks them approved and a separate importer produces a new immutable
artifact.  Topic-signature labels are never used here: they were empirically
found to select the wrong section often enough to poison contrastive training.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
from typing import Any


SYNTH = pathlib.Path("data/processed/synth_questions_v1.jsonl")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
GOLD = pathlib.Path("data/processed/gold_verified_v2.jsonl")
SILVER_OUT = pathlib.Path("data/processed/retrieval_silver_title_candidates_v1.jsonl")
QUEUE_OUT = pathlib.Path("data/annotation/retrieval_pair_review_queue_v1.csv")
AUDIT_OUT = pathlib.Path("results/runs/retrieval_curation_v1.json")

REPEALED = re.compile(
    r"^\s*\[?\s*(omitted|repealed|রহিত করা হইয়াছে|বিলুপ্ত)", re.IGNORECASE
)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalise_question(text: str) -> str:
    return " ".join(text.casefold().split())


def is_repealed(chunk: dict[str, Any]) -> bool:
    return bool(
        REPEALED.match(chunk.get("provision_title_bn", ""))
        or REPEALED.match(chunk.get("text_bn", ""))
    )


def candidate_row(source: dict[str, Any], chunk: dict[str, Any], round_number: int) -> dict[str, Any]:
    """Map legacy synthetic fields to the Colab retrieval data contract."""
    return {
        "qid": f"silver_{source['qid']}",
        "question": source["question_bn"],
        "positive_chunk_ids": source["gold_chunk_ids"],
        "hard_negative_chunk_ids": [],
        "provision_id": chunk["provision_id"],
        "act_id": chunk["act_id"],
        "domain": chunk["domain"],
        "lang_tag": source.get("lang_tag", "unknown"),
        "source_qid": source["qid"],
        "label_source": "provision_title",
        "annotation_mode": "generated",
        "curation_status": "silver_auto_curated",
        "review_status": "pending",
        "eligible_for_headline_training": False,
        "selection_round": round_number,
        "content_overlap": source.get("content_overlap"),
        "jaccard": source.get("jaccard"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-review", type=int, default=3000)
    parser.add_argument("--force", action="store_true", help="replace candidate outputs only")
    args = parser.parse_args()
    if args.max_review < 1:
        parser.error("--max-review must be positive")

    outputs = (SILVER_OUT, QUEUE_OUT, AUDIT_OUT)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        parser.error("outputs already exist; candidate artifacts are versioned. "
                     "Use --force only to regenerate this non-final review queue: "
                     + ", ".join(str(path) for path in existing))

    chunks = {row["chunk_id"]: row for row in read_jsonl(CORPUS)}
    frozen_gold_provisions = {
        provision
        for row in read_jsonl(GOLD)
        for provision in row.get("relevant_provision_ids", [])
    }
    synth = read_jsonl(SYNTH)
    counters: collections.Counter[str] = collections.Counter()
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for row in synth:
        counters["synthetic_rows"] += 1
        if not (row.get("quality_tier") == "tier_b" and row.get("label_source") == "provision_title"):
            counters["excluded_non_tier_b_title"] += 1
            continue
        ids = row.get("gold_chunk_ids") or []
        if len(ids) != 1 or ids[0] not in chunks:
            counters["excluded_missing_or_ambiguous_chunk"] += 1
            continue
        chunk = chunks[ids[0]]
        if is_repealed(chunk):
            counters["excluded_repealed"] += 1
            continue
        if chunk["provision_id"] in frozen_gold_provisions:
            counters["excluded_frozen_gold_provision"] += 1
            continue
        key = (normalise_question(row["question_bn"]), chunk["provision_id"])
        if key in seen_pairs:
            counters["excluded_duplicate_question_provision"] += 1
            continue
        seen_pairs.add(key)
        candidates.append((row, chunk))

    # Lowest lexical overlap is preferred.  Round-robin selection gives every
    # provision one candidate before any provision receives a second row.
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for source, chunk in candidates:
        grouped[chunk["provision_id"]].append((source, chunk))
    for rows in grouped.values():
        rows.sort(key=lambda pair: (
            pair[0].get("content_overlap", 1.0),
            pair[0].get("jaccard", 1.0),
            pair[0]["qid"],
        ))

    selected: list[dict[str, Any]] = []
    max_depth = max((len(rows) for rows in grouped.values()), default=0)
    for round_number in range(1, max_depth + 1):
        for provision_id in sorted(grouped):
            rows = grouped[provision_id]
            if len(rows) < round_number or len(selected) >= args.max_review:
                continue
            selected.append(candidate_row(*rows[round_number - 1], round_number))
        if len(selected) >= args.max_review:
            break

    silver = [candidate_row(source, chunk, 0) for source, chunk in candidates]
    assert len({row["qid"] for row in silver}) == len(silver)
    assert not (
        {row["provision_id"] for row in silver} & frozen_gold_provisions
    ), "frozen gold provisions leaked into silver data"

    SILVER_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SILVER_OUT.open("w", encoding="utf-8") as fh:
        for row in silver:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    queue_fields = [
        "qid", "question", "positive_chunk_id", "provision_id", "act_id", "domain",
        "lang_tag", "provision_title_bn", "provision_no_ascii", "provision_text_preview",
        "label_source", "curation_status", "review_status", "reviewer", "review_notes",
        "selection_round", "content_overlap", "jaccard",
    ]
    QUEUE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=queue_fields)
        writer.writeheader()
        for row in selected:
            chunk = chunks[row["positive_chunk_ids"][0]]
            writer.writerow({
                "qid": row["qid"],
                "question": row["question"],
                "positive_chunk_id": row["positive_chunk_ids"][0],
                "provision_id": row["provision_id"],
                "act_id": row["act_id"],
                "domain": row["domain"],
                "lang_tag": row["lang_tag"],
                "provision_title_bn": chunk.get("provision_title_bn", ""),
                "provision_no_ascii": chunk.get("provision_no_ascii", ""),
                "provision_text_preview": " ".join(chunk.get("text_raw", "").split())[:600],
                "label_source": row["label_source"],
                "curation_status": row["curation_status"],
                "review_status": "pending",
                "reviewer": "",
                "review_notes": "",
                "selection_round": row["selection_round"],
                "content_overlap": row["content_overlap"],
                "jaccard": row["jaccard"],
            })

    audit = {
        "artifact_version": "v1",
        "purpose": "silver pretraining candidates and human-review queue; not reviewed training data",
        "inputs": {"synthetic": str(SYNTH), "corpus": str(CORPUS), "frozen_gold": str(GOLD)},
        "selection": {
            "accepted_source": "tier_b + provision_title only",
            "excluded_source": ["topic_signature", "tier_c", "anchor"],
            "frozen_gold_provisions_excluded": True,
            "repealed_excluded": True,
            "selection_policy": "round-robin by provision, then lowest lexical overlap",
        },
        "counts": {
            **dict(counters),
            "silver_pairs": len(silver),
            "silver_provisions": len({row["provision_id"] for row in silver}),
            "silver_acts": len({row["act_id"] for row in silver}),
            "review_queue_pairs": len(selected),
            "review_queue_provisions": len({row["provision_id"] for row in selected}),
            "review_queue_acts": len({row["act_id"] for row in selected}),
            "review_queue_languages": dict(collections.Counter(row["lang_tag"] for row in selected)),
            "review_queue_domains": dict(collections.Counter(row["domain"] for row in selected)),
        },
        "review_gate": "Every queue row is pending. Only approved rows may be promoted to a reviewed training artifact.",
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["counts"], ensure_ascii=False, indent=2))
    print(f"wrote {SILVER_OUT}")
    print(f"wrote {QUEUE_OUT}")
    print(f"wrote {AUDIT_OUT}")


if __name__ == "__main__":
    main()
