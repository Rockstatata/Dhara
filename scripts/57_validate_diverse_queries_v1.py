"""Validate an SSH-generated augmentation shard and produce trainable rows.

    python scripts/57_validate_diverse_queries_v1.py \
      --raw data/processed/augmentation_raw_v1_part000.jsonl \
      --output data/processed/augmentation_validated_v1_part000.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dhara.normalize import _PUNCT_TOKENS, aggressive  # noqa: E402
from dhara.synth import NO_GAP_LOANWORDS, STOPWORDS, content_overlap  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
DEV = pathlib.Path("data/processed/dev_retrieval_v3.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v3.jsonl")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def content_words(text: str) -> set[str]:
    return set(aggressive(text).split()) - STOPWORDS - _PUNCT_TOKENS - NO_GAP_LOANWORDS


def bangla_majority(text: str) -> bool:
    bangla = sum("\u0980" <= char <= "\u09ff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    return bangla >= latin and bangla > 0


def recover_parsed(row: dict[str, Any]) -> dict[str, Any]:
    parsed = row.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    # Older generator shards may contain a complete question followed by a
    # truncated facts list.  Recover that question so it can still pass the
    # independent lexical/leakage gates below.
    text = str(row.get("raw_output") or "")
    marker = re.search(r'"question"\s*:\s*"', text)
    if not marker:
        return {}
    start = marker.end() - 1
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if char == '"' and not escaped:
            try:
                question = json.loads(text[start:index + 1])
            except json.JSONDecodeError:
                break
            if isinstance(question, str) and question.strip():
                return {"question": question, "facts_preserved": []}
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    return {}


def percentile(values: list[int], fraction: float) -> int:
    return values[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    parser.add_argument("--train", type=pathlib.Path, default=TRAIN)
    parser.add_argument("--dev", type=pathlib.Path, default=DEV)
    parser.add_argument("--test", type=pathlib.Path, default=TEST)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite validated shard: {args.output}")
    corpus = {row["chunk_id"]: row for row in read_jsonl(args.corpus)}
    train, dev, test = read_jsonl(args.train), read_jsonl(args.dev), read_jsonl(args.test)
    human = [row for row in train if row.get("source") == "human_adjudicated_v2"]
    low, high = percentile(sorted(len(row["question"].split()) for row in human), .05), percentile(sorted(len(row["question"].split()) for row in human), .95)
    blocked_questions = {normalize(row["question"]) for row in [*train, *dev, *test]}
    held_out = {corpus[chunk_id]["provision_id"] for row in [*dev, *test] for chunk_id in row["positive_chunk_ids"]}
    raw = read_jsonl(args.raw)

    provisional: list[dict[str, Any]] = []
    for row in raw:
        parsed = recover_parsed(row)
        question = " ".join(str(parsed.get("question", "")).split())
        chunk_ids = row.get("positive_chunk_ids", [])
        if not question or not chunk_ids or chunk_ids[0] not in corpus:
            continue
        document = corpus[chunk_ids[0]]
        passage = f"{document.get('provision_title_bn') or ''} {document.get('text_raw') or document.get('text_bn') or ''}"
        title = document.get("provision_title_bn") or document.get("provision_title_en") or ""
        title_terms = content_words(title)
        title_overlap = len(content_words(question) & title_terms) / max(len(title_terms), 1)
        overlap = content_overlap(question, passage)
        provisional.append({
            "qid": f"llm_aug_v1_{row['prompt_id']}", "question": question,
            "positive_chunk_ids": chunk_ids, "hard_negative_chunk_ids": [],
            "provision_id": row["provision_id"], "act_id": document["act_id"], "domain": row["domain"],
            "lang_tag": "bengali", "source": "llm_augmentation_v1",
            "annotation_mode": "private_llm_generation", "label_source": "approved_provision_augmented_query",
            "review_status": "automated_quality_gated", "eligible_for_headline_training": True,
            "target_words": row["target_words"], "question_word_count": len(question.split()),
            "content_overlap": round(overlap, 4), "title_content_overlap": round(title_overlap, 4),
            "facts_preserved": parsed.get("facts_preserved", []),
        })
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for row in provisional:
        owners[normalize(row["question"])].add(row["provision_id"])
    valid = [
        row for row in provisional
        if row["provision_id"] not in held_out
        and normalize(row["question"]) not in blocked_questions
        and len(owners[normalize(row["question"])]) == 1
        and bangla_majority(row["question"])
        and low <= row["question_word_count"] <= high
        and row["content_overlap"] <= .20
        and row["title_content_overlap"] == 0.0
    ]
    assert len({row["qid"] for row in valid}) == len(valid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, valid)
    print(json.dumps({
        "raw_rows": len(raw), "parseable_rows": len(provisional), "validated_rows": len(valid),
        "human_length_band": [low, high],
        "rejection_reasons_are": ["bad JSON", "leakage", "cross-provision duplicate", "non-Bangla", "length", "body overlap > .20", "title overlap"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
