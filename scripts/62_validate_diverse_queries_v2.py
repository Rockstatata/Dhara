"""Validate a v2 augmentation shard and produce trainable rows.

    python scripts/62_validate_diverse_queries_v2.py \
      --raw data/processed/augmentation_raw_v2_part000.jsonl \
      --output data/processed/augmentation_validated_v3/part000.jsonl

The gates are the ones script 57 already used, and they are kept unchanged on
purpose: measured against the same corpus, 89.2% of the 251 human-adjudicated
training questions, 93.2% of dev v3 and 92.0% of test v3 pass all of them.  The
gates are therefore calibrated to real citizen questions, and a low yield is
evidence about the generator, not a reason to loosen a threshold.

What is new is reporting.  Script 57 printed only a total, so a 17% yield could
not be attributed to a cause without a throwaway script.  This prints the kill
count per gate, the co-occurrence of gates, and the length shape of the accepted
rows against the human distribution - the three things a pilot run has to answer
before a full 9,424-row generation is worth paying for.
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

BODY_OVERLAP_MAX = 0.20
GATES = ("no_question", "unknown_chunk", "held_out_provision", "duplicate_of_split",
         "cross_provision_duplicate", "not_bangla", "length_outside_human_band",
         "body_overlap", "title_overlap")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def content_words(text: str) -> set[str]:
    return set(aggressive(text).split()) - STOPWORDS - _PUNCT_TOKENS - NO_GAP_LOANWORDS


def bangla_majority(text: str) -> bool:
    bangla = sum("ঀ" <= char <= "৿" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    return bangla >= latin and bangla > 0


def extract_question(row: dict[str, Any]) -> str:
    """v2 rows carry a cleaned `question`; older shards need recovery."""
    if isinstance(row.get("question"), str) and row["question"].strip():
        return " ".join(row["question"].split())
    parsed = row.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("question"), str):
        return " ".join(parsed["question"].split())
    text = str(row.get("raw_output") or "")
    marker = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if marker:
        try:
            return " ".join(json.loads(f'"{marker.group(1)}"').split())
        except json.JSONDecodeError:
            return " ".join(marker.group(1).split())
    return ""


def deciles(values: list[int]) -> list[int]:
    ordered = sorted(values)
    if not ordered:
        return []
    return [ordered[round((len(ordered) - 1) * i / 10)] for i in range(11)]


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, default=None,
                        help="where to write the gate report JSON (default: alongside --output)")
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
    human_lengths = [len(row["question"].split()) for row in human]
    low, high = percentile(human_lengths, 0.05), percentile(human_lengths, 0.95)
    blocked_questions = {normalize(row["question"]) for row in [*train, *dev, *test]}
    held_out = {
        corpus[chunk_id]["provision_id"]
        for row in [*dev, *test] for chunk_id in row["positive_chunk_ids"] if chunk_id in corpus
    }
    raw = read_jsonl(args.raw)

    scored: list[dict[str, Any]] = []
    empty = 0
    for row in raw:
        question = extract_question(row)
        chunk_ids = row.get("positive_chunk_ids") or []
        if not question:
            empty += 1
            continue
        if not chunk_ids or chunk_ids[0] not in corpus:
            scored.append({"question": question, "failures": ["unknown_chunk"]})
            continue
        document = corpus[chunk_ids[0]]
        passage = f"{document.get('provision_title_bn') or ''} {document.get('text_raw') or document.get('text_bn') or ''}"
        title = document.get("provision_title_bn") or document.get("provision_title_en") or ""
        title_terms = content_words(title)
        scored.append({
            "qid": f"llm_aug_v2_{row['prompt_id']}", "question": question,
            "positive_chunk_ids": chunk_ids, "hard_negative_chunk_ids": [],
            "provision_id": row["provision_id"], "act_id": document["act_id"], "domain": row["domain"],
            "lang_tag": "bengali", "source": "llm_augmentation_v2",
            "annotation_mode": "private_llm_generation", "label_source": "approved_provision_augmented_query",
            "review_status": "automated_quality_gated", "eligible_for_headline_training": True,
            "target_words": row["target_words"], "question_word_count": len(question.split()),
            "content_overlap": round(content_overlap(question, passage), 4),
            "title_content_overlap": round(len(content_words(question) & title_terms) / max(len(title_terms), 1), 4),
            "constrained_decoding": bool(row.get("constrained_decoding")),
            "generation_model": row.get("generation_model"),
        })

    owners: dict[str, set[str]] = collections.defaultdict(set)
    for row in scored:
        if "failures" not in row:
            owners[normalize(row["question"])].add(row["provision_id"])

    kills: collections.Counter = collections.Counter()
    combinations: collections.Counter = collections.Counter()
    valid: list[dict[str, Any]] = []
    for row in scored:
        failures = list(row.get("failures", []))
        if not failures:
            key = normalize(row["question"])
            if row["provision_id"] in held_out:
                failures.append("held_out_provision")
            if key in blocked_questions:
                failures.append("duplicate_of_split")
            if len(owners[key]) > 1:
                failures.append("cross_provision_duplicate")
            if not bangla_majority(row["question"]):
                failures.append("not_bangla")
            if not low <= row["question_word_count"] <= high:
                failures.append("length_outside_human_band")
            if row["content_overlap"] > BODY_OVERLAP_MAX:
                failures.append("body_overlap")
            if row["title_content_overlap"] != 0.0:
                failures.append("title_overlap")
        kills.update(failures)
        combinations[tuple(failures) or ("PASS",)] += 1
        if not failures:
            valid.append({key_: value for key_, value in row.items() if key_ != "failures"})

    kills["no_question"] = empty
    assert len({row["qid"] for row in valid}) == len(valid)
    write_jsonl(args.output, valid)

    overlaps = sorted(row["content_overlap"] for row in scored if "failures" not in row)
    report = {
        "raw_rows": len(raw),
        "rows_with_a_question": len(raw) - empty,
        "validated_rows": len(valid),
        "yield_of_raw": round(len(valid) / max(len(raw), 1), 4),
        "kills_per_gate": {gate: kills.get(gate, 0) for gate in GATES},
        "failure_combinations": [
            {"gates": list(gates), "rows": count} for gates, count in combinations.most_common(12)
        ],
        "body_overlap_median_all_generated": overlaps[len(overlaps) // 2] if overlaps else None,
        "body_overlap_p90_all_generated": overlaps[int(len(overlaps) * 0.9)] if overlaps else None,
        "human_body_overlap_median": 0.0,
        "human_body_overlap_p90": 0.091,
        "human_length_band": [low, high],
        "accepted_word_deciles": deciles([row["question_word_count"] for row in valid]),
        "human_word_deciles": deciles(human_lengths),
        "target_word_deciles": deciles([row["target_words"] for row in scored if "failures" not in row]),
        "gates_are_calibrated": "89.2% of human train / 93.2% of dev v3 / 92.0% of test v3 pass these gates",
    }
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
