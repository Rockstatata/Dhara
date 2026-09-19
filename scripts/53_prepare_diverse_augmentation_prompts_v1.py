"""Prepare private LLM prompts for human-profile legal-query augmentation.

    python scripts/53_prepare_diverse_augmentation_prompts_v1.py

The output is a prompt ledger, not labels. It is restricted to provisions with
already approved training labels and excludes every frozen dev/test provision.
The generator must return JSON candidates which are then passed through a
separate validation-and-review step before promotion.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
from typing import Any


CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
DEV = pathlib.Path("data/processed/dev_retrieval_v3.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v3.jsonl")
OUT = pathlib.Path("data/processed/diverse_augmentation_prompt_ledger_v1.jsonl")
AUDIT = pathlib.Path("results/runs/diverse_augmentation_prompt_ledger_v1.json")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def content_words(text: str) -> list[str]:
    return sorted({word for word in re.findall(r"[\w\u0980-\u09ff]+", text.casefold()) if len(word) > 2})


def percentile(values: list[int], fraction: float) -> int:
    return values[round((len(values) - 1) * fraction)]


def main() -> None:
    if OUT.exists() or AUDIT.exists():
        raise SystemExit("v1 prompt artifacts already exist; create a new version rather than overwrite")
    corpus = read_jsonl(CORPUS)
    chunk = {row["chunk_id"]: row for row in corpus}
    train = read_jsonl(TRAIN)
    human = [row for row in train if row.get("source") == "human_adjudicated_v2"]
    approved = [row for row in train if row.get("label_source") == "human_approved_title_pair"]
    held_out_chunks = {
        chunk_id for row in [*read_jsonl(DEV), *read_jsonl(TEST)] for chunk_id in row["positive_chunk_ids"]
    }
    held_out_provisions = {chunk[chunk_id]["provision_id"] for chunk_id in held_out_chunks}
    approved_by_provision: dict[str, dict[str, Any]] = {}
    for row in approved:
        approved_by_provision.setdefault(row["provision_id"], row)
    human_lengths = sorted(len(row["question"].split()) for row in human)
    target_lengths = [
        percentile(human_lengths, .10), percentile(human_lengths, .35),
        percentile(human_lengths, .60), percentile(human_lengths, .85),
    ]

    prompts: list[dict[str, Any]] = []
    for provision_id, label in sorted(approved_by_provision.items()):
        if provision_id in held_out_provisions:
            continue
        document = chunk[label["positive_chunk_ids"][0]]
        title = document.get("provision_title_bn") or document.get("provision_title_en") or ""
        source_text = " ".join((document.get("text_raw") or document.get("text_bn") or "").split())[:1800]
        banned = content_words(title)
        for variant, target_words in enumerate(target_lengths, 1):
            prompt = (
                "Write one Bangla citizen-style legal question about the statutory situation below. "
                "Preserve material actors, conditions, dates, amounts, authority, and timeline where present, "
                "but express them in an everyday scenario rather than legal prose. Do not quote or reuse any "
                "content word from the SECTION_TITLE banned list. Do not mention section numbers or act names. "
                f"Target approximately {target_words} Bangla words. Return ONLY JSON: "
                '{"question":"...","facts_preserved":["..."]}.\n'
                f"SECTION_TITLE_BANNED_WORDS: {banned}\nSOURCE_PROVISION: {source_text}"
            )
            prompts.append({
                "prompt_id": f"aug_v1_{provision_id}_{variant}",
                "provision_id": provision_id,
                "positive_chunk_ids": label["positive_chunk_ids"],
                "domain": label["domain"],
                "target_words": target_words,
                "banned_title_content_words": banned,
                "prompt": prompt,
                "source": "private_llm_augmentation_candidate",
                "review_status": "pending_generation_and_validation",
            })
    assert not ({row["provision_id"] for row in prompts} & held_out_provisions)
    write_jsonl(OUT, prompts)
    audit = {
        "approved_provisions_prompted": len({row["provision_id"] for row in prompts}),
        "prompts": len(prompts),
        "variants_per_provision": len(target_lengths),
        "human_length_targets": target_lengths,
        "held_out_provisions_excluded": len(held_out_provisions),
        "promotion_requirement": "Run lexical, duplicate, factual-preservation, and human review gates before training.",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
