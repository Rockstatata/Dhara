"""Build the v2 probe set from the human-verified gold, in script 13/26/28's schema.

    python scripts/32_build_probes.py

Everything in results/runs/*.json today is scored against probe_questions.jsonl
(built 2026-08-31, before round-2 human verification found ~57% of section-level
labels wrong). data/processed/gold_verified_v2.jsonl is the verified replacement
(552 rows, 18 disagreements adjudicated) but nothing has been scored against it
yet. This script is that bridge.

Split:
  - answerable == True  (480 rows) -> data/processed/probe_questions_v2.jsonl,
    same schema as the old probe file, so scripts 13/18/26/28 work unmodified.
  - answerable == False (72 rows)  -> data/processed/abstain_calibration_v2.jsonl,
    held out for abstention-threshold calibration, never fed to a retriever.

Two fields don't exist in gold_verified_v2.jsonl and have to be derived:
  - lang_tag: majority vote of corpus language over each question's
    relevant_chunk_ids ('english'/'bengali'), 'mixed' on a tie or a split gold set.
  - block: reused from the `domain` field. It's informational grouping carried
    through per_query, not a gate on correctness, so domain is the cheapest
    correct choice and doubles as a sanity cross-check against result_slices.

gold_test_v1.jsonl is never read or touched here — register/domain slicing for
v2 reads register+domain directly off gold_verified_v2.jsonl instead (see
scripts/28_result_slices.py --gold).
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
GOLD_V2 = pathlib.Path("data/processed/gold_verified_v2.jsonl")
TEST_SPLIT = pathlib.Path("data/processed/test_split_v1.json")
OUT_PROBES = pathlib.Path("data/processed/probe_questions_v2.jsonl")
OUT_ABSTAIN = pathlib.Path("data/processed/abstain_calibration_v2.jsonl")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def chunk_language_map(corpus_path: pathlib.Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["chunk_id"]] = row["language"]
    return mapping


def lang_tag_for(chunk_ids: list[str], language_of: dict[str, str]) -> str:
    langs = [language_of[c] for c in chunk_ids if c in language_of]
    if not langs:
        return "mixed"
    counts = Counter(langs)
    if len(counts) == 1:
        return next(iter(counts))
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "mixed"
    # majority present but not unanimous -> the gold set spans both languages,
    # report as mixed rather than picking a side arbitrarily.
    return "mixed"


def main() -> None:
    gold = read_jsonl(GOLD_V2)
    language_of = chunk_language_map(CORPUS)

    missing_chunks: set[str] = set()
    probes: list[dict] = []
    abstain: list[dict] = []

    for row in gold:
        chunk_ids = row["relevant_chunk_ids"]
        missing_chunks.update(c for c in chunk_ids if c not in language_of)

        if row["answerable"]:
            probes.append({
                "qid": row["qid"],
                "question_bn": row["question_bn"],
                "gold_chunk_ids": chunk_ids,
                "gold_provision_ids": row["relevant_provision_ids"],
                "lang_tag": lang_tag_for(chunk_ids, language_of),
                "annotation_mode": row.get("annotation_mode", "assisted"),
                "block": row.get("domain", "other"),
            })
        else:
            abstain.append({
                "qid": row["qid"],
                "question_bn": row["question_bn"],
                "verdict": row["verdict"],
                "register": row.get("register"),
                "domain": row.get("domain"),
            })

    if missing_chunks:
        print(f"WARNING: {len(missing_chunks)} gold chunk_ids not found in corpus, "
              f"e.g. {sorted(missing_chunks)[:5]}")

    OUT_PROBES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PROBES.open("w", encoding="utf-8") as fh:
        for p in probes:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    with OUT_ABSTAIN.open("w", encoding="utf-8") as fh:
        for a in abstain:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"wrote {OUT_PROBES} ({len(probes)} rows)")
    print(f"wrote {OUT_ABSTAIN} ({len(abstain)} rows)")
    print(f"  lang_tag distribution: {dict(Counter(p['lang_tag'] for p in probes))}")
    print(f"  block (domain) distribution: {dict(Counter(p['block'] for p in probes))}")

    if TEST_SPLIT.exists():
        split = json.loads(TEST_SPLIT.read_text(encoding="utf-8"))
        split_qids = set(split["test_qids"]) | set(split["train_qids"])
        gold_qids = {r["qid"] for r in gold}
        missing_from_gold = split_qids - gold_qids
        if missing_from_gold:
            print(f"WARNING: {len(missing_from_gold)} test_split_v1.json qids are "
                  f"absent from gold_verified_v2.jsonl, e.g. "
                  f"{sorted(missing_from_gold)[:5]} — the frozen split may not "
                  f"apply cleanly to v2.")
        else:
            print(f"OK: all {len(split_qids)} test_split_v1.json qids are present "
                  f"in gold_verified_v2.jsonl — the frozen 200/352 split still applies.")
        unanswerable_in_split = split_qids & {r["qid"] for r in gold if not r["answerable"]}
        if unanswerable_in_split:
            print(f"  note: {len(unanswerable_in_split)} split qids are unanswerable "
                  f"and therefore absent from probe_questions_v2.jsonl (present in "
                  f"abstain_calibration_v2.jsonl instead).")
    else:
        print(f"WARNING: {TEST_SPLIT} not found, skipped split-coverage check.")


if __name__ == "__main__":
    main()
