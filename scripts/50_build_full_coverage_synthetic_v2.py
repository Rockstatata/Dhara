"""Build an exhaustive, traceable synthetic coverage layer for Dhara.

    python scripts/50_build_full_coverage_synthetic_v2.py

This produces six deterministic, provision-grounded query forms for each active
parent provision: concise title-oriented queries and longer citizen-style
narratives. It is a *coverage-pretraining* resource, not human-written data:
every row remains explicitly marked synthetic. The trainable export excludes
every provision in the frozen human dev/test pool.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
from typing import Any

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.normalize import aggressive  # noqa: E402
from dhara.synth import STOPWORDS  # noqa: E402


CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
DEV = pathlib.Path("data/processed/dev_retrieval_v3.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v3.jsonl")
TRAIN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
CANDIDATES_OUT = pathlib.Path("data/processed/synthetic_coverage_candidates_v2.jsonl")
PRETRAIN_OUT = pathlib.Path("data/processed/pretrain_retrieval_coverage_v1.jsonl")
AUDIT_OUT = pathlib.Path("results/runs/synthetic_coverage_v2_audit.json")

REPEALED = re.compile(r"^\s*\[?\s*(omitted|repealed|রহিত করা হইয়াছে|বিলুপ্ত)", re.IGNORECASE)

BN_TEMPLATES = (
    "আমার ঘটনার সঙ্গে এই বিষয়টি জড়িত: ‘{fact}’। এ অবস্থায় কোন আইনি নিয়ম প্রযোজ্য?",
    "আমি একটি নোটিশ/সিদ্ধান্ত পাওয়ার পর বুঝতে পারছি না কী করব। বিষয়টির প্রেক্ষাপট হলো ‘{fact}’। আমার অধিকার, বাধ্যবাধকতা এবং পরের পদক্ষেপ কী?",
    "একজন {person} হিসেবে একটি বাস্তব সমস্যায় পড়েছি। ঘটনার গুরুত্বপূর্ণ অংশ: ‘{fact}’। কোথায় আবেদন বা যোগাযোগ করব এবং কোন শর্ত পূরণ করতে হতে পারে?",
    "বিষয়টি নিয়ে অন্য পক্ষের সঙ্গে বিরোধ হয়েছে। তারা বলছে ‘{fact}’। এই দাবি বা সিদ্ধান্তটি আমার ক্ষেত্রে কীভাবে দেখা হবে, এবং আমি কীভাবে বিষয়টি সমাধানের চেষ্টা করতে পারি?",
    "ঘटनাটি অনেক দিন ধরে চলেছে এবং এখন আমার করণীয় জানতে চাই। প্রাসঙ্গিক তথ্য হলো ‘{fact}’। কোনো প্রক্রিয়া, সময়সীমা, অনুমতি বা দায় থাকলে সহজ ভাষায় ব্যাখ্যা করুন।",
    "কর্তৃপক্ষের কাছে বিষয়টি তুলতে চাই, কিন্তু নিয়মটি বুঝতে পারছি না: ‘{fact}’। কার ক্ষমতা আছে, কী কাগজপত্র বা পদক্ষেপ লাগতে পারে, এবং আমার জন্য কী ফল হতে পারে?",
)
EN_TEMPLATES = (
    "My situation involves the following fact: ‘{fact}’. Which legal rule may apply?",
    "After receiving a notice or decision, I am unsure what to do. The relevant context is ‘{fact}’. What are my rights, obligations, and next steps?",
    "As a {person}, I have a practical problem. An important part of it is ‘{fact}’. Where should I apply or ask for help, and what conditions might I need to meet?",
    "I am in a dispute with another party. They rely on the following point: ‘{fact}’. How could that claim or decision be assessed in my situation, and how can I try to resolve it?",
    "This has been going on for some time and I need to know what to do. The relevant detail is ‘{fact}’. Please explain any process, deadline, permission, or responsibility in plain language.",
    "I want to raise this with the appropriate authority, but I do not understand the rule: ‘{fact}’. Who has authority, what documents or actions may be required, and what could happen next?",
)

BN_PERSONS = ("নাগরিক", "আবেদনকারী", "ক্ষতিগ্রস্ত ব্যক্তি", "সংশ্লিষ্ট পক্ষ")
EN_PERSONS = ("citizen", "applicant", "affected person", "concerned party")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[int], fraction: float) -> int:
    assert values
    return values[round((len(values) - 1) * fraction)]


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def is_repealed(row: dict[str, Any]) -> bool:
    return bool(
        REPEALED.match(row.get("provision_title_bn") or "")
        or REPEALED.match(row.get("text_bn") or "")
        or REPEALED.match(row.get("text_raw") or "")
    )


def bangla_majority(text: str) -> bool:
    bangla = sum("\u0980" <= char <= "\u09ff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    return bangla >= latin


def subject_for(row: dict[str, Any]) -> str:
    subject = (row.get("provision_title_bn") or "").strip()
    if not subject:
        subject = " ".join((row.get("text_raw") or row.get("text_bn") or "").split()[:22])
    if not subject:
        subject = f"section {row.get('provision_no_ascii') or ''} of {row.get('act_title_bn') or row.get('act_title_en') or 'this Act'}"
    return subject[:420]


def content_tokens(text: str) -> set[str]:
    return set(aggressive(text).split()) - STOPWORDS


def fact_for(row: dict[str, Any]) -> str:
    """Extract a short source clause without inventing a legal fact."""
    text = " ".join((row.get("text_raw") or row.get("text_bn") or "").split())
    if not text:
        return "the circumstances described in my case"
    # Keep one bounded source-derived clause. It provides factual anchoring for
    # narrative forms while avoiding an entire long statutory paragraph.
    pieces = re.split(r"(?<=[.!?।;:])\s+", text)
    title_terms = content_tokens(row.get("provision_title_bn") or "")
    clause = next((piece for piece in pieces if len(piece.split()) >= 5), pieces[0])
    # The question must not copy the section title.  Remove its content terms
    # from the factual anchor, leaving body-level facts where available.
    words = [
        word for word in clause.split()
        if not (set(aggressive(word).split()) & title_terms)
    ]
    fact = " ".join(words[:36]).strip(" ,;:")[:520]
    return fact if len(content_tokens(fact)) >= 3 else "the circumstances described in my case"


def qid_for(provision_id: str, variant: int) -> str:
    digest = hashlib.sha1(f"coverage-v2:{provision_id}:{variant}".encode("utf-8")).hexdigest()
    return f"coverage_v2_{digest[:16]}"


def overlap_metrics(question: str, passage_tokens: set[str]) -> tuple[float, float]:
    """Compute the established audit metrics with passage normalization cached."""
    question_tokens = set(aggressive(question).split())
    content_tokens = question_tokens - STOPWORDS
    content_overlap = len(content_tokens & passage_tokens) / max(len(content_tokens), 1)
    jaccard = len(question_tokens & passage_tokens) / max(len(question_tokens | passage_tokens), 1)
    return content_overlap, jaccard


def title_content_overlap(question: str, title: str) -> float:
    title_terms = content_tokens(title)
    if not title_terms:
        return 0.0
    return len(content_tokens(question) & title_terms) / len(title_terms)


def remove_title_terms(text: str, title: str) -> str:
    """Enforce the title-copy gate even when a template has a generic title term."""
    title_terms = content_tokens(title)
    if not title_terms:
        return text
    return " ".join(
        word for word in text.split()
        if not (set(aggressive(word).split()) & title_terms)
    )


def contextual_person(provision_id: str, variant: int, bangla: bool) -> str:
    people = BN_PERSONS if bangla else EN_PERSONS
    digest = hashlib.sha1(f"person:{provision_id}:{variant}".encode("utf-8")).digest()
    return people[digest[0] % len(people)]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approve-project-synthetic",
        action="store_true",
        help="record the project team's explicit approval to use this synthetic coverage layer in training",
    )
    args = parser.parse_args()
    if not args.approve_project_synthetic:
        raise SystemExit("refusing to mark synthetic rows training-eligible; pass --approve-project-synthetic after team approval")
    if any(path.exists() for path in (CANDIDATES_OUT, PRETRAIN_OUT, AUDIT_OUT)):
        raise SystemExit("v2 coverage outputs already exist; create a new version rather than overwrite them")
    corpus = read_jsonl(CORPUS)
    chunk_to_provision = {row["chunk_id"]: row["provision_id"] for row in corpus}
    # Each provision contributes its first chunk only; later chunks are the same
    # user-visible section and must not multiply its synthetic supervision.
    parent: dict[str, dict[str, Any]] = {}
    for row in corpus:
        current = parent.get(row["provision_id"])
        if current is None or row.get("sub_idx", 0) < current.get("sub_idx", 0):
            parent[row["provision_id"]] = row
    if not DEV.exists() or not TEST.exists() or not TRAIN.exists():
        raise SystemExit("build human-aware v3 splits first so human target lengths and dev/test exclusions exist")
    held_out = {
        chunk_to_provision[chunk_id]
        for row in [*read_jsonl(DEV), *read_jsonl(TEST)]
        for chunk_id in row["positive_chunk_ids"]
    }
    human_train = [
        row for row in read_jsonl(TRAIN)
        if row.get("source") == "human_adjudicated_v2"
    ]
    if not human_train:
        raise SystemExit("train_retrieval_v4 has no natural human training rows")
    human_lengths = sorted(len(row["question"].split()) for row in human_train)
    human_length_floor = percentile(human_lengths, 0.05)
    human_length_ceiling = percentile(human_lengths, 0.95)

    candidates: list[dict[str, Any]] = []
    counters: collections.Counter[str] = collections.Counter()
    for provision_number, (provision_id, row) in enumerate(sorted(parent.items()), 1):
        if is_repealed(row):
            counters["repealed_provisions"] += 1
            continue
        fact = fact_for(row)
        is_bangla = bangla_majority(row.get("text_raw") or row.get("text_bn") or row.get("provision_title_bn") or "")
        templates = BN_TEMPLATES if is_bangla else EN_TEMPLATES
        passage = f"{row.get('provision_title_bn') or ''} {row.get('text_raw') or row.get('text_bn') or ''}".strip()
        passage_tokens = set(aggressive(passage).split())
        for variant, template in enumerate(templates, 1):
            question = template.format(
                fact=fact,
                person=contextual_person(provision_id, variant, is_bangla),
            )
            question = remove_title_terms(question, row.get("provision_title_bn") or "")
            overlap, jaccard = overlap_metrics(question, passage_tokens)
            title_overlap = title_content_overlap(question, row.get("provision_title_bn") or "")
            candidates.append({
                "qid": qid_for(provision_id, variant),
                "question": question,
                "positive_chunk_ids": [row["chunk_id"]],
                "hard_negative_chunk_ids": [],
                "provision_id": provision_id,
                "act_id": row["act_id"],
                "domain": row["domain"],
                "lang_tag": "bengali" if bangla_majority(question) else "english",
                "source": "synthetic",
                "annotation_mode": "deterministic_template_generation",
                "label_source": "title_or_body_grounded_coverage",
                "synthetic_variant": variant,
                "review_status": "approved_project_instruction",
                "approval_record": "Project team approved use of this synthetic coverage layer on 2026-09-16.",
                "eligible_for_headline_training": False,
                "held_out_for_human_evaluation": provision_id in held_out,
                "content_overlap": round(overlap, 4),
                "jaccard": round(jaccard, 4),
                "title_content_overlap": round(title_overlap, 4),
                "title_overlap_gate": "pass" if title_overlap == 0 else "flag",
                "question_word_count": len(question.split()),
            })
        counters["active_provisions"] += 1
        if provision_number % 5000 == 0:
            print(
                f"generated candidates for {provision_number}/{len(parent)} parent provisions",
                flush=True,
            )

    # Identical generated questions assigned to different provisions are unsafe
    # for in-batch-negative training. Keep them in the coverage ledger, but not
    # the training export. Match the natural-human length band as well.
    question_owners: dict[str, set[str]] = collections.defaultdict(set)
    for row in candidates:
        question_owners[normalize_question(row["question"])].add(row["provision_id"])
    for row in candidates:
        ambiguous = len(question_owners[normalize_question(row["question"])]) > 1
        length_pass = human_length_floor <= row["question_word_count"] <= human_length_ceiling
        row["cross_provision_duplicate_gate"] = "flag" if ambiguous else "pass"
        row["human_length_profile_gate"] = "pass" if length_pass else "flag"
        row["eligible_for_coverage_pretraining"] = bool(
            not row["held_out_for_human_evaluation"] and not ambiguous and length_pass
        )
    trainable = [row for row in candidates if row["eligible_for_coverage_pretraining"]]
    assert len(candidates) == counters["active_provisions"] * len(BN_TEMPLATES)
    assert len({row["qid"] for row in candidates}) == len(candidates)
    assert not ({row["provision_id"] for row in trainable} & held_out)
    assert all(
        count == len(BN_TEMPLATES)
        for count in collections.Counter(row["provision_id"] for row in candidates).values()
    )
    assert all(row["title_overlap_gate"] == "pass" for row in candidates), "title-copy gate failed"
    assert all(row["cross_provision_duplicate_gate"] == "pass" for row in trainable)
    assert all(row["human_length_profile_gate"] == "pass" for row in trainable)
    write_jsonl(CANDIDATES_OUT, candidates)
    write_jsonl(PRETRAIN_OUT, trainable)

    overlaps = sorted(row["content_overlap"] for row in candidates)
    word_counts = sorted(len(row["question"].split()) for row in candidates)
    audit = {
        "version": "v2",
        "purpose": "exhaustive project-approved synthetic coverage pretraining; not human-authored data",
        "corpus_parent_provisions": len(parent),
        "active_provisions_covered": counters["active_provisions"],
        "repealed_provisions_excluded": counters["repealed_provisions"],
        "examples_per_active_provision": len(BN_TEMPLATES),
        "all_active_acts_covered": len({row["act_id"] for row in candidates}),
        "candidate_rows": len(candidates),
        "trainable_rows": len(trainable),
        "natural_human_training_word_count": {
            "p05": human_length_floor,
            "median": percentile(human_lengths, 0.5),
            "p95": human_length_ceiling,
        },
        "training_gates": {
            "held_out_provision_excluded": sum(row["held_out_for_human_evaluation"] for row in candidates),
            "cross_provision_duplicate_excluded": sum(row["cross_provision_duplicate_gate"] == "flag" for row in candidates),
            "outside_human_length_profile_excluded": sum(row["human_length_profile_gate"] == "flag" for row in candidates),
        },
        "held_out_provisions_excluded_from_train": len(held_out),
        "language_counts": dict(collections.Counter(row["lang_tag"] for row in candidates)),
        "domain_counts": dict(collections.Counter(row["domain"] for row in candidates)),
        "content_overlap": {
            "median": overlaps[len(overlaps) // 2],
            "p90": overlaps[int(0.9 * (len(overlaps) - 1))],
        },
        "title_content_overlap": {
            "maximum": max(row["title_content_overlap"] for row in candidates),
            "gate": "all rows must be zero; title terms are removed from factual anchors",
        },
        "question_word_count": {
            "minimum": word_counts[0],
            "median": word_counts[len(word_counts) // 2],
            "p90": word_counts[int(0.9 * (len(word_counts) - 1))],
            "maximum": word_counts[-1],
        },
        "limitations": [
            "Rows are deterministic synthetic templates grounded in a provision title or source text.",
            "They are project-approved for training but remain synthetic, not independently written citizen narratives.",
            "Only rows that pass duplicate and natural-human length gates enter the coverage-pretraining export.",
            "Use as a separate pretraining ablation, then fine-tune on approved title pairs and natural human questions.",
        ],
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
