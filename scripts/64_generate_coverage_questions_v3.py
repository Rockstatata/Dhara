"""Generate balanced, full-corpus synthetic question->provision pairs.

    python scripts/64_generate_coverage_questions_v3.py                       # every active provision
    python scripts/64_generate_coverage_questions_v3.py --target-rows 9424    # balanced subset

This is the tier-B generator from `scripts/14_generate_questions.py` - provision
title rewritten through `data/lexicon/register_map.json` into a citizen frame -
with three changes, and it needs no LLM runtime.

1.  **Coverage is the point.** Script 14's tier B skipped any provision a topic
    template had already reached and required 25+ words to be "askable", so it
    was a filler for gaps. Here every active provision is generated for, and the
    output is balanced: the same number of variants per provision, and the
    subset sampler draws round-robin across Acts so no large Act dominates.

2.  **Length is sampled from the human distribution.** Script 14's tier B
    emitted the bare frame, median ~11 words, against a human median of 68 -
    the exact mismatch `configs/synth_narrative.yaml` was written to fix, but
    the narrative scaffolding was only ever applied to tier A. Here each variant
    draws a target word count from the 251 human-adjudicated questions and the
    letter is composed up to that length.

3.  **Every row is tiered by the gates it passes, not silently mixed.**
    `quality_tier` is `strict` when the row passes the same gates script 62
    applies (title content-overlap 0, body content-overlap <= 0.20, inside the
    human p05-p95 length band) and `coverage` otherwise.

**Read this before using the output.** A title-derived question is only as
non-copying as the lexicon rewrite that produced it, and `register_map.json`
holds ~190 hand-authored terms against 1,226 Acts. Provisions the lexicon does
not reach keep their title's vocabulary, which is what `coverage` marks. Those
rows are legitimate for the coverage-pretraining ablation, the same role
`pretrain_retrieval_coverage_v1.jsonl` already has; they are not human-like
citizen questions and must not be described as such. Only the `strict` tier is
eligible for the headline training claim. The counts are printed so the split is
a reported number rather than an assumption.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import pathlib
import random
import re
import sys
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.normalize import aggressive  # noqa: E402
from dhara.synth import STOPWORDS, Narrative, content_overlap, is_boilerplate, is_repealed  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
DEV = pathlib.Path("data/processed/dev_retrieval_v3.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v3.jsonl")
EXCLUDED = pathlib.Path("data/processed/excluded_repealed_v1.jsonl")
LEXICON = pathlib.Path("data/lexicon/register_map.json")
NARRATIVE = pathlib.Path("configs/synth_narrative.yaml")
OUT = pathlib.Path("data/processed/synth_coverage_questions_v3.jsonl")
AUDIT = pathlib.Path("results/runs/synth_coverage_questions_v3.json")
SAMPLE = pathlib.Path("results/tables/synth_coverage_questions_v3_sample.csv")

BODY_OVERLAP_MAX = 0.20
SEED = 20260917

# Drafting scaffolding that carries no subject matter. Removing it from a title
# before the frame is filled is not an overlap trick - these words are what make
# a title read as statute rather than as a thing a person has a problem with.
TITLE_NOISE = {
    "বিধান", "বিধানাবলী", "বিশেষ", "সংক্রান্ত", "সম্পর্কিত", "সম্পর্কে", "বিষয়ে",
    "ইত্যাদি", "প্রভৃতি", "সাধারণ", "প্রয়োগ", "ক্ষমতা", "সংরক্ষণ", "অতঃপর",
    "উক্ত", "তৎসংক্রান্ত", "প্রযোজ্যতা", "ব্যাখ্যা", "রহিতকরণ", "হেফাজতকরণ",
}
TITLE_TRIM = re.compile(r"\s*(ইত্যাদি|প্রভৃতি|,\s*ইত্যাদি)\s*$")


def load_tier_b_helpers():
    """Reuse script 14's `lexicon_rewrite` rather than forking it.

    A second copy of the rewrite would drift from the one that produced the
    already-approved pairs, and then two batches of training data would have
    been built by two slightly different rules.  The module name starts with a
    digit, so it has to be loaded by path.
    """
    path = pathlib.Path(__file__).resolve().parent / "14_generate_questions.py"
    spec = importlib.util.spec_from_file_location("tier_b_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.lexicon_rewrite, module.QUESTION_FRAMES_BN


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def content_words(text: str) -> set[str]:
    return set(aggressive(text).split()) - STOPWORDS


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[round((len(values) - 1) * fraction)]


def deciles(values: list[int]) -> list[int]:
    ordered = sorted(values)
    return [ordered[round((len(ordered) - 1) * i / 10)] for i in range(11)] if ordered else []


def strip_title_noise(title: str) -> str:
    """Drop drafting scaffolding, keep the subject."""
    cleaned = TITLE_TRIM.sub("", title).strip(" ।.:-")
    kept = [word for word in cleaned.split() if word.strip("।,;:-") not in TITLE_NOISE]
    return " ".join(kept).strip(" ।.:-") or cleaned


def compose_to_length(narrative: Narrative, core: str, domain: str,
                      rng: random.Random, target_words: int) -> str:
    """`Narrative.compose`, but adding background clauses until a length is met.

    `Narrative.compose` draws its background count from fixed weights, which
    gives a length distribution the caller cannot steer.  Retrieval training
    needs the generated lengths to track the human distribution, so the count is
    driven by the target instead.  Composition order is unchanged - opener,
    background, the ask, closing - because a real legal-advice letter puts the
    question last and a model trained on the ask-first order would meet a
    different shape at evaluation time.
    """
    speaker = narrative.speaker_of(core)
    pool = list(narrative.backgrounds.get("generic", []))
    pool += narrative.backgrounds.get(domain, [])
    if domain in narrative.sensitive_domains:
        pool += narrative.backgrounds.get("sensitive", [])
    pool = narrative._compatible(pool, speaker)

    opener_pool = [text for text in narrative.openers.get("neutral", [])]
    if speaker:
        opener_pool += narrative.openers.get(speaker, [])
    opener_pool = narrative._compatible(opener_pool, speaker)

    closing = rng.choice(narrative.closings).strip() if narrative.closings else ""
    core = core.strip()
    parts: list[str] = []
    if opener_pool:
        opener = rng.choice(opener_pool).strip()
        if opener:
            parts.append(opener)

    def words(items: list[str]) -> int:
        return len(" ".join([*items, core, closing]).split())

    available = pool[:]
    rng.shuffle(available)
    for clause in available:
        if words(parts) >= target_words:
            break
        parts.append(clause.strip())
    parts.append(core)
    if closing:
        parts.append(closing)
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()


def qid_for(question: str, chunk_id: str) -> str:
    digest = hashlib.sha1(f"{question}|{chunk_id}".encode("utf-8")).hexdigest()[:12]
    return f"synth_cov_v3_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    parser.add_argument("--train", type=pathlib.Path, default=TRAIN)
    parser.add_argument("--dev", type=pathlib.Path, default=DEV)
    parser.add_argument("--test", type=pathlib.Path, default=TEST)
    parser.add_argument("--excluded", type=pathlib.Path, default=EXCLUDED)
    parser.add_argument("--lexicon", type=pathlib.Path, default=LEXICON)
    parser.add_argument("--narrative", type=pathlib.Path, default=NARRATIVE)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--audit", type=pathlib.Path, default=AUDIT)
    parser.add_argument("--sample", type=pathlib.Path, default=SAMPLE)
    parser.add_argument("--per-provision", type=int, default=4,
                        help="variants per provision; the same for every provision, which is the balance")
    parser.add_argument("--target-rows", type=int, default=0,
                        help="if set, emit a balanced subset of this size, drawn round-robin across Acts")
    parser.add_argument("--min-title-chars", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.out.exists() or args.audit.exists():
        raise SystemExit("v3 coverage artifacts already exist; create a new version rather than overwrite")

    lexicon_rewrite, frames = load_tier_b_helpers()
    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))
    narrative = Narrative.from_yaml(args.narrative)
    corpus = read_jsonl(args.corpus)
    by_chunk = {row["chunk_id"]: row for row in corpus}

    train, dev, test = read_jsonl(args.train), read_jsonl(args.dev), read_jsonl(args.test)
    human_lengths = [len(row["question"].split()) for row in train
                     if row.get("source") == "human_adjudicated_v2"]
    low, high = percentile(human_lengths, 0.05), percentile(human_lengths, 0.95)
    blocked_questions = {" ".join(row["question"].casefold().split()) for row in [*train, *dev, *test]}
    held_out = {by_chunk[chunk_id]["provision_id"]
                for row in [*dev, *test] for chunk_id in row["positive_chunk_ids"] if chunk_id in by_chunk}

    # One chunk per provision: a provision split across sub-chunks would
    # otherwise get several times the variants of a provision that was not, and
    # the per-provision balance would be balance in name only.
    excluded = {row["chunk_id"] for row in read_jsonl(args.excluded)} if args.excluded.exists() else set()
    primary: dict[str, dict[str, Any]] = {}
    counters: collections.Counter = collections.Counter()
    for row in corpus:
        chunk = RowChunk(row)
        if row["chunk_id"] in excluded or is_repealed(chunk):
            counters["skipped_repealed"] += 1
            continue
        # "Short title and commencement" is a real provision that no citizen has
        # a question about; a question generated from it is a false positive
        # waiting to be retrieved.
        if is_boilerplate(chunk):
            counters["skipped_boilerplate"] += 1
            continue
        if row["provision_id"] in held_out:
            counters["skipped_held_out"] += 1
            continue
        title = (row.get("provision_title_bn") or "").strip()
        if len(title) < args.min_title_chars:
            counters["skipped_no_title"] += 1
            continue
        current = primary.get(row["provision_id"])
        if current is None or int(row.get("sub_idx") or 0) < int(current.get("sub_idx") or 0):
            primary[row["provision_id"]] = row

    rows: list[dict[str, Any]] = []
    for provision_id, document in sorted(primary.items()):
        title = (document.get("provision_title_bn") or "").strip()
        passage = f"{title} {document.get('text_raw') or document.get('text_bn') or ''}"
        title_terms = content_words(title)
        domain = document.get("domain") or ""
        seen_questions: set[str] = set()
        for variant in range(1, args.per_provision + 1):
            rng = random.Random(f"{args.seed}:{provision_id}:{variant}")
            target_words = rng.choice(human_lengths)
            topic_text, rewritten = lexicon_rewrite(strip_title_noise(title), lexicon, rng)
            if not topic_text or len(topic_text) < 4:
                counters["skipped_empty_after_rewrite"] += 1
                continue
            core = rng.choice(frames).format(topic=topic_text)
            question = compose_to_length(narrative, core, domain, rng, target_words)
            key = " ".join(question.casefold().split())
            if key in seen_questions or key in blocked_questions:
                counters["skipped_duplicate"] += 1
                continue
            seen_questions.add(key)

            body_overlap = content_overlap(question, passage)
            title_overlap = len(content_words(question) & title_terms) / max(len(title_terms), 1)
            word_count = len(question.split())
            strict = (title_overlap == 0.0 and body_overlap <= BODY_OVERLAP_MAX
                      and low <= word_count <= high)
            rows.append({
                "qid": qid_for(question, document["chunk_id"]),
                "question": question,
                "positive_chunk_ids": [document["chunk_id"]],
                "hard_negative_chunk_ids": [],
                "provision_id": provision_id,
                "act_id": document["act_id"],
                "domain": domain,
                "lang_tag": "bengali",
                "source": "synthetic_coverage_v3",
                "annotation_mode": "template_generated",
                "label_source": "provision_title_register_rewrite",
                "review_status": "not_reviewed",
                "generator": "tier_b_narrative_v3",
                "quality_tier": "strict" if strict else "coverage",
                "lexicon_rewritten": bool(rewritten),
                "eligible_for_headline_training": bool(strict),
                "target_words": target_words,
                "question_word_count": word_count,
                "content_overlap": round(body_overlap, 4),
                "title_content_overlap": round(title_overlap, 4),
            })
            counters[f"generated:{'strict' if strict else 'coverage'}"] += 1

    # Cross-provision duplicates are a labelling error, not a style problem: the
    # same string pointing at two provisions teaches the encoder a contradiction.
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        owners[" ".join(row["question"].casefold().split())].add(row["provision_id"])
    before = len(rows)
    rows = [row for row in rows if len(owners[" ".join(row["question"].casefold().split())]) == 1]
    counters["dropped_cross_provision_duplicate"] = before - len(rows)

    if args.target_rows:
        rows = balanced_subset(rows, args.target_rows, args.seed)

    assert len({row["qid"] for row in rows}) == len(rows)
    write_jsonl(args.out, rows)

    strict_rows = [row for row in rows if row["quality_tier"] == "strict"]
    overlaps = sorted(row["content_overlap"] for row in rows)
    audit = {
        "rows": len(rows),
        "provisions_covered": len({row["provision_id"] for row in rows}),
        "acts_covered": len({row["act_id"] for row in rows}),
        "corpus_active_provisions_with_titles": len(primary),
        "per_provision_requested": args.per_provision,
        "tier_counts": dict(collections.Counter(row["quality_tier"] for row in rows)),
        "strict_provisions_covered": len({row["provision_id"] for row in strict_rows}),
        "strict_acts_covered": len({row["act_id"] for row in strict_rows}),
        "lexicon_rewritten_rows": sum(1 for row in rows if row["lexicon_rewritten"]),
        "body_overlap": {
            "median": overlaps[len(overlaps) // 2] if overlaps else None,
            "p90": overlaps[int(len(overlaps) * 0.9)] if overlaps else None,
            "human_median": 0.0, "human_p90": 0.091,
        },
        "word_deciles": deciles([row["question_word_count"] for row in rows]),
        "human_word_deciles": deciles(human_lengths),
        "human_length_band": [low, high],
        "rows_per_domain": dict(collections.Counter(row["domain"] for row in rows).most_common()),
        "skips": {key: value for key, value in sorted(counters.items())},
        "use_restriction": (
            "Only quality_tier=strict is eligible for headline training. quality_tier=coverage "
            "keeps the provision title's vocabulary and belongs in the coverage-pretraining "
            "ablation alongside pretrain_retrieval_coverage_v1.jsonl."
        ),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sample_rng = random.Random(args.seed)
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    with args.sample.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "qid", "provision_id", "domain", "quality_tier", "question_word_count",
            "content_overlap", "title_content_overlap", "question",
        ])
        writer.writeheader()
        for row in sample_rng.sample(rows, min(60, len(rows))):
            writer.writerow({key: row[key] for key in writer.fieldnames})

    print(json.dumps(audit, ensure_ascii=False, indent=2))


def balanced_subset(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin across Acts, then across provisions inside each Act.

    Taking the first N rows would hand most of the budget to the Penal Code and
    the CrPC, which have hundreds of provisions each, and leave single-provision
    Acts unrepresented. Round-robin gives every Act a row before any Act gets a
    second, so the cut stays balanced at whatever size it is stopped at.
    """
    by_act: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_act[row["act_id"]].append(row)
    rng = random.Random(seed)
    for act_rows in by_act.values():
        rng.shuffle(act_rows)
        act_rows.sort(key=lambda row: (row["provision_id"], row["qid"]))
    order = sorted(by_act)
    rng.shuffle(order)
    picked: list[dict[str, Any]] = []
    cursor = 0
    while len(picked) < target:
        progressed = False
        for act_id in order:
            if cursor < len(by_act[act_id]):
                picked.append(by_act[act_id][cursor])
                progressed = True
                if len(picked) >= target:
                    break
        if not progressed:
            break
        cursor += 1
    return picked


class RowChunk:
    """`is_repealed` reads attributes; corpus rows are dicts."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.__dict__.update(row)
        self.provision_title_bn = row.get("provision_title_bn") or ""
        self.text_bn = row.get("text_bn") or row.get("text_raw") or ""
        self.n_words = int(row.get("n_words") or 0)

    def __getattr__(self, name: str) -> Any:
        return None


if __name__ == "__main__":
    main()
