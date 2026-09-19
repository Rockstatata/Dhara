"""Prepare v2 LLM prompts for human-profile legal-query augmentation.

    python scripts/60_prepare_diverse_augmentation_prompts_v2.py

The v1 ledger produced questions that failed the calibrated quality gates: the
Qwen shard had a median body content-overlap of 0.36 against a human median of
0.000, and its word count sat at ~25 regardless of the requested target (18, 55,
83 and 120 all came back at 20-26 words).  Three prompt-side causes were
identified and are fixed here:

1.  v1 banned only the *title* content words, so nothing discouraged copying the
    provision body.  v2 also bans the rarest (highest-IDF) body content words,
    which are the ones whose reuse is diagnostic of copying.  Common legal
    vocabulary is deliberately left available, because a citizen question about
    a bank does have to be able to say "bank"; the calibrated 0.20 body-overlap
    gate already tolerates a few shared common words.
2.  v1 asked for a word count in prose, which the model ignored entirely.  v2
    replaces the number with a *structural* scaffold (how many sentences, and
    what each one must contain) and keeps the number only as a secondary hint.
3.  v1 gave no exemplars, so the model had no reference for citizen register.
    v2 embeds two real human-adjudicated questions of comparable length, drawn
    from a different domain so no content can leak through them.

Target lengths are drawn from the empirical human length distribution rather
than from four fixed percentiles, so the generated corpus can match the human
*shape* (deciles 9/18/37/50/59/68/83/99/109/139/214) instead of four spikes.

The output is a prompt ledger, not labels.  It is restricted to provisions with
already approved training labels and excludes every frozen dev/test provision.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dhara.normalize import aggressive  # noqa: E402
from dhara.synth import STOPWORDS  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN = pathlib.Path("data/processed/train_retrieval_v4.jsonl")
DEV = pathlib.Path("data/processed/dev_retrieval_v3.jsonl")
TEST = pathlib.Path("data/processed/test_retrieval_v3.jsonl")
OUT = pathlib.Path("data/processed/diverse_augmentation_prompt_ledger_v2.jsonl")
AUDIT = pathlib.Path("results/runs/diverse_augmentation_prompt_ledger_v2.json")

VARIANTS_PER_PROVISION = 4
MAX_BANNED_BODY_WORDS = 60
SOURCE_CHAR_LIMIT = 1400
SEED = 20260917


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def content_words(text: str) -> list[str]:
    """Surface forms exactly as the overlap metric counts them.

    `content_overlap` compares `aggressive()`-normalized whitespace tokens with
    no stemming, so banning the same surface forms makes the ban list and the
    metric agree token for token.  Bangla is agglutinative, so an inflected
    variant is a different token to both - that is consistent, not a gap.
    """
    return [word for word in aggressive(text).split() if word not in STOPWORDS and len(word) > 2]


def document_frequency(corpus: list[dict[str, Any]]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for document in corpus:
        text = f"{document.get('provision_title_bn') or ''} {document.get('text_raw') or document.get('text_bn') or ''}"
        counter.update(set(content_words(text)))
    return counter


def scaffold(target_words: int) -> str:
    """Sentence-level structure, because the word count alone was ignored."""
    if target_words < 25:
        return (
            "STRUCTURE: exactly one short sentence. State the problem and ask the question in that "
            "one sentence. No greeting, no background."
        )
    if target_words < 60:
        return (
            "STRUCTURE: exactly two sentences. Sentence 1 states what happened to the person. "
            "Sentence 2 asks what the law says they can do. Invent one concrete detail "
            "(a name, a date, or a taka amount)."
        )
    if target_words < 100:
        return (
            "STRUCTURE: four sentences. (1) who the person is and their situation, (2) what the other "
            "side did, (3) what the person already tried, (4) the question they want answered. "
            "Invent concrete details: names, dates, taka amounts, place names."
        )
    return (
        "STRUCTURE: six or more sentences, told as a complaint someone would write to a legal-advice "
        "column. Cover: who the person is, how the situation started, what the other side did, what "
        "was said or promised, what the person already tried, and only at the very end the question "
        "they want answered. Invent rich concrete details: names, dates, taka amounts, place names, "
        "relationships. Do not summarise - narrate."
    )


def sample_targets(human_lengths: list[int], provision_id: str) -> list[int]:
    """Four draws from the human length distribution, one per quartile.

    Stratifying by quartile keeps every provision's four variants spread across
    short, medium and long, while sampling within the quartile means the
    aggregate distribution reproduces the human shape rather than four spikes.
    """
    rng = random.Random(f"{SEED}:{provision_id}")
    ordered = sorted(human_lengths)
    size = len(ordered)
    targets = []
    for quartile in range(VARIANTS_PER_PROVISION):
        lower = size * quartile // VARIANTS_PER_PROVISION
        upper = max(lower + 1, size * (quartile + 1) // VARIANTS_PER_PROVISION)
        targets.append(rng.choice(ordered[lower:upper]))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    parser.add_argument("--train", type=pathlib.Path, default=TRAIN)
    parser.add_argument("--dev", type=pathlib.Path, default=DEV)
    parser.add_argument("--test", type=pathlib.Path, default=TEST)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--audit", type=pathlib.Path, default=AUDIT)
    args = parser.parse_args()
    if args.out.exists() or args.audit.exists():
        raise SystemExit("v2 prompt artifacts already exist; create a new version rather than overwrite")

    corpus = read_jsonl(args.corpus)
    chunk = {row["chunk_id"]: row for row in corpus}
    frequency = document_frequency(corpus)
    train = read_jsonl(args.train)
    human = [row for row in train if row.get("source") == "human_adjudicated_v2"]
    approved = [row for row in train if row.get("label_source") == "human_approved_title_pair"]
    held_out_provisions = {
        chunk[chunk_id]["provision_id"]
        for row in [*read_jsonl(args.dev), *read_jsonl(args.test)]
        for chunk_id in row["positive_chunk_ids"]
        if chunk_id in chunk
    }
    approved_by_provision: dict[str, dict[str, Any]] = {}
    for row in approved:
        approved_by_provision.setdefault(row["provision_id"], row)

    exemplars = sorted(
        ({"question": " ".join(row["question"].split()), "domain": row.get("domain", ""),
          "words": len(row["question"].split())} for row in human),
        key=lambda item: item["words"],
    )
    human_lengths = [item["words"] for item in exemplars]

    def pick_exemplars(target_words: int, domain: str) -> list[str]:
        """Two human questions of comparable length from a *different* domain.

        A same-domain exemplar could hand the model vocabulary that belongs to
        the target provision, which would show up as overlap it did not have to
        read the statute to produce.
        """
        candidates = [item for item in exemplars if item["domain"] != domain]
        candidates.sort(key=lambda item: (abs(item["words"] - target_words), item["question"]))
        return [item["question"] for item in candidates[:2]]

    prompts: list[dict[str, Any]] = []
    for provision_id, label in sorted(approved_by_provision.items()):
        if provision_id in held_out_provisions:
            continue
        document = chunk[label["positive_chunk_ids"][0]]
        title = document.get("provision_title_bn") or ""
        body = " ".join((document.get("text_raw") or document.get("text_bn") or "").split())
        source_text = body[:SOURCE_CHAR_LIMIT]
        banned_title = sorted(set(content_words(title)))
        rare_body = sorted(
            set(content_words(source_text)) - set(banned_title),
            key=lambda word: (frequency[word], word),
        )[:MAX_BANNED_BODY_WORDS]
        banned = banned_title + rare_body
        domain = label.get("domain", "")

        for variant, target_words in enumerate(sample_targets(human_lengths, provision_id), 1):
            shown = pick_exemplars(target_words, domain)
            prompt = (
                "You are writing training data for a Bangla legal-search system. Write ONE question "
                "in Bangla, in the voice of an ordinary citizen with no legal training, about the "
                "statutory situation below.\n\n"
                "RULES:\n"
                "- Describe the situation as it would happen in real life. Never use legal prose.\n"
                "- Do NOT reuse any word from BANNED_WORDS. Find everyday wording instead.\n"
                "- Do NOT mention section numbers, article numbers, or act names.\n"
                "- Preserve who is involved, what triggers the situation, and what is at stake.\n"
                f"- Aim for roughly {target_words} Bangla words.\n"
                f"{scaffold(target_words)}\n\n"
                "Two real citizen questions, shown only so you match their voice and length. They are "
                "about unrelated matters; do not borrow their subject:\n"
                + "".join(f"EXAMPLE: {question}\n" for question in shown)
                + f"\nBANNED_WORDS: {' '.join(banned)}\n"
                f"SOURCE_PROVISION: {source_text}\n\n"
                "Output ONLY the Bangla question text. No JSON, no quotes, no labels, no explanation."
            )
            prompts.append({
                "prompt_id": f"aug_v2_{provision_id}_{variant}",
                "provision_id": provision_id,
                "positive_chunk_ids": label["positive_chunk_ids"],
                "domain": domain,
                "target_words": target_words,
                "banned_title_content_words": banned_title,
                "banned_body_content_words": rare_body,
                "exemplar_questions": shown,
                "prompt": prompt,
                "source": "private_llm_augmentation_candidate_v2",
                "review_status": "pending_generation_and_validation",
            })

    assert not ({row["provision_id"] for row in prompts} & held_out_provisions)
    write_jsonl(args.out, prompts)
    lengths = sorted(row["target_words"] for row in prompts)
    audit = {
        "approved_provisions_prompted": len({row["provision_id"] for row in prompts}),
        "prompts": len(prompts),
        "variants_per_provision": VARIANTS_PER_PROVISION,
        "target_word_deciles": [lengths[round((len(lengths) - 1) * i / 10)] for i in range(11)],
        "human_word_deciles": [human_lengths[round((len(human_lengths) - 1) * i / 10)] for i in range(11)],
        "max_banned_body_words": MAX_BANNED_BODY_WORDS,
        "median_banned_words": sorted(
            len(row["banned_title_content_words"]) + len(row["banned_body_content_words"]) for row in prompts
        )[len(prompts) // 2],
        "held_out_provisions_excluded": len(held_out_provisions),
        "promotion_requirement": "Run lexical, duplicate, length-shape and human review gates before training.",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
