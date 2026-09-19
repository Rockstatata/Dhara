"""Generate synthetic question->provision training pairs.

    python scripts/14_generate_questions.py
    python scripts/14_generate_questions.py --tiers a          # tier A only
    python scripts/14_generate_questions.py --sample 20        # eyeball the output

Writes data/processed/synth_questions_v1.jsonl and
results/tables/synth_generation_stats.csv.

Two generators, deliberately different in kind:

**Tier A — citizen topic templates.** Questions are written first, in colloquial
Bangla, without seeing any provision (configs/synth_topics.yaml), then matched to
provisions by a topic signature. This is the inversion that makes the dataset
valid: a template cannot leak a provision's vocabulary because it was authored
before the provision was chosen. See src/dhara/synth.py for the full argument.

**Tier B — title paraphrase.** For provisions no topic reaches, the provision
title is rewritten through the register lexicon (data/lexicon/register_map.json)
into a question. Broad coverage, weaker register fidelity, and honestly labelled
as such. A title with no lexicon hit yields tier C, which is coverage filler.

Both are weak labels. Nothing here is human-adjudicated and nothing here may ever
enter the gold set — `09_run_eval.py`'s leakage assertion is the backstop, and
this script writes `source: synthetic` on every row so that assertion has
something to key on.

**The register filter runs here, not downstream.** A question containing a banned
statute term for its topic is discarded at generation time and counted, so the
discard rate is a reported number rather than a silent one.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.normalize import aggressive  # noqa: E402
from dhara.synth import (  # noqa: E402
    Narrative,
    select_positives,
    Topic,
    is_askable,
    is_repealed,
    make_row,
    match_score_pre,
    normalized_signature,
    overlap_summary,
    prepare,
)

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TOPICS = pathlib.Path("configs/synth_topics.yaml")
ANCHORS = pathlib.Path("configs/synth_anchors.yaml")
TEMPLATES_EXTRA = pathlib.Path("configs/synth_templates_extra.yaml")
NARRATIVE = pathlib.Path("configs/synth_narrative.yaml")
LEXICON = pathlib.Path("data/lexicon/register_map.json")
OUT = pathlib.Path("data/processed/synth_questions_v1.jsonl")
STATS = pathlib.Path("results/tables/synth_generation_stats.csv")

# A topic match below this is noise — one incidental signature word in a long
# section about something else.
MIN_MATCH = 0.12
# Each generated question is paired with at most this many provisions. More than
# a handful and one template floods training with near-identical examples.
MAX_POSITIVES_PER_QUESTION = 3
# §5.3's filter threshold. Reported, not hidden.
MAX_CONTENT_OVERLAP = 0.5


def load_topics(path: pathlib.Path) -> list[Topic]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    topics = [Topic(**t) for t in raw["topics"]]

    # Second authoring pass, merged in. Kept in its own file so the annotated
    # topic definitions stay readable; the extras are phrasings, not topics.
    if TEMPLATES_EXTRA.exists():
        extra = yaml.safe_load(TEMPLATES_EXTRA.read_text(encoding="utf-8"))["extra"]
        known = {t.topic_id for t in topics}
        for topic_id in extra:
            if topic_id not in known:
                raise SystemExit(
                    f"{TEMPLATES_EXTRA} has templates for unknown topic '{topic_id}'"
                )
        for topic in topics:
            topic.templates = topic.templates + extra.get(topic.topic_id, [])
    return topics


def load_anchors(path: pathlib.Path, chunks: list) -> dict[str, list]:
    """topic_id -> [Chunk, ...], resolved from (act_id, section) and verified.

    An anchor naming a section that is not in the corpus is a typo or a scraping
    gap, and either way it must be loud: a silently-dropped anchor turns into a
    topic quietly falling back to signature matching, which is the failure this
    file exists to prevent.
    """
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["anchors"]
    by_act_section: dict[tuple[str, str], list] = {}
    for chunk in chunks:
        by_act_section.setdefault((chunk.act_id, chunk.provision_no_ascii), []).append(chunk)

    resolved: dict[str, list] = {}
    unresolved: list[str] = []
    for topic_id, entries in raw.items():
        found = []
        for entry in entries:
            for section in entry["sections"]:
                hit = by_act_section.get((entry["act_id"], section))
                if hit:
                    found.extend(hit)
                else:
                    unresolved.append(f"{topic_id}: {entry['act_id']} s{section}")
        resolved[topic_id] = found
    if unresolved:
        print(f"  WARNING: {len(unresolved)} anchors did not resolve against the corpus:")
        for u in unresolved[:20]:
            print(f"    {u}")
    return resolved


# --------------------------------------------------------------------------
# Tier A
# --------------------------------------------------------------------------


def generate_tier_a(
    topics: list[Topic],
    chunks: list,
    anchors: dict[str, list],
    seed: int,
    narrative: Narrative | None = None,
    variants_per_core: int = 2,
) -> tuple[list[dict], dict]:
    counters: collections.Counter = collections.Counter()
    rows: list[dict] = []

    # Normalize the corpus once, not once per topic. Askability does not depend
    # on the topic either, so both are hoisted out of the matching loop.
    askable = [c for c in chunks if is_askable(c)]
    haystacks = prepare(askable)
    print(f"  tier A: matching {len(topics)} topics against {len(askable)} askable chunks")

    for topic in topics:
        signature, require = normalized_signature(topic)
        scored = []
        for chunk, hay in zip(askable, haystacks):
            score = match_score_pre(hay, signature, require)
            if score >= MIN_MATCH:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: -pair[0])
        counters[f"matched:{topic.topic_id}"] = len(scored)
        if not scored:
            counters["topics_with_no_match"] += 1
            continue

        cores = topic.render_all(max_per_template=10, seed=seed)
        counters["cores_rendered"] += len(cores)

        # Wrap each core ask in citizen-letter scaffolding. Real questions are
        # long narratives (median 74 words) and the bare cores are one-liners
        # (median 11); training on one and evaluating on the other is a
        # distribution gap worth closing. Each core yields several distinct
        # letters, which also raises the distinct-question count -- 413 was thin.
        # (core ask, composed letter) pairs. The core is carried through because
        # positive selection must read the ASK, not the scaffolding: a background
        # clause like "কার কাছে গেলে সঠিক পরামর্শ পাব" and a closing like "কী করা
        # উচিত" both look like a procedure question, so running intent detection
        # over the finished letter would tag nearly every question `procedure`.
        if narrative is not None:
            questions = []
            for core in cores:
                for letter in narrative.variants(core, topic.domain, topic.topic_id,
                                                 variants_per_core, seed):
                    questions.append((core, letter))
        else:
            questions = [(core, core) for core in cores]
        counters["questions_rendered"] += len(questions)

        # Every question that survives the register filter is paired with the
        # strongest few provisions; the rotation offset spreads coverage so a
        # topic's 40th-best provision still gets training signal instead of only
        # its top three appearing over and over.
        kept_questions = []
        for core, question in questions:
            leaked = topic.violates_register(question)
            if leaked:
                counters["discarded_register_leak"] += 1
                continue
            kept_questions.append((core, question))

        # Full relevant set per topic, so negative mining can avoid treating a
        # genuinely-relevant provision as a hard negative.
        relevant_provisions = sorted({c.provision_id for _s, c in scored[:50]})

        anchored = anchors.get(topic.topic_id, [])
        anchor_provisions = sorted({c.provision_id for c in anchored})
        counters[f"anchored:{topic.topic_id}"] = len(anchored)
        if not anchored:
            counters["topics_without_anchors"] += 1

        # Anchored positives are the training signal, and they are selected PER
        # QUESTION rather than per topic.
        #
        # The cross product this replaced was wrong in a way that reads as right:
        # a topic's anchors are all genuinely about the topic, so pairing every
        # question with every anchor looks safe. Measured, it gave a median of 6
        # positives per question of which typically one or two answer the actual
        # ask -- "কম বয়সে বিয়ে দিলে বাবা-মায়ের কি শাস্তি হয়?" was labelled with
        # the prohibition, the registrar's duty and the compensation sections,
        # and not with the parents' punishment section. MNRL pulls all of them
        # toward the question with equal force, so the wrong ones outvote the
        # right one.
        #
        # `select_positives` matches the ask's intent against the provision
        # title, and falls back to the full anchor list when it recognises
        # nothing -- counted, so the fallback rate is visible rather than
        # assumed.
        for core, question in kept_questions:
            selected, mode = select_positives(core, anchored)
            counters[f"positive_selection:{mode}"] += 1
            counters["positives_emitted"] += len(selected)
            for chunk in selected:
                row = make_row(
                    question,
                    chunk,
                    topic_id=topic.topic_id,
                    generator="tier_a",
                    quality_tier="tier_a",
                    match_score_value=1.0,
                )
                row["label_source"] = "anchor"
                row["positive_selection"] = mode
                row["also_relevant_provision_ids"] = [
                    p for p in anchor_provisions + relevant_provisions
                    if p != chunk.provision_id
                ]
                rows.append(row)

        # Signature matches beyond the anchors keep coverage up, at a tier that
        # says plainly they are weaker. Anchored provisions are skipped so the
        # same pair is not emitted twice under two labels.
        anchor_ids = {c.chunk_id for c in anchored}
        expansion = [(sc, ch) for sc, ch in scored if ch.chunk_id not in anchor_ids]
        for i, (_core, question) in enumerate(kept_questions):
            if not expansion:
                break
            offset = (i * MAX_POSITIVES_PER_QUESTION) % len(expansion)
            picked = [
                expansion[(offset + j) % len(expansion)]
                for j in range(min(MAX_POSITIVES_PER_QUESTION, len(expansion)))
            ]
            for score, chunk in picked:
                row = make_row(
                    question,
                    chunk,
                    topic_id=topic.topic_id,
                    generator="tier_a",
                    quality_tier="tier_a_expansion",
                    match_score_value=score,
                )
                row["also_relevant_provision_ids"] = [
                    p for p in anchor_provisions + relevant_provisions
                    if p != chunk.provision_id
                ]
                rows.append(row)

    return rows, counters


# --------------------------------------------------------------------------
# Tier B / C — provision-title questions
# --------------------------------------------------------------------------

QUESTION_FRAMES_BN = [
    "{topic} নিয়ে আইনে কী বলা আছে?",
    "{topic} বিষয়ে নিয়ম কী?",
    "{topic} ক্ষেত্রে আমার কী করা উচিত?",
    "{topic} সংক্রান্ত একটা সমস্যায় পড়েছি, কোথায় যাব?",
    "{topic} বিষয়ে কোন আইন প্রযোজ্য হবে?",
]

# Legal-Bangla morphology that reads as statute drafting rather than speech.
TITLE_TRIM = re.compile(r"\s*(ইত্যাদি|প্রভৃতি|,\s*ইত্যাদি)\s*$")


def lexicon_rewrite(title: str, lexicon: dict, rng: random.Random) -> tuple[str, bool]:
    """Rewrite a provision title into citizen phrasing. Returns (text, rewritten)."""
    text = TITLE_TRIM.sub("", title).strip(" ।.:-")
    rewritten = False
    # Longest keys first, so "criminal breach of trust" wins over "trust".
    for table in (lexicon["bn"], lexicon["en"]):
        for legal in sorted(table, key=len, reverse=True):
            if legal.startswith("_"):
                continue
            if legal.lower() in text.lower():
                citizen = rng.choice(table[legal])
                pattern = re.compile(re.escape(legal), re.IGNORECASE)
                text = pattern.sub(citizen, text, count=1)
                rewritten = True
    return text.strip(), rewritten


def generate_tier_b(
    chunks: list,
    lexicon: dict,
    covered_provisions: set[str],
    seed: int,
    max_per_chunk: int = 2,
) -> tuple[list[dict], dict]:
    counters: collections.Counter = collections.Counter()
    rows: list[dict] = []
    rng = random.Random(seed)

    for chunk in chunks:
        if chunk.provision_id in covered_provisions:
            counters["skipped_already_tier_a"] += 1
            continue
        if not is_askable(chunk):
            counters["skipped_not_askable"] += 1
            continue
        title = (chunk.provision_title_bn or "").strip()
        if len(title) < 6:
            counters["skipped_no_title"] += 1
            continue

        topic_text, rewritten = lexicon_rewrite(title, lexicon, rng)
        if not topic_text or len(topic_text) < 4:
            counters["skipped_empty_after_rewrite"] += 1
            continue

        tier = "tier_b" if rewritten else "tier_c"
        frames = rng.sample(QUESTION_FRAMES_BN, k=min(max_per_chunk, len(QUESTION_FRAMES_BN)))
        for frame in frames:
            question = frame.format(topic=topic_text)
            rows.append(
                make_row(
                    question,
                    chunk,
                    topic_id="_title",
                    generator="tier_b",
                    quality_tier=tier,
                    match_score_value=1.0,
                )
            )
            counters[f"generated:{tier}"] += 1

    return rows, counters


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--tiers", default="ab", help="which generators to run: a, b, or ab")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--sample", type=int, default=0, help="print N examples and exit")
    ap.add_argument("--max-overlap", type=float, default=MAX_CONTENT_OVERLAP)
    ap.add_argument("--variants", type=int, default=2,
                    help="narrative letters composed per core ask")
    ap.add_argument("--no-narrative", action="store_true",
                    help="emit bare one-line cores (the pre-2026-09-04 behaviour)")
    args = ap.parse_args()

    chunks = list(schema.read_jsonl(args.corpus))
    print(f"corpus: {len(chunks)} chunks")
    print(f"  repealed/omitted (excluded from generation): {sum(1 for c in chunks if is_repealed(c))}")
    print(f"  askable: {sum(1 for c in chunks if is_askable(c))}")

    topics = load_topics(TOPICS)
    lexicon = json.loads(LEXICON.read_text(encoding="utf-8"))
    print(f"topics: {len(topics)} | lexicon: {len(lexicon['bn'])} bn + {len(lexicon['en'])} en entries")

    rows: list[dict] = []
    counters: collections.Counter = collections.Counter()

    anchors = load_anchors(ANCHORS, chunks)
    print(f"anchors: {sum(len(v) for v in anchors.values())} chunks across {len(anchors)} topics")

    narrative = None
    if not args.no_narrative and NARRATIVE.exists():
        narrative = Narrative.from_yaml(NARRATIVE)
        print(f"narrative scaffolding: {sum(len(v) for v in narrative.openers.values())} openers, "
              f"{sum(len(v) for v in narrative.backgrounds.values())} background clauses, "
              f"{len(narrative.closings)} closings")

    if "a" in args.tiers:
        a_rows, a_counters = generate_tier_a(
            topics, chunks, anchors, args.seed,
            narrative=narrative, variants_per_core=args.variants,
        )
        rows += a_rows
        counters.update(a_counters)
        print(f"tier A: {len(a_rows)} pairs")

    covered = {r["gold_provision_ids"][0] for r in rows}
    if "b" in args.tiers:
        b_rows, b_counters = generate_tier_b(chunks, lexicon, covered, args.seed)
        rows += b_rows
        counters.update(b_counters)
        print(f"tier B/C: {len(b_rows)} pairs")

    # §5.3 filter. Applied to every tier and counted per tier so the report can
    # say exactly what was discarded and why.
    before = len(rows)
    discarded_by_tier: collections.Counter = collections.Counter()
    kept = []
    for row in rows:
        if row["content_overlap"] > args.max_overlap:
            discarded_by_tier[row["quality_tier"]] += 1
            continue
        kept.append(row)
    rows = kept
    print(f"overlap filter (> {args.max_overlap}): discarded {before - len(rows)} of {before}")
    for tier, n in sorted(discarded_by_tier.items()):
        print(f"    {tier}: {n}")

    if args.sample:
        for row in random.Random(0).sample(rows, min(args.sample, len(rows))):
            print(f"\n[{row['quality_tier']} / {row['topic_id']} / overlap {row['content_overlap']}]")
            print(f"  Q: {row['question_bn']}")
            print(f"  A: {row['gold_chunk_ids'][0]} ({row['lang_tag']}, {row['domain']})")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    tiers = collections.Counter(r["quality_tier"] for r in rows)
    domains = collections.Counter(r["domain"] for r in rows)
    langs = collections.Counter(r["lang_tag"] for r in rows)
    provisions = {r["gold_provision_ids"][0] for r in rows}
    questions = {r["question_bn"] for r in rows}

    print(f"\nwrote {args.out}: {len(rows)} pairs")
    print(f"  distinct questions : {len(questions)}")
    print(f"  distinct provisions: {len(provisions)}")
    print(f"  tiers   : {dict(tiers)}")
    print(f"  language: {dict(langs)}")
    print(f"  domains : {dict(domains.most_common(8))}")
    print(f"  overlap : {overlap_summary([r['content_overlap'] for r in rows])}")

    STATS.parent.mkdir(parents=True, exist_ok=True)
    with STATS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["pairs", len(rows)])
        writer.writerow(["distinct_questions", len(questions)])
        writer.writerow(["distinct_provisions", len(provisions)])
        for tier, n in sorted(tiers.items()):
            writer.writerow([f"tier_{tier}", n])
        for lang, n in sorted(langs.items()):
            writer.writerow([f"lang_{lang}", n])
        for domain, n in domains.most_common():
            writer.writerow([f"domain_{domain}", n])
        for key, n in sorted(counters.items()):
            if not key.startswith("matched:"):
                writer.writerow([key, n])
        for key, value in overlap_summary([r["content_overlap"] for r in rows]).items():
            writer.writerow([f"overlap_{key}", value])
    print(f"wrote {STATS}")


if __name__ == "__main__":
    main()
