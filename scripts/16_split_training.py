"""Split the synthetic pairs into train/dev, and assert the splits do not leak.

    python scripts/16_split_training.py
    python scripts/16_split_training.py --variant full

Writes data/processed/train_{variant}_v1.jsonl, data/processed/dev_{variant}_v1.jsonl
and results/tables/split_stats.csv.

## What the split unit has to be, and why it is not "chunk" here

CLAUDE.md says split by chunk, not by question, because synthetic questions
generated *from* a chunk are near-duplicates of each other and splitting by
question puts paraphrases of the same text on both sides.

This generator inverts that direction. Questions come from a hand-authored topic
inventory and are then matched to provisions, so the near-duplicate axis is the
**topic**, not the chunk: one topic's fifteen phrasings of "my husband divorced
me, what now" are the near-duplicates, and they are spread across every anchor
provision that topic owns. Splitting by chunk would put those phrasings on both
sides of the split — exactly the failure the original rule was written to
prevent, arrived at from the other direction.

So tier A splits by topic. A dev topic is entirely unseen at training time,
which makes dev a genuine generalisation test rather than a memorisation check.
Tier B/C splits by chunk, where the original reasoning does apply.

Chunks are deliberately *not* forced disjoint across splits. The corpus is shared
at evaluation time by construction — every retriever searches all 39,484 chunks —
so a provision appearing in both splits is not leakage. Questions leaking is
leakage, and that is what the assertions below check.

## Two variants, because 42% of gold provisions are also synthetic positives

`strict` (default) drops every pair whose positive provision appears anywhere in
the gold set. The fine-tuned model then never sees a gold answer during training,
so any improvement on the gold set has to be transfer — the model learned to
bridge register generally, not to memorise 92 specific provisions. This is the
variant the headline claim is measured on.

`full` keeps them. Reported as an ablation. Its number is not wrong, it just
answers a weaker question, and labelling which is which is the whole job.

## A third variant, because half the labels were wrong

`anchor` applies the same gold-provision exclusion as `strict` and additionally
keeps only pairs whose positive was named by hand in `configs/synth_anchors.yaml`
(`label_source: anchor`), discarding the keyword-matched ones
(`label_source: topic_signature`).

The discarded labels are not marginally worse, they are mostly wrong. Inspection
of ten random signature pairs on 2026-09-04 found roughly eight bad, and the
failures are English keyword collisions: `maintenance_khorposh` (a wife's
maintenance question) matched *The Chartered Accountants Ordinance, 1961* —
"Maintenance of branch offices"; `land_dispossession` matched the Penal Code's
"Possession of coin by person who knew it to be counterfeit"; `dower_denmohor`
matched the Rangamati Hill District Council Act. Under
MultipleNegativesRankingLoss a wrong positive does not merely fail to teach — it
actively pulls an unrelated provision toward citizen-question phrasing.

The cost is coverage: 173 distinct positive provisions instead of 1,557. That is
accepted deliberately. Transfer here has to come from learning the register
mapping (colloquial narrative to formal legal Bangla), and a correct mapping over
173 provisions teaches that better than a mapping over 1,557 where a third of the
targets are the wrong section of the wrong Act. `configs/synth_anchors.yaml` says
the same thing in its own header: the anchors are what training leans on.

Neither variant ever contains a gold *question*: synthetic questions are
template-generated and share no text with the 552 mined questions. The assertion
below checks that rather than assuming it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.normalize import aggressive  # noqa: E402

SYNTH = pathlib.Path("data/processed/synth_questions_v1.jsonl")
PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
GOLD = pathlib.Path("data/processed/gold_test_v1.jsonl")
AUDIT = pathlib.Path("results/runs/overlap_audit.json")
STATS = pathlib.Path("results/tables/split_stats.csv")

DEV_FRACTION = 0.15


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def bucket(key: str, seed: int) -> float:
    """Stable hash to [0, 1). Deterministic across runs and machines.

    Python's hash() is salted per process, so using it here would reshuffle the
    split on every run and silently invalidate any comparison between two
    training runs.
    """
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_key(row: dict) -> str:
    """The unit that must not straddle the split."""
    if row["quality_tier"].startswith("tier_a"):
        return f"topic:{row['topic_id']}"
    return f"chunk:{row['gold_chunk_ids'][0]}"


def assert_no_leakage(train: list[dict], dev: list[dict], gold_questions: set[str]) -> None:
    """Fail loudly. A silent leak is worth more marks lost than a crashed script."""
    train_units = {split_key(r) for r in train}
    dev_units = {split_key(r) for r in dev}
    straddling = train_units & dev_units
    assert not straddling, f"{len(straddling)} split units in both train and dev: {sorted(straddling)[:5]}"

    train_q = {r["question_bn"] for r in train}
    dev_q = {r["question_bn"] for r in dev}
    shared = train_q & dev_q
    assert not shared, f"{len(shared)} question strings in both train and dev, e.g. {sorted(shared)[:3]}"

    # The gold set is touched exactly once, at the end. Not here, not for
    # debugging. This checks question text only — the corpus is shared by design.
    for name, rows in (("train", train), ("dev", dev)):
        leaked = {r["question_bn"] for r in rows} & gold_questions
        assert not leaked, f"{len(leaked)} gold questions leaked into {name}: {sorted(leaked)[:3]}"

    # Near-duplicate check: normalized forms, in case two templates differ only
    # in punctuation or digit form.
    train_norm = {aggressive(q) for q in train_q}
    dev_norm = {aggressive(q) for q in dev_q}
    near = train_norm & dev_norm
    assert not near, f"{len(near)} normalized question forms in both splits"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=pathlib.Path, default=SYNTH)
    ap.add_argument("--variant", choices=["strict", "full", "anchor"], default="strict")
    ap.add_argument("--tiers", default="", help="comma-separated; default = audit's recommended tiers")
    ap.add_argument("--dev-fraction", type=float, default=DEV_FRACTION)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = read_jsonl(args.synth)
    print(f"synthetic pairs: {len(rows)}")

    # Tier selection defers to the §5.3 audit rather than being hard-coded, so a
    # regenerated dataset with different overlap automatically changes what is
    # trained on instead of quietly training on failed tiers.
    if args.tiers:
        tiers = set(args.tiers.split(","))
        print(f"tiers (from --tiers): {sorted(tiers)}")
    else:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        tiers = set(audit["recommended_tiers"])
        print(f"tiers (from {AUDIT}, §5.3 audit): {sorted(tiers)}")
        for tier, verdict in audit["tiers"].items():
            if verdict["verdict"] == "fail":
                print(f"  excluded {tier}: median overlap {verdict['content_overlap']['median']} "
                      f"vs real p90 {audit['real']['content_overlap']['p90']}")
    rows = [r for r in rows if r["quality_tier"] in tiers]
    print(f"after tier filter: {len(rows)}")

    probes = read_jsonl(PROBES)
    gold_provisions = {p for row in probes for p in row["gold_provision_ids"]}
    gold_questions = {row["question_bn"] for row in probes}
    if GOLD.exists():
        for row in read_jsonl(GOLD):
            gold_questions.add(row["question_bn"])
            gold_provisions.update(row.get("relevant_provision_ids") or [])

    # `anchor` keeps only hand-named positives. Applied before the gold-provision
    # filter so the printed counts read in the order the filters are applied.
    if args.variant == "anchor":
        before = len(rows)
        rows = [r for r in rows if r.get("label_source") == "anchor"]
        print(f"anchor: dropped {before - len(rows)} keyword-matched pairs "
              f"(label_source != 'anchor'); {len(rows)} hand-anchored pairs remain")

    if args.variant in ("strict", "anchor"):
        before = len(rows)
        rows = [r for r in rows if r["gold_provision_ids"][0] not in gold_provisions]
        print(f"{args.variant}: dropped {before - len(rows)} pairs whose positive is a gold provision "
              f"({len(gold_provisions)} gold provisions)")

    train, dev = [], []
    for row in rows:
        (dev if bucket(split_key(row), args.seed) < args.dev_fraction else train).append(row)

    assert_no_leakage(train, dev, gold_questions)
    print(f"\ntrain {len(train)} | dev {len(dev)} — leakage assertions passed")

    out_train = pathlib.Path(f"data/processed/train_{args.variant}_v1.jsonl")
    out_dev = pathlib.Path(f"data/processed/dev_{args.variant}_v1.jsonl")
    for dest, subset in ((out_train, train), (out_dev, dev)):
        with dest.open("w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {dest}")

    def describe(subset: list[dict]) -> dict:
        return {
            "pairs": len(subset),
            "questions": len({r["question_bn"] for r in subset}),
            "provisions": len({r["gold_provision_ids"][0] for r in subset}),
            "topics": len({r["topic_id"] for r in subset}),
            "english": sum(1 for r in subset if r["lang_tag"] == "english"),
            "bengali": sum(1 for r in subset if r["lang_tag"] == "bengali"),
            "anchored": sum(1 for r in subset if r.get("label_source") == "anchor"),
        }

    stats = {"train": describe(train), "dev": describe(dev)}
    for name, s in stats.items():
        print(f"  {name}: {s}")
    print(f"  dev topics: {sorted({r['topic_id'] for r in dev})}")

    STATS.parent.mkdir(parents=True, exist_ok=True)
    write_header = not STATS.exists()
    with STATS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["variant", "split", "pairs", "questions", "provisions", "topics",
                             "english", "bengali", "anchored"])
        for name, s in stats.items():
            writer.writerow([args.variant, name, s["pairs"], s["questions"], s["provisions"],
                             s["topics"], s["english"], s["bengali"], s["anchored"]])
    print(f"wrote {STATS}")


if __name__ == "__main__":
    main()
