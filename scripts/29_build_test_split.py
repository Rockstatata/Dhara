"""Split the 552 gold questions into a frozen test set and a training pool.

    python scripts/29_build_test_split.py
    python scripts/29_build_test_split.py --n-test 200

Writes data/processed/test_split_v1.json — the qid lists — and nothing else. No
model, no labels, no reordering of anything: this file only decides which
questions are allowed to be trained on.

## Why split at all

Every number in the project is currently measured on all 552 questions, and the
model is forbidden from training on any of them (`16_split_training.py --variant
strict` drops every pair whose positive is a gold provision). That keeps the
evaluation honest and wastes the single best supervision signal available: 552
real citizen questions labelled to provision level, in exactly the distribution
the system is tested on. The synthetic training pairs are an imitation of these.

`docs/Dhara_Proposal.md` §3 specifies a 200-question gold set. We have 552. So
200 stay frozen as the test set and 352 become training data — roughly seven
times more real supervision than the model has ever seen.

## What the split costs, stated plainly

**A wider interval.** At n=200 the 95% CI on Recall@5 is about ±7 points, so
adjacent rungs will sometimes be indistinguishable. CLAUDE.md anticipated this.

**The test 200 are not pristine.** Every design decision so far — document
template, hyperparameters, rerank depth, abstention fallbacks — was made while
looking at all 552. The 200 are unseen by the *model* but not by *us*, and the
paper must say so. It is still strictly better than the current arrangement,
where the same questions serve as dev and test simultaneously.

## How the 200 are chosen

Stratified on three axes at once, because each carries a claim the results table
has to support:

  - `lang_tag` — the cross-language wall (english / bengali / mixed);
  - `domain` — the per-domain breakdown;
  - triage band from `27_label_triage.py` — so the test set is not accidentally
    made of the labels most likely to be wrong.

Proportional allocation within strata, largest-remainder rounding, fixed seed.
The training pool is everything else.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
GOLD = pathlib.Path("data/processed/gold_test_v1.jsonl")
TRIAGE = pathlib.Path("results/runs/label_triage.json")
OUT = pathlib.Path("data/processed/test_split_v1.json")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    probes = read_jsonl(PROBES)
    domain_of = {}
    for row in read_jsonl(GOLD):
        domain_of.setdefault(row["qid"], (row.get("domain") or "unknown").strip())
    band_of = {}
    if TRIAGE.exists():
        band_of = {r["qid"]: r["band"]
                   for r in json.loads(TRIAGE.read_text(encoding="utf-8"))["rows"]}

    strata: dict[tuple, list[str]] = collections.defaultdict(list)
    for p in probes:
        key = (p["lang_tag"], domain_of.get(p["qid"], "unknown"),
               band_of.get(p["qid"], "untriaged"))
        strata[key].append(p["qid"])

    rng = random.Random(args.seed)
    for qids in strata.values():
        qids.sort()
        rng.shuffle(qids)

    # Largest-remainder allocation: floor everywhere, then hand the leftovers to
    # the strata with the biggest fractional parts. Rounding each stratum
    # independently would drift several questions away from the target.
    share = args.n_test / len(probes)
    exact = {key: len(qids) * share for key, qids in strata.items()}
    take = {key: int(v) for key, v in exact.items()}
    remainder = args.n_test - sum(take.values())
    for key, _frac in sorted(exact.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:remainder]:
        take[key] += 1

    test_qids: list[str] = []
    for key, qids in strata.items():
        test_qids.extend(qids[: take[key]])
    test = set(test_qids)
    train = [p["qid"] for p in probes if p["qid"] not in test]

    def profile(qids: list[str]) -> dict:
        by_lang = collections.Counter(p["lang_tag"] for p in probes if p["qid"] in set(qids))
        by_band = collections.Counter(band_of.get(q, "untriaged") for q in qids)
        by_dom = collections.Counter(domain_of.get(q, "unknown") for q in qids)
        return {"n": len(qids), "lang": dict(by_lang), "band": dict(by_band),
                "domain": dict(by_dom.most_common(8))}

    payload = {
        "seed": args.seed,
        "n_test": len(test_qids),
        "n_train": len(train),
        "test_qids": sorted(test_qids),
        "train_qids": sorted(train),
        "test_profile": profile(sorted(test_qids)),
        "train_profile": profile(sorted(train)),
        "note": (
            "test_qids are frozen: never trained on, scored once per configuration. "
            "train_qids may be used as real question->provision training pairs alongside "
            "the synthetic ones. Design decisions made before 2026-09-05 were taken while "
            "looking at all 552 questions, so the test set is unseen by the model but not "
            "by the authors; the paper must say so."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"test  {len(test_qids)} questions")
    for k, v in payload["test_profile"].items():
        if k != "n":
            print(f"    {k}: {v}")
    print(f"train {len(train)} questions")
    for k, v in payload["train_profile"].items():
        if k != "n":
            print(f"    {k}: {v}")
    print(f"\nwrote {args.out}")
    print("\nNext: build verification sheets for the test set only —")
    print("  python scripts/25_make_verification_sheets.py --annotators A B C D \\")
    print("      --qids data/processed/test_split_v1.json --n-single 160 --n-double 40")


if __name__ == "__main__":
    main()
