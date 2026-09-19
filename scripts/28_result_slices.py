"""The ladder table and the slices the proposal calls the proof of the thesis.

    python scripts/28_result_slices.py
    python scripts/28_result_slices.py --runs bm25_tuned bge_m3_zeroshot bge_m3_finetuned_anchor_v3

Writes results/tables/ladder.csv, ladder_by_register.csv, ladder_by_domain.csv
and results/runs/result_slices.json.

## Why this exists

`docs/Dhara_Proposal.md` §9.3 names four result slices. Only the first was being
produced. The second is the one the proposal describes as "the proof of your
thesis":

    Colloquial vs. formal breakdown. The same table split by query register.
    Prediction: BM25's gap between the two is large; the fine-tuned bi-encoder's
    gap is small. This table shows the lexical gap exists and that your method
    specifically closes it.

Nothing was computing it, because `register` lives in the gold set and never
travelled into the run JSONs -- those carry `lang_tag`, which is a property of
the *corpus* (is the gold provision written in Bangla or English), not of the
*question*. The two are easy to confuse and answer different questions:

  - `lang_tag` answers "can a retriever cross the language barrier".
  - `register` answers "can a retriever cross the vocabulary gap between how a
    citizen writes and how a statute is written" -- which is this project's
    actual claim.

This script joins on `qid` and reports both, plus the per-domain slice, over any
set of runs. It reads only `results/runs/*.json`, so every number it prints was
emitted by a scoring script rather than typed by hand.

## A caveat that must travel with the register table

The gold set is 729 colloquial to 65 formal questions. A 65-question slice has a
95% confidence interval of roughly +/-12 points on a recall figure, so the formal
column will rarely separate two adjacent rungs on its own. The comparison that
carries weight is the *gap* between the columns within one rung, and how that gap
narrows as the ladder climbs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

GOLD = pathlib.Path("data/processed/gold_test_v1.jsonl")
RUNS = pathlib.Path("results/runs")
TABLES = pathlib.Path("results/tables")

# The ladder in the order the report presents it: lexical floor, untrained
# dense, the two untrained document/reranking changes, then the trained arm.
DEFAULT_RUNS = [
    "bm25_tuned",
    "bge_m3_zeroshot",
    "bge_m3_acttitle",
    "bge_m3_acttitle_rerank",
    "bge_m3_finetuned_anchor_v3",
]
CUTOFFS = (1, 5, 10)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def recall(ranks: list, k: int) -> float:
    if not ranks:
        return float("nan")
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list, cutoff: int = 10) -> float:
    if not ranks:
        return float("nan")
    return sum(1.0 / r for r in ranks if r is not None and r <= cutoff) / len(ranks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--gold", type=pathlib.Path, default=GOLD)
    ap.add_argument("--tag", default="",
                    help="suffix appended to every output filename, e.g. '_v2', "
                         "so a rescore never overwrites the frozen v1 tables")
    ap.add_argument("--domains", nargs="+", default=None,
                    help="explicit domain list for the by-domain slice, in report "
                         "order (default: top-8 by question count, which can bury "
                         "a thin-but-confirmed domain like cybercrime)")
    args = ap.parse_args()

    gold = read_jsonl(args.gold)
    register_of: dict[str, str] = {}
    domain_of: dict[str, str] = {}
    for row in gold:                      # one row per qid; duplicates are identical
        register_of.setdefault(row["qid"], (row.get("register") or "unknown").strip())
        domain_of.setdefault(row["qid"], (row.get("domain") or "unknown").strip())

    loaded: dict[str, dict] = {}
    for run_id in args.runs:
        path = RUNS / f"{run_id}.json"
        if not path.exists():
            print(f"  skipping {run_id}: {path} not found")
            continue
        loaded[run_id] = json.loads(path.read_text(encoding="utf-8"))
    assert loaded, "no runs found -- nothing to tabulate"

    # A ladder is a paired comparison.  Matching aggregate `n` values is not
    # enough: the same qids must occur in the same order, otherwise its rows
    # can look comparable while describing different evaluation populations.
    reference_id, reference_run = next(iter(loaded.items()))
    reference_qids = [row["qid"] for row in reference_run["per_query"]]
    for run_id, run in loaded.items():
        qids = [row["qid"] for row in run["per_query"]]
        if qids != reference_qids:
            raise ValueError(
                f"{run_id} does not use the same ordered probe qids as "
                f"{reference_id}; build separate slice tables for it."
            )

    report: dict = {"run_id": f"result_slices{args.tag}", "runs": list(loaded), "slices": {}}

    # ---------------------------------------------------------------- ladder
    print("\n=== LADDER (all questions) ===")
    header = f"  {'run':32s} {'n':>4s} " + " ".join(f"{'R@'+str(k):>7s}" for k in CUTOFFS) + f" {'MRR@10':>7s}"
    print(header)
    ladder_rows = []
    for run_id, run in loaded.items():
        ranks = [r["rank"] for r in run["per_query"]]
        row = {"run": run_id, "n": len(ranks),
               **{f"R@{k}": round(recall(ranks, k), 4) for k in CUTOFFS},
               "MRR@10": round(mrr(ranks), 4)}
        ladder_rows.append(row)
        print(f"  {run_id:32s} {row['n']:4d} " +
              " ".join(f"{row[f'R@{k}']:7.3f}" for k in CUTOFFS) + f" {row['MRR@10']:7.3f}")
    report["slices"]["ladder"] = ladder_rows

    # -------------------------------------------------------------- register
    # The thesis table. Reported as two columns plus the gap between them,
    # because the gap is the quantity the claim is about.
    print("\n=== BY REGISTER — colloquial vs formal (§9.3, the thesis table) ===")
    counts = Counter(register_of.get(r["qid"], "unknown")
                     for r in next(iter(loaded.values()))["per_query"])
    print(f"  question counts: {dict(counts)}")
    print(f"  {'run':32s} {'colloq R@5':>11s} {'formal R@5':>11s} {'gap':>8s}   "
          f"{'colloq R@10':>12s} {'formal R@10':>12s} {'gap':>8s}")
    register_rows = []
    for run_id, run in loaded.items():
        by_reg: dict[str, list] = defaultdict(list)
        for r in run["per_query"]:
            by_reg[register_of.get(r["qid"], "unknown")].append(r["rank"])
        row = {"run": run_id}
        for reg in ("colloquial", "formal"):
            for k in CUTOFFS:
                row[f"{reg}_R@{k}"] = round(recall(by_reg.get(reg, []), k), 4)
            row[f"{reg}_n"] = len(by_reg.get(reg, []))
        for k in CUTOFFS:
            row[f"gap_R@{k}"] = round(row[f"formal_R@{k}"] - row[f"colloquial_R@{k}"], 4)
        register_rows.append(row)
        print(f"  {run_id:32s} {row['colloquial_R@5']:11.3f} {row['formal_R@5']:11.3f} "
              f"{row['gap_R@5']:+8.3f}   {row['colloquial_R@10']:12.3f} "
              f"{row['formal_R@10']:12.3f} {row['gap_R@10']:+8.3f}")
    report["slices"]["by_register"] = register_rows
    print("\n  A positive gap means formal questions are easier than colloquial ones.")
    print("  The claim is that the gap is large for BM25 and shrinks as the ladder climbs.")
    print(f"  Formal n={register_rows[0].get('formal_n', 0)}: a ~±12 point interval, so read")
    print("  the gap's direction and trend, not small differences between adjacent rungs.")

    # ---------------------------------------------------------------- domain
    print("\n=== BY DOMAIN (R@10) ===")
    if args.domains:
        domains = args.domains
    else:
        domains = [d for d, _ in Counter(
            domain_of.get(r["qid"], "unknown")
            for r in next(iter(loaded.values()))["per_query"]).most_common(8)]
    print(f"  {'run':32s} " + " ".join(f"{d[:11]:>12s}" for d in domains))
    domain_rows = []
    for run_id, run in loaded.items():
        by_dom: dict[str, list] = defaultdict(list)
        for r in run["per_query"]:
            by_dom[domain_of.get(r["qid"], "unknown")].append(r["rank"])
        row = {"run": run_id}
        for d in domains:
            row[d] = round(recall(by_dom.get(d, []), 10), 4)
            row[f"{d}_n"] = len(by_dom.get(d, []))
        domain_rows.append(row)
        print(f"  {run_id:32s} " + " ".join(f"{row[d]:12.3f}" for d in domains))
    report["slices"]["by_domain"] = domain_rows
    print(f"  n per domain: " + ", ".join(f"{d}={domain_rows[0][f'{d}_n']}" for d in domains))

    TABLES.mkdir(parents=True, exist_ok=True)
    for name, rows in (("ladder", ladder_rows),
                       ("ladder_by_register", register_rows),
                       ("ladder_by_domain", domain_rows)):
        dest = TABLES / f"{name}{args.tag}.csv"
        with dest.open("w", newline="", encoding="utf-8") as fh:
            fields = list(rows[0])
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {dest}")

    out = RUNS / f"result_slices{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
