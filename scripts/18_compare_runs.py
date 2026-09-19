"""Compare two runs with a paired bootstrap CI on the difference.

    python scripts/18_compare_runs.py --a results/runs/bge_m3_finetuned_strict.json \
                                      --b results/runs/bge_m3_zeroshot.json

Writes results/tables/compare_{A}_vs_{B}.csv and results/runs/compare_{A}_vs_{B}.json.

Every rung-to-rung claim in this project goes through here. At n=552 the 95% CI
on a recall difference is roughly +/-4 points, so a 2-point gain is not a gain,
and a bar chart narrated as if it were is the specific failure this script
exists to prevent.

The bootstrap is **paired**: both runs answered the same 552 questions in the
same order, so resampling questions rather than runs cancels the shared
query-difficulty variance. Unpaired, a 552-question comparison has almost no
power; paired, it can resolve differences of about four points. The script
refuses to run if the two `per_query` lists are not aligned on qid, because
silently comparing different question sets would produce a confident number
about nothing.

Slices are reported separately because the project's whole claim is that the
English-gold slice behaves differently from the Bengali-gold slice. An overall
number that hides a large gain on one and a loss on the other is worse than no
number.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.metrics import CUTOFFS, paired_bootstrap, recall_at_k  # noqa: E402

GROUPS = ("all", "english", "bengali", "mixed")


def load(path: pathlib.Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    if "per_query" not in run:
        raise SystemExit(
            f"{path} has no per_query. Significance testing needs it, and "
            "regenerating it later means re-running the whole experiment."
        )
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=pathlib.Path, required=True, help="the new/candidate run")
    ap.add_argument("--b", type=pathlib.Path, required=True, help="the baseline/control run")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_a, run_b = load(args.a), load(args.b)
    pq_a = {r["qid"]: r for r in run_a["per_query"]}
    pq_b = {r["qid"]: r for r in run_b["per_query"]}

    shared = [q for q in pq_a if q in pq_b]
    if len(shared) != len(pq_a) or len(shared) != len(pq_b):
        raise SystemExit(
            f"runs answer different question sets: {len(pq_a)} vs {len(pq_b)}, "
            f"{len(shared)} shared. A paired test on unaligned questions is meaningless."
        )
    # Fixed order, so the bootstrap is reproducible from the seed alone.
    shared.sort()
    print(f"A = {run_a['run_id']}  ({run_a.get('checkpoint')}, fine_tuned={run_a.get('fine_tuned')})")
    print(f"B = {run_b['run_id']}  ({run_b.get('checkpoint')}, fine_tuned={run_b.get('fine_tuned')})")
    print(f"paired over {len(shared)} questions, {args.resamples} resamples\n")

    results = []
    for group in GROUPS:
        qids = [q for q in shared if group == "all" or pq_a[q]["lang_tag"] == group]
        if not qids:
            continue
        ranks_a = [pq_a[q]["rank"] for q in qids]
        ranks_b = [pq_b[q]["rank"] for q in qids]
        print(f"--- {group}  (n={len(qids)})")
        for k in CUTOFFS:
            a = recall_at_k(ranks_a, k)
            b = recall_at_k(ranks_b, k)
            boot = paired_bootstrap(ranks_a, ranks_b, k=k, n_resamples=args.resamples, seed=args.seed)
            verdict = "significant" if boot["significant"] else "within noise"
            lo, hi = boot["ci95"]
            print(f"  R@{k:<3d} A={a:.3f}  B={b:.3f}  diff={boot['delta']:+.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
            results.append({
                "group": group, "n": len(qids), "k": k,
                "a": round(a, 4), "b": round(b, 4),
                "delta": boot["delta"], "ci_lo": lo, "ci_hi": hi,
                "significant": boot["significant"],
            })
        print()

    stem = f"{run_a['run_id']}_vs_{run_b['run_id']}"
    table = pathlib.Path(f"results/tables/compare_{stem}.csv")
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {table}")

    out = pathlib.Path(f"results/runs/compare_{stem}.json")
    out.write_text(json.dumps({
        "run_a": run_a["run_id"], "run_b": run_b["run_id"],
        "run_a_file": str(args.a), "run_b_file": str(args.b),
        "checkpoint_a": run_a.get("checkpoint"), "checkpoint_b": run_b.get("checkpoint"),
        "n_paired": len(shared), "resamples": args.resamples, "seed": args.seed,
        "comparisons": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    significant = [r for r in results if r["significant"]]
    print(f"\n{len(significant)} of {len(results)} comparisons are outside the noise floor.")
    if not significant:
        print("No difference survives a paired bootstrap. Report that plainly — a null "
              "result honestly stated is worth more here than a narrated bar chart.")


if __name__ == "__main__":
    main()
