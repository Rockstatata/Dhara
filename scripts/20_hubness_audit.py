"""Diagnose retrieval hubness and query-space concentration between two indexes.

    python scripts/20_hubness_audit.py \
        --a models/index_bge_m3_finetuned_v1 --a-id bge_m3_finetuned_strict_v2 \
        --b models/index_bge_m3_zeroshot_v1  --b-id bge_m3_zeroshot

Writes results/runs/hubness_<A>_vs_<B>.json and
results/tables/hubness_<A>_vs_<B>.csv.

## What this measures, and why it earned its own script

Fine-tune attempt 2 produced a result that recall alone could not explain:
overall recall flat, English R@1/R@5 significantly *worse*, mixed-language
R@50/R@100 significantly *better*. No collapse — the encoder was healthy by
every check the notebook ran (corpus cosine 0.83 against zero-shot).

The cause was in the query distribution, not the document one. Fine-tuning on
413 template-generated questions — which share sentence frames, length and
narrative shape by construction — taught the encoder a narrow "citizen legal
question" direction. Every probe question drifted toward it: mean cosine to the
query centroid rose 0.699 -> 0.822. Whichever documents happened to sit near
that direction then became nearest neighbour to a disproportionate share of
questions, displacing the genuinely correct provision at rank 1. Two chunks
took 43 and 44 of 333 English top-1 slots; distinct top-1 documents fell from
168 to 144.

This is textbook **hubness**, the well-known high-dimensional nearest-neighbour
pathology, induced here by contrastive fine-tuning on a homogeneous query set.
It is invisible to aggregate recall — deep recall can even improve, because the
hub region is broadly topically correct — which is exactly why it needs its own
instrument rather than being inferred from a recall table.

Critically, it is *not* memorisation: the worst hub in attempt 2
(`family_1444_s5`) appeared **zero** times as a training positive, having been
dropped by the strict variant for being a gold provision. Training frequency
barely predicted which documents became hubs (mean top-1 gain 1.51 for training
positives vs 1.08 for everything else). Anyone re-reading these numbers should
not reach for "it overfit the anchors" — the evidence rules that out.

## Reading the output

- `query_concentration` — mean cosine of each query to the query centroid.
  Rising sharply means the queries are bunching; that is the upstream cause.
- `distinct_top1` — how many different documents win rank 1 across the probe
  set. Falling means a few documents are absorbing the ranking.
- `top1_gini` — inequality of the top-1 distribution, 0 = perfectly spread.
- `hubs` — documents with the largest top-1 gain from B to A, with their
  cosine to the query centroid in each index and their training frequency, so
  the memorisation explanation can be checked rather than assumed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
TRAIN = pathlib.Path("data/processed/train_strict_v1_negatives.jsonl")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TOP_N_HUBS = 15


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_index(directory: pathlib.Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    corpus = np.load(directory / "embeddings.npy").astype(np.float32)
    query = np.load(directory / "probe_query.npy").astype(np.float32)
    ids = json.loads((directory / "chunk_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return corpus, query, ids, manifest


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def concentration(query: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cosine of each query to the query centroid, and the centroid."""
    normed = unit(query)
    centroid = normed.mean(0)
    centroid /= np.linalg.norm(centroid)
    return float((normed @ centroid).mean()), centroid


def gini(counts: list[int]) -> float:
    """Inequality of the top-1 distribution. 0 = every document wins equally."""
    if not counts:
        return 0.0
    values = sorted(counts)
    n = len(values)
    total = sum(values)
    if total == 0:
        return 0.0
    cumulative = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def top1_counts(corpus: np.ndarray, query: np.ndarray, ids: list[str]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for i in range(query.shape[0]):
        counts[ids[int(np.argmax(corpus @ query[i]))]] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=pathlib.Path, required=True, help="candidate index")
    ap.add_argument("--b", type=pathlib.Path, required=True, help="baseline index")
    ap.add_argument("--a-id", required=True)
    ap.add_argument("--b-id", required=True)
    ap.add_argument("--probes", type=pathlib.Path, default=PROBES)
    ap.add_argument("--train", type=pathlib.Path, default=TRAIN)
    args = ap.parse_args()

    probes = read_jsonl(args.probes)
    corpus_a, query_a, ids_a, manifest_a = load_index(args.a)
    corpus_b, query_b, ids_b, manifest_b = load_index(args.b)
    assert ids_a == ids_b, "indexes have different chunk ordering; cannot compare directly"
    assert query_a.shape[0] == query_b.shape[0] == len(probes), "probe count mismatch"

    train_freq: collections.Counter = collections.Counter()
    if args.train.exists():
        train_freq.update(r["gold_chunk_ids"][0] for r in read_jsonl(args.train))

    lang_of = {i: p["lang_tag"] for i, p in enumerate(probes)}

    report: dict = {
        "run_id": f"hubness_{args.a_id}_vs_{args.b_id}",
        "a": args.a_id, "b": args.b_id,
        "a_index": str(args.a), "b_index": str(args.b),
        "a_fine_tuned": manifest_a.get("fine_tuned"),
        "b_fine_tuned": manifest_b.get("fine_tuned"),
        "n_probes": len(probes),
        "groups": {},
    }

    rows_out = []
    for group in ("all", "english", "bengali", "mixed"):
        idx = [i for i in range(len(probes)) if group == "all" or lang_of[i] == group]
        if not idx:
            continue
        qa, qb = query_a[idx], query_b[idx]

        conc_a, cen_a = concentration(qa)
        conc_b, cen_b = concentration(qb)
        counts_a = top1_counts(corpus_a, qa, ids_a)
        counts_b = top1_counts(corpus_b, qb, ids_b)

        entry = {
            "n": len(idx),
            "query_concentration": {"a": round(conc_a, 4), "b": round(conc_b, 4),
                                    "delta": round(conc_a - conc_b, 4)},
            "distinct_top1": {"a": len(counts_a), "b": len(counts_b),
                              "delta": len(counts_a) - len(counts_b)},
            "max_top1_share": {"a": max(counts_a.values()), "b": max(counts_b.values())},
            "top1_gini": {"a": round(gini(list(counts_a.values())), 4),
                          "b": round(gini(list(counts_b.values())), 4)},
        }
        report["groups"][group] = entry

        print(f"--- {group} (n={len(idx)})")
        print(f"  query concentration : {conc_b:.4f} -> {conc_a:.4f}  ({conc_a - conc_b:+.4f})")
        print(f"  distinct top-1 docs : {len(counts_b)} -> {len(counts_a)}")
        print(f"  max top-1 by one doc: {max(counts_b.values())} -> {max(counts_a.values())}")
        print(f"  top-1 gini          : {gini(list(counts_b.values())):.4f} -> "
              f"{gini(list(counts_a.values())):.4f}")

        rows_out.append({
            "group": group, "n": len(idx),
            "concentration_b": round(conc_b, 4), "concentration_a": round(conc_a, 4),
            "distinct_top1_b": len(counts_b), "distinct_top1_a": len(counts_a),
            "max_top1_b": max(counts_b.values()), "max_top1_a": max(counts_a.values()),
            "gini_b": round(gini(list(counts_b.values())), 4),
            "gini_a": round(gini(list(counts_a.values())), 4),
        })

        if group == "all":
            corpus_a_n, corpus_b_n = unit(corpus_a), unit(corpus_b)
            sim_a, sim_b = corpus_a_n @ cen_a, corpus_b_n @ cen_b
            gains = sorted(
                ((counts_a[c] - counts_b.get(c, 0), c) for c in counts_a),
                reverse=True,
            )[:TOP_N_HUBS]
            hubs = []
            for gain, chunk_id in gains:
                j = ids_a.index(chunk_id)
                hubs.append({
                    "chunk_id": chunk_id,
                    "top1_a": counts_a[chunk_id], "top1_b": counts_b.get(chunk_id, 0),
                    "top1_gain": gain,
                    "cosine_to_query_centroid_a": round(float(sim_a[j]), 4),
                    "cosine_to_query_centroid_b": round(float(sim_b[j]), 4),
                    "train_pairs_as_positive": train_freq.get(chunk_id, 0),
                })
            report["hubs"] = hubs
            print("\n  top hubs (largest top-1 gain):")
            print("    chunk_id                       gain   A    B   cos_cen_A  train_pos")
            for h in hubs:
                print(f"    {h['chunk_id']:30s} {h['top1_gain']:+4d} {h['top1_a']:4d} "
                      f"{h['top1_b']:4d}   {h['cosine_to_query_centroid_a']:.4f}    "
                      f"{h['train_pairs_as_positive']}")

            in_train = [g for g, c in gains if train_freq.get(c, 0) > 0]
            not_train = [g for g, c in gains if train_freq.get(c, 0) == 0]
            report["hub_training_membership"] = {
                "mean_gain_in_training": round(sum(in_train) / max(len(in_train), 1), 3),
                "n_in_training": len(in_train),
                "mean_gain_not_in_training": round(sum(not_train) / max(len(not_train), 1), 3),
                "n_not_in_training": len(not_train),
                "note": "If these are comparable, hubness is NOT explained by memorising "
                        "training positives.",
            }
        print()

    out_run = pathlib.Path(f"results/runs/hubness_{args.a_id}_vs_{args.b_id}.json")
    out_run.parent.mkdir(parents=True, exist_ok=True)
    out_run.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_run}")

    out_table = pathlib.Path(f"results/tables/hubness_{args.a_id}_vs_{args.b_id}.csv")
    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {out_table}")


if __name__ == "__main__":
    main()
