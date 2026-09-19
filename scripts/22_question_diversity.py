"""Measure how varied a question set is — embedding-wise and structurally.

    python scripts/22_question_diversity.py --questions data/processed/train_strict_v1.jsonl
    python scripts/22_question_diversity.py --compare-to-probes

Writes results/tables/question_diversity.csv and results/runs/question_diversity.json.

## Why this exists

Fine-tune attempt 2 failed by a mechanism aggregate recall could not see: the
training questions were structurally homogeneous, the encoder learned their
shared shape as a direction, every probe question drifted onto that direction
(mean cosine to the query centroid 0.699 -> 0.822), and a handful of documents
sitting near it became nearest neighbour to everything. See DECISIONS.md
2026-09-03 and `scripts/20_hubness_audit.py`.

`20_hubness_audit.py` diagnoses that *after* a GPU fine-tune. This script
measures the upstream cause *before* one, on CPU, so a template rewrite can be
checked in minutes instead of after an hour of Colab.

## The two things measured, and why both

**Embedding concentration** — mean cosine of each question to the question
centroid. This is the same quantity that predicted attempt 2's failure, so it
is the primary number. A training set whose concentration is far above the real
probe questions' is teaching a narrower notion of "legal question" than the one
it will be tested on.

**Structural signatures** — length spread, distinct opening bigrams, distinct
closing trigrams, type-token ratio. Templates collapse along exactly these
axes: every question ending "কী করব?" shares a closing trigram no matter how
different its topic is. These are cheap, need no model, and localise *which*
axis is collapsed, which the single embedding number cannot.

The real mined questions are the reference on both. They are what the system is
actually evaluated against, so their diversity is the target to imitate — not a
maximum to exceed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.normalize import aggressive  # noqa: E402

PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
INDEX = pathlib.Path("models/index_bge_m3_zeroshot_v1")
OUT_TABLE = pathlib.Path("results/tables/question_diversity.csv")
OUT_RUN = pathlib.Path("results/runs/question_diversity.json")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def structural(questions: list[str]) -> dict:
    """Model-free shape statistics. Cheap, and they localise the collapsed axis."""
    toks = [aggressive(q).split() for q in questions]
    lengths = [len(t) for t in toks]

    opens = collections.Counter(" ".join(t[:2]) for t in toks if len(t) >= 2)
    closes = collections.Counter(" ".join(t[-3:]) for t in toks if len(t) >= 3)
    vocab = collections.Counter(w for t in toks for w in t)
    total_tokens = sum(lengths)

    def top_share(counter: collections.Counter, k: int = 5) -> float:
        if not counter:
            return 0.0
        return sum(n for _, n in counter.most_common(k)) / sum(counter.values())

    return {
        "n": len(questions),
        "len_median": statistics.median(lengths) if lengths else 0,
        "len_p10": sorted(lengths)[len(lengths) // 10] if lengths else 0,
        "len_p90": sorted(lengths)[min(len(lengths) * 9 // 10, len(lengths) - 1)] if lengths else 0,
        "len_stdev": round(statistics.pstdev(lengths), 2) if len(lengths) > 1 else 0.0,
        "distinct_open_bigrams": len(opens),
        "open_bigram_top5_share": round(top_share(opens), 4),
        "distinct_close_trigrams": len(closes),
        "close_trigram_top5_share": round(top_share(closes), 4),
        "type_token_ratio": round(len(vocab) / max(total_tokens, 1), 4),
        "most_common_closes": [f"{c} ({n})" for c, n in closes.most_common(5)],
    }


def concentration(vectors: np.ndarray) -> float:
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    centroid = normed.mean(0)
    centroid /= np.linalg.norm(centroid)
    return float((normed @ centroid).mean())


def mean_pairwise(vectors: np.ndarray, sample: int = 300, seed: int = 0) -> float:
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    if len(normed) > sample:
        rng = np.random.default_rng(seed)
        normed = normed[rng.choice(len(normed), sample, replace=False)]
    sim = normed @ normed.T
    iu = np.triu_indices(len(normed), k=1)
    return float(sim[iu].mean())


def encode(questions: list[str], manifest: dict, cache: pathlib.Path | None) -> np.ndarray:
    if cache and cache.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(questions):
            print(f"  reusing cached embeddings {cached.shape} from {cache}")
            return cached.astype(np.float32)

    from sentence_transformers import SentenceTransformer

    print(f"  encoding {len(questions)} questions with {manifest['checkpoint']} (CPU, slow)")
    model = SentenceTransformer(manifest["checkpoint"])
    model.max_seq_length = manifest["max_seq_length"]
    prefix = manifest.get("query_prefix", "")
    vectors = model.encode([prefix + q for q in questions], batch_size=8,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=True).astype(np.float16)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    return vectors.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=pathlib.Path, action="append", default=None,
                    help="JSONL with question_bn; repeatable")
    ap.add_argument("--label", action="append", default=None, help="label per --questions")
    ap.add_argument("--index", type=pathlib.Path, default=INDEX)
    ap.add_argument("--structural-only", action="store_true",
                    help="skip embeddings (no model load, instant)")
    args = ap.parse_args()

    sources = args.questions or [pathlib.Path("data/processed/train_strict_v1.jsonl")]
    labels = args.label or [p.stem for p in sources]
    assert len(labels) == len(sources), "--label count must match --questions count"

    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))

    datasets: dict[str, list[str]] = {}
    probe_rows = read_jsonl(PROBES)
    datasets["real_mined_probes"] = sorted({r["question_bn"] for r in probe_rows})
    for path, label in zip(sources, labels):
        datasets[label] = sorted({r["question_bn"] for r in read_jsonl(path)})

    report: dict = {"run_id": "question_diversity", "index": str(args.index), "sets": {}}
    table_rows = []

    for label, questions in datasets.items():
        entry = structural(questions)
        if not args.structural_only:
            cache = None
            if label == "real_mined_probes":
                cache = args.index / "probe_query.npy"
            vectors = encode(questions, manifest, cache)
            # The cached probe_query.npy is in probe-file order, not sorted order;
            # only reuse it when the counts line up and re-encode otherwise.
            if cache and vectors.shape[0] == len(probe_rows) and label == "real_mined_probes":
                questions_for_vectors = [r["question_bn"] for r in probe_rows]
                if questions_for_vectors != questions:
                    order = {q: i for i, q in enumerate(questions_for_vectors)}
                    idx = [order[q] for q in questions if q in order]
                    vectors = vectors[idx]
            entry["concentration"] = round(concentration(vectors), 4)
            entry["mean_pairwise_cosine"] = round(mean_pairwise(vectors), 4)
        report["sets"][label] = entry

        print(f"\n=== {label} (n={entry['n']}) ===")
        if "concentration" in entry:
            print(f"  concentration (mean cosine to centroid) : {entry['concentration']:.4f}")
            print(f"  mean pairwise cosine                    : {entry['mean_pairwise_cosine']:.4f}")
        print(f"  length words   median {entry['len_median']}  p10 {entry['len_p10']}  "
              f"p90 {entry['len_p90']}  sd {entry['len_stdev']}")
        print(f"  distinct opening bigrams : {entry['distinct_open_bigrams']:5d}  "
              f"(top-5 cover {entry['open_bigram_top5_share']:.1%})")
        print(f"  distinct closing trigrams: {entry['distinct_close_trigrams']:5d}  "
              f"(top-5 cover {entry['close_trigram_top5_share']:.1%})")
        print(f"  type-token ratio         : {entry['type_token_ratio']:.4f}")
        print(f"  most common endings      : {', '.join(entry['most_common_closes'][:3])}")

        row = {"set": label, **{k: v for k, v in entry.items() if k != "most_common_closes"}}
        table_rows.append(row)

    ref = report["sets"].get("real_mined_probes", {})
    print("\n=== versus the real mined questions (the distribution being tested on) ===")
    for label, entry in report["sets"].items():
        if label == "real_mined_probes":
            continue
        bits = []
        if "concentration" in entry and "concentration" in ref:
            d = entry["concentration"] - ref["concentration"]
            bits.append(f"concentration {d:+.4f}")
        bits.append(f"close-trigram top5 share {entry['close_trigram_top5_share'] - ref['close_trigram_top5_share']:+.1%}")
        bits.append(f"length sd {entry['len_stdev'] - ref['len_stdev']:+.1f}")
        print(f"  {label}: " + ", ".join(bits))
    print("\n  Closer to zero on every axis is the goal. A large positive concentration")
    print("  delta is the attempt-2 failure signature; a large positive close-trigram")
    print("  share means the templates all end the same way.")

    OUT_RUN.parent.mkdir(parents=True, exist_ok=True)
    OUT_RUN.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in table_rows for k in r})
        writer = csv.DictWriter(fh, fieldnames=["set"] + [f for f in fields if f != "set"])
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"\nwrote {OUT_RUN}")
    print(f"wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
