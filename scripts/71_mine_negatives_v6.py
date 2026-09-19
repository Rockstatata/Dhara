"""Mine hard negatives (ranks 5-30) for train_retrieval_v6.jsonl.

    python scripts/71_mine_negatives_v6.py

Same method as `67_mine_negatives_v5.py` (CLAUDE.md: ranks 5-30 of the
zero-shot dense index, never 1-4; global positive-provision exclusion
across the whole file). CPU only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

import numpy as np

import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dhara import schema  # noqa: E402
from dhara.synth import is_repealed  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN = pathlib.Path("data/processed/train_retrieval_v6.jsonl")
INDEX = pathlib.Path("models/index_bge_m3_zeroshot_v1")
OUT = pathlib.Path("data/processed/train_retrieval_v6_negatives.jsonl")
REPORT = pathlib.Path("results/runs/mine_negatives_v6.json")

RANK_LO, RANK_HI = 5, 30
N_NEGATIVES = 8
DEPTH = 60


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_questions(questions: list[str], manifest: dict, cache: pathlib.Path) -> np.ndarray:
    fingerprint = hashlib.sha1("\n".join(questions).encode("utf-8")).hexdigest()
    stamp = cache.with_suffix(".fingerprint")
    if cache.exists() and stamp.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(questions) and stamp.read_text(encoding="utf-8").strip() == fingerprint:
            print(f"  reusing cached query embeddings {cached.shape} from {cache}")
            return cached.astype(np.float32)
        print(f"  cache {cache} is stale — re-encoding")

    # Fast warm-start: reuse train_v5_query.npy for existing questions
    v5_cache = cache.parent / "train_v5_query.npy"
    v5_file = pathlib.Path("data/processed/train_retrieval_v5.jsonl")
    cached_vecs: dict[str, np.ndarray] = {}
    if v5_cache.exists() and v5_file.exists():
        v5_rows = read_jsonl(v5_file)
        v5_questions = sorted({r["question"] for r in v5_rows})
        v5_embs = np.load(v5_cache)
        if len(v5_questions) == len(v5_embs):
            print(f"  warm-starting with {len(v5_questions)} cached questions from v5")
            for q, vec in zip(v5_questions, v5_embs):
                cached_vecs[q] = vec

    missing = [q for q in questions if q not in cached_vecs]
    if missing:
        from sentence_transformers import SentenceTransformer

        print(f"  encoding {len(missing)} missing questions with {manifest['checkpoint']} (CPU)")
        model = SentenceTransformer(manifest["checkpoint"])
        model.max_seq_length = manifest["max_seq_length"]
        prefix = manifest.get("query_prefix", "")
        new_vecs = model.encode(
            [prefix + q for q in missing],
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float16)
        for q, vec in zip(missing, new_vecs):
            cached_vecs[q] = vec
    else:
        print("  all questions covered by cached embeddings!")

    dim = len(next(iter(cached_vecs.values())))
    vectors = np.empty((len(questions), dim), dtype=np.float16)
    for i, q in enumerate(questions):
        vectors[i] = cached_vecs[q]

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    stamp.write_text(fingerprint, encoding="utf-8")
    return vectors.astype(np.float32)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=pathlib.Path, default=TRAIN)
    ap.add_argument("--index", type=pathlib.Path, default=INDEX)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--n-negatives", type=int, default=N_NEGATIVES)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(args.train)
    print(f"{args.train}: {len(rows)} pairs")

    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    corpus_emb = np.load(args.index / "embeddings.npy").astype(np.float32)
    chunk_ids = json.loads((args.index / "chunk_ids.json").read_text(encoding="utf-8"))

    provision_of, repealed = {}, set()
    for chunk in schema.read_jsonl(args.corpus):
        provision_of[chunk.chunk_id] = chunk.provision_id
        if is_repealed(chunk):
            repealed.add(chunk.chunk_id)
    provisions = np.array([provision_of[c] for c in chunk_ids])
    repealed_mask = np.array([c in repealed for c in chunk_ids])
    print(f"corpus: {len(chunk_ids)} indexed chunks, {repealed_mask.sum()} repealed")

    for row in rows:
        if "provision_id" not in row or row["provision_id"] is None:
            row["provision_id"] = provision_of.get(row["positive_chunk_ids"][0])

    questions = sorted({r["question"] for r in rows})
    if args.limit:
        questions = questions[: args.limit]
        rows = [r for r in rows if r["question"] in set(questions)]
    q_index = {q: i for i, q in enumerate(questions)}

    cache = args.index / "train_v6_query.npy"
    query_emb = encode_questions(questions, manifest, cache)

    print(f"  ranking {len(questions)} questions against {len(chunk_ids)} chunks")
    ranked: dict[str, np.ndarray] = {}
    for question, i in q_index.items():
        scores = corpus_emb @ query_emb[i]
        scores[repealed_mask] = -np.inf
        top = np.argpartition(scores, -DEPTH)[-DEPTH:]
        ranked[question] = top[np.argsort(scores[top])[::-1]]

    all_positives = {row["provision_id"] for row in rows}
    print(f"  global exclusion set: {len(all_positives)} provisions that are a positive somewhere in this file")

    counters: collections.Counter = collections.Counter()
    n_negatives_hist: list[int] = []
    out_rows: list[dict] = []
    for row in rows:
        excluded = {row["provision_id"]} | all_positives
        order = ranked[row["question"]]
        negatives = []
        for position, idx in enumerate(order, 1):
            if position < RANK_LO:
                continue
            if position > RANK_HI:
                break
            if provisions[idx] in excluded:
                counters["skipped_relevant_unlabelled"] += 1
                continue
            negatives.append(chunk_ids[idx])
            if len(negatives) >= args.n_negatives:
                break

        n_negatives_hist.append(len(negatives))
        if not negatives:
            counters["pairs_with_no_negatives"] += 1
        record = dict(row)
        record["hard_negative_chunk_ids"] = negatives
        record["negative_source"] = f"bge_m3_zeroshot ranks {RANK_LO}-{RANK_HI}"
        record["question_bn"] = row["question"]
        record["gold_chunk_ids"] = row["positive_chunk_ids"]
        record["negative_chunk_ids"] = negatives
        out_rows.append(record)
        counters["pairs"] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for record in out_rows:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Also overwrite args.train directly so notebooks reading train_retrieval_v6.jsonl get negatives in-place
    if args.train.resolve() != OUT.resolve():
        with args.train.open("w", encoding="utf-8") as fh:
            for record in out_rows:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  also updated {args.train} in-place with hard negatives")

    hist = np.array(n_negatives_hist)
    report = {
        "pairs": int(counters["pairs"]),
        "pairs_with_full_8_negatives": int((hist == args.n_negatives).sum()),
        "pairs_with_no_negatives": int(counters["pairs_with_no_negatives"]),
        "mean_negatives_per_pair": float(hist.mean()) if len(hist) else 0.0,
        "skipped_relevant_unlabelled": int(counters["skipped_relevant_unlabelled"]),
        "distinct_questions_encoded": len(questions),
        "output": str(OUT),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
