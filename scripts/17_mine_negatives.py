"""Mine hard negatives for contrastive fine-tuning, from the cached dense index.

    python scripts/17_mine_negatives.py --variant strict
    python scripts/17_mine_negatives.py --variant strict --limit 20   # smoke test

Writes data/processed/train_{variant}_v1_negatives.jsonl and
results/tables/negatives_stats.csv.

## Ranks 5-30, and why the top four are poison

The obvious hard negative is the highest-scoring wrong answer. It is also, very
often, a right answer nobody labelled: this corpus has ~10 near-identical
local-government Acts and several overlapping family-law statutes, so the
provision ranked #2 for "my husband divorced me" is frequently a genuine second
correct answer that the topic inventory simply did not anchor. Training against
it teaches the model that a correct answer is wrong, which is worse than having
no hard negatives at all.

So candidates come from ranks 5-30 (CLAUDE.md), and three further exclusions are
applied on top:

  - the pair's own positive provision;
  - `also_relevant_provision_ids` — every other provision the same topic anchors
    or signature-matches, which is exactly the relevant-but-unlabelled set;
  - repealed and omitted provisions, which should never be a training target in
    either direction.

## Why the dense index and not BM25

Negatives should be hard *for the model being trained*. BM25's mistakes are not
BGE-m3's mistakes — on the English-gold slice BM25 retrieves nothing relevant at
all, so its top-30 would be near-random text rather than a hard negative. The
negatives therefore come from the zero-shot BGE-m3 index, which is the model
fine-tuning starts from.

Questions are encoded on CPU here (413 distinct questions, a few minutes). The
corpus side is the cached index; nothing is re-embedded.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.synth import is_repealed  # noqa: E402

INDEX = pathlib.Path("models/index_bge_m3_zeroshot_v1")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
STATS = pathlib.Path("results/tables/negatives_stats.csv")

RANK_LO, RANK_HI = 5, 30
N_NEGATIVES = 8
DEPTH = 60  # enough headroom that exclusions cannot empty the 5-30 window


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def encode_questions(questions: list[str], manifest: dict, cache: pathlib.Path) -> np.ndarray:
    """Encode with the same checkpoint and settings that built the corpus index.

    Reusing the manifest rather than re-specifying the checkpoint is deliberate:
    a query embedded by a different model, or with a different prefix, than the
    passages is a silent and total failure, and it looks like a bad result rather
    than a bug.
    """
    # Cache validity is checked on the CONTENT of the question list, not its
    # length. Regenerating the training data with different templates can leave
    # the count identical while every string changed -- a length-only check then
    # silently mines negatives for question A using question B's embedding, and
    # nothing downstream would notice. Nearly shipped exactly that on
    # 2026-09-04, hence the fingerprint.
    fingerprint = hashlib.sha1("\n".join(questions).encode("utf-8")).hexdigest()
    stamp = cache.with_suffix(".fingerprint")
    if cache.exists() and stamp.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(questions) and stamp.read_text(encoding="utf-8").strip() == fingerprint:
            print(f"  reusing cached query embeddings {cached.shape} from {cache}")
            return cached.astype(np.float32)
        print(f"  cache {cache} is stale (count or content changed) — re-encoding")
    elif cache.exists():
        print(f"  cache {cache} has no fingerprint — re-encoding to be safe")

    from sentence_transformers import SentenceTransformer

    print(f"  encoding {len(questions)} questions with {manifest['checkpoint']} (CPU)")
    model = SentenceTransformer(manifest["checkpoint"])
    model.max_seq_length = manifest["max_seq_length"]
    prefix = manifest.get("query_prefix", "")
    vectors = model.encode(
        [prefix + q for q in questions],
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    stamp.write_text(fingerprint, encoding="utf-8")
    print(f"  cached to {cache}")
    return vectors.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["strict", "full", "anchor"], default="strict")
    ap.add_argument("--index", type=pathlib.Path, default=INDEX)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--n-negatives", type=int, default=N_NEGATIVES)
    ap.add_argument("--limit", type=int, default=0, help="only process N distinct questions")
    args = ap.parse_args()

    train_path = pathlib.Path(f"data/processed/train_{args.variant}_v1.jsonl")
    rows = read_jsonl(train_path)
    print(f"{train_path}: {len(rows)} pairs")

    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    corpus_emb = np.load(args.index / "embeddings.npy").astype(np.float32)
    chunk_ids = json.loads((args.index / "chunk_ids.json").read_text(encoding="utf-8"))
    index_of = {cid: i for i, cid in enumerate(chunk_ids)}

    provision_of, repealed = {}, set()
    for chunk in schema.read_jsonl(args.corpus):
        provision_of[chunk.chunk_id] = chunk.provision_id
        if is_repealed(chunk):
            repealed.add(chunk.chunk_id)
    provisions = np.array([provision_of[c] for c in chunk_ids])
    repealed_mask = np.array([c in repealed for c in chunk_ids])
    print(f"corpus: {len(chunk_ids)} indexed chunks, {repealed_mask.sum()} repealed (excluded as negatives)")

    questions = sorted({r["question_bn"] for r in rows})
    if args.limit:
        questions = questions[: args.limit]
        rows = [r for r in rows if r["question_bn"] in set(questions)]
    q_index = {q: i for i, q in enumerate(questions)}

    cache = args.index / f"train_{args.variant}_query.npy"
    query_emb = encode_questions(questions, manifest, cache)

    # Top-DEPTH once per distinct question, reused by every pair that shares it.
    print(f"  ranking {len(questions)} questions against {len(chunk_ids)} chunks")
    ranked: dict[str, np.ndarray] = {}
    for question, i in q_index.items():
        scores = corpus_emb @ query_emb[i]
        scores[repealed_mask] = -np.inf
        top = np.argpartition(scores, -DEPTH)[-DEPTH:]
        ranked[question] = top[np.argsort(scores[top])[::-1]]

    # GLOBAL exclusion set: every provision that is *anyone's* positive in this
    # training file, not just the current pair's own topic.
    #
    # The first version of this script excluded only `gold_provision_ids` plus
    # the current row's `also_relevant_provision_ids` — the relevant set for
    # THAT topic. It never checked whether a candidate was a different topic's
    # actual answer. Measured after the fact: 7.7% of mined negative slots
    # (2,335 of 30,162) were literally another pair's positive — e.g. a
    # provision anchored to `inheritance_share` mined as a negative for
    # `will_probate`, both pulling from the Succession Act. Training on that
    # teaches the model to push a correct answer away from a question that
    # correctly matches it, and because MultipleNegativesRankingLoss compares
    # every anchor in a batch against every other item's embedding, one
    # contaminated pair corrupts the gradient for every other pair sharing that
    # batch — this is the primary suspect for the embedding-space collapse
    # documented in DECISIONS.md 2026-09-03 (fine-tuned corpus embeddings ended
    # up at mean cosine 0.16 against their zero-shot counterparts: not "shifted
    # by training", but nearly unrelated).
    all_positives = {row["gold_provision_ids"][0] for row in rows}
    print(f"  global exclusion set: {len(all_positives)} provisions that are a positive somewhere in this file")

    out = pathlib.Path(f"data/processed/train_{args.variant}_v1_negatives.jsonl")
    counters: collections.Counter = collections.Counter()
    n_negatives_hist: list[int] = []

    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            excluded = (
                set(row["gold_provision_ids"])
                | set(row.get("also_relevant_provision_ids") or [])
                | all_positives
            )
            order = ranked[row["question_bn"]]

            # Rank positions are 1-based and counted over the unfiltered list, so
            # "ranks 5-30" means what it says regardless of how many candidates
            # the exclusions remove.
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
            record["negative_chunk_ids"] = negatives
            record["negative_source"] = f"bge_m3_zeroshot ranks {RANK_LO}-{RANK_HI}"
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            counters["pairs"] += 1
            counters["negatives"] += len(negatives)

    mean_negs = sum(n_negatives_hist) / max(len(n_negatives_hist), 1)
    print(f"\nwrote {out}")
    print(f"  pairs                     : {counters['pairs']}")
    print(f"  negatives                 : {counters['negatives']} (mean {mean_negs:.2f} per pair)")
    print(f"  skipped relevant-unlabelled: {counters['skipped_relevant_unlabelled']}")
    print(f"  pairs with zero negatives : {counters['pairs_with_no_negatives']}")

    STATS.parent.mkdir(parents=True, exist_ok=True)
    write_header = not STATS.exists()
    with STATS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["variant", "pairs", "negatives", "mean_per_pair",
                             "skipped_relevant_unlabelled", "pairs_with_zero", "rank_window"])
        writer.writerow([args.variant, counters["pairs"], counters["negatives"], round(mean_negs, 3),
                         counters["skipped_relevant_unlabelled"], counters["pairs_with_no_negatives"],
                         f"{RANK_LO}-{RANK_HI}"])
    print(f"wrote {STATS}")


if __name__ == "__main__":
    main()
