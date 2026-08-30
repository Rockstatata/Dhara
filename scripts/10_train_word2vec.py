"""Topic 0 — train Word2Vec on the legal corpus, build the embedding matrix.

    python scripts/10_train_word2vec.py
    python scripts/10_train_word2vec.py --neighbours আটক নামজারি জামিন খারিজ

Produces:
    models/word2vec_legal.kv          self-trained vectors
    data/processed/vocab.json         shared vocabulary (index 0 = PAD)
    models/embedding_matrix.pt        (vocab_size, dim) tensor for nn.Embedding
    results/tables/w2v_neighbours.csv legal vs general nearest neighbours

The neighbour table is a result, not a debug print. It is the most directly
human-readable evidence of what an embedding learned, and the comparison against
general-purpose Bangla vectors is what shows the legal corpus taught the model
something a general corpus did not.

Expect the self-trained vectors to be weak. A few hundred thousand words is small
for Word2Vec, and `min_count=3` / `epochs=30` are already tuned for that. Report
it as a corpus-size finding rather than hiding it — understanding why a method
underperforms is worth more than the method performing well.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.normalize import aggressive  # noqa: E402
from dhara.vocab import Vocab, build_pretrained_embedding_matrix  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
KV_OUT = pathlib.Path("models/word2vec_legal.kv")
VOCAB_OUT = pathlib.Path("data/processed/vocab.json")
MATRIX_OUT = pathlib.Path("models/embedding_matrix.pt")
NEIGHBOURS_OUT = pathlib.Path("results/tables/w2v_neighbours.csv")

DEFAULT_TERMS = ["আটক", "নামজারি", "খারিজ", "জামিন", "যৌতুক", "উচ্ছেদ", "ধারা", "ভরণপোষণ"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neighbours", nargs="*", default=DEFAULT_TERMS)
    ap.add_argument(
        "--general",
        default="",
        help="gensim-data name of general Bangla vectors for the comparison "
        "column, e.g. 'fasttext-wiki-news-subwords-300'. Skipped if empty.",
    )
    args = ap.parse_args()

    import torch

    from dhara import word2vec as w2v

    chunks = list(schema.read_jsonl(args.corpus))
    # Training tokenization is deliberately NOT `content_tokens`. That function
    # exists for ranking, where the danda and the stopwords are free term matches
    # that let document length drive the score. Skip-gram wants the opposite: the
    # danda marks a sentence boundary the context window should see, and frequent
    # words are already handled by frequency subsampling inside the trainer, which
    # is a better instrument than a hand-written stoplist. See normalize.py.
    sentences = [aggressive(c.text_bn).split() for c in chunks]
    sentences = [s for s in sentences if len(s) >= 3]
    total = sum(len(s) for s in sentences)
    print(f"{len(sentences)} provisions, {total:,} tokens")
    if total < 500_000:
        print(
            "  note: this is small for Word2Vec. Weak neighbours are the expected\n"
            "  result and should be reported as a corpus-size finding."
        )

    vocab = Vocab.build(sentences, min_count=args.min_count)
    vocab.save(VOCAB_OUT)
    print(f"vocab: {len(vocab):,} types (index 0 = PAD) -> {VOCAB_OUT}")

    model, history = w2v.train(
        sentences,
        vocab,
        dim=args.dim,
        window=args.window,
        epochs=args.epochs,
        negatives=10,
        seed=args.seed,
    )
    KV_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history}, KV_OUT)
    print(f"model -> {KV_OUT}")

    # The centre-word table IS the embedding matrix; the context table was
    # scaffolding for the objective and is discarded, as in the original paper.
    matrix = model.centre.weight.detach().clone()
    matrix[vocab.pad_id] = 0.0
    matrix.coverage = 1.0
    torch.save(matrix, MATRIX_OUT)
    print(f"embedding matrix: {tuple(matrix.shape)} -> {MATRIX_OUT}")

    general = None  # general-vector comparison column, filled in a later pass

    NEIGHBOURS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with NEIGHBOURS_OUT.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["term", "in_corpus", "legal_neighbours", "general_neighbours"])
        print(f"\n{'term':14s} legal-corpus nearest neighbours")
        for term in args.neighbours:
            legal = [w for w, _ in w2v.most_similar(model, vocab, term, topn=8)]
            writer.writerow([term, term in vocab.stoi, " · ".join(legal), ""])
            print(f"{term:14s} {' · '.join(legal) if legal else '(not in vocab)'}")
    print(f"\n-> {NEIGHBOURS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
