"""Train the Topic-3 LSTM language model on formal legal Bangla.

    python scripts/39_train_language_model.py

Trained on data/processed/corpus_v1.jsonl's text_bn (~3.85M tokens over
39,484 chunks) -- the full corpus, no subsetting needed at this scale on CPU.
Writes models/language_model.pt. scripts/40_measure_perplexity.py then uses
it to measure the register gap independently of any retriever.
"""

from __future__ import annotations

import pathlib
import sys

import torch

torch.set_num_threads(8)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.models.language_model import save, train  # noqa: E402
from dhara.vocab import Vocab  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
VOCAB = pathlib.Path("data/processed/vocab.json")
MATRIX = pathlib.Path("models/embedding_matrix.pt")
OUT = pathlib.Path("models/language_model.pt")

HIDDEN = 256
SEQ_LEN = 64
BATCH_SIZE = 64
EPOCHS = 2
LR = 1e-3
SEED = 42


def main() -> None:
    # Stream rather than materialize the full Chunk list -- only text_bn is
    # needed here, and this machine runs with limited RAM headroom alongside
    # normal desktop use (DECISIONS.md 2026-09-15/16).
    texts = [c.text_bn for c in schema.read_jsonl(CORPUS)]
    print(f"{len(texts)} chunks")

    vocab = Vocab.load(VOCAB)
    embedding_matrix = torch.load(MATRIX, map_location="cpu", weights_only=False)
    print(f"vocab {len(vocab):,} types, embedding matrix {tuple(embedding_matrix.shape)}")

    model, history = train(
        texts, vocab, embedding_matrix,
        hidden=HIDDEN, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, seed=SEED,
    )

    config = {
        "hidden": HIDDEN, "num_layers": 2, "dim": int(embedding_matrix.shape[1]),
        "vocab_size": len(vocab), "vocab_path": str(VOCAB), "embedding_matrix_path": str(MATRIX),
        "corpus_file": str(CORPUS), "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr": LR, "seed": SEED, "history": history,
    }
    save(model, OUT, config)
    print(f"wrote {OUT}")

    print("\nsample generations (temperature 1.0):")
    for i in range(3):
        print(f"  {model.generate(vocab, max_len=30, seed=100+i)}")


if __name__ == "__main__":
    main()
