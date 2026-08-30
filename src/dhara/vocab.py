"""Topic 0 — vocabulary and the pretrained embedding matrix.

Every RNN in this project starts here: a vocabulary built once from the corpus,
and a `(vocab_size, dim)` float tensor used to initialise `nn.Embedding`. Sharing
one vocabulary across the classifier, the tagger, the language model and the
seq2seq model is not tidiness — it means the same token id means the same thing
everywhere, so an embedding matrix trained in one place can be reused in another.

Index 0 is always PAD. That is load-bearing: `CrossEntropyLoss(ignore_index=0)`
in the tagger and the language model depends on it, so padding is never trained
on and never contributes to a loss.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from dataclasses import dataclass

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int]

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, tokens: list[str], add_bos_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_bos_eos:
            return [self.stoi[BOS]] + ids + [self.stoi[EOS]]
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids if 0 <= i < len(self.itos)]

    @classmethod
    def build(cls, token_lists: list[list[str]], min_count: int = 2) -> "Vocab":
        counts = Counter(t for tokens in token_lists for t in tokens)
        kept = sorted(
            (t for t, n in counts.items() if n >= min_count),
            key=lambda t: (-counts[t], t),
        )
        itos = SPECIALS + kept
        return cls(itos=itos, stoi={t: i for i, t in enumerate(itos)})

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "Vocab":
        itos = json.loads(path.read_text(encoding="utf-8"))["itos"]
        return cls(itos=itos, stoi={t: i for i, t in enumerate(itos)})


def build_pretrained_embedding_matrix(
    vocab: Vocab,
    vectors,
    dim: int,
    seed: int = 42,
):
    """Map a vocabulary onto pretrained vectors -> (vocab_size, dim) float tensor.

    `vectors` is anything with `__contains__` and `__getitem__` returning a vector
    — a gensim KeyedVectors, or a plain dict. Tokens the pretrained model never
    saw are drawn from a small normal rather than left at zero: a row of zeros
    makes every unseen token identical to PAD and to each other, which quietly
    collapses part of the input space.

    PAD stays exactly zero, on purpose.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    matrix = torch.normal(
        mean=0.0, std=0.1, size=(len(vocab), dim), generator=generator
    )
    matrix[vocab.pad_id] = 0.0

    hits = 0
    for token, index in vocab.stoi.items():
        if token in SPECIALS:
            continue
        if token in vectors:
            matrix[index] = torch.tensor(vectors[token], dtype=torch.float32)
            hits += 1
    matrix.coverage = hits / max(len(vocab) - len(SPECIALS), 1)  # type: ignore[attr-defined]
    return matrix
