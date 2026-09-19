"""Topic 0 / Rung 2 — mean-pooled self-trained Word2Vec, cosine similarity.

The centre-word table from src/dhara/word2vec.py (models/word2vec_legal.kv) is
the embedding; this file only adds the retrieval wrapper the proposal's §8.2
calls for: mean-pool a document's (or query's) token vectors, compare by
cosine. No training happens here.

Tokenization is `aggressive()`, matching how the vectors were trained
(scripts/10_train_word2vec.py uses `aggressive(text).split()`, never
`content_tokens()` -- stopwords and the danda are meaningful context-window
signal for skip-gram, not free term matches to strip). Using `light()` here
instead would silently query the model with tokens it never learned, which is
CLAUDE.md's normalization-level rule in reverse: Word2Vec gets aggressive, not
light.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from ..normalize import aggressive
from ..schema import Chunk
from ..vocab import Vocab
from .base import Retriever

DEFAULT_KV = pathlib.Path("models/word2vec_legal.kv")
DEFAULT_VOCAB = pathlib.Path("data/processed/vocab.json")


class Word2VecRetriever(Retriever):
    name = "word2vec"

    def __init__(
        self,
        kv_path: pathlib.Path = DEFAULT_KV,
        vocab_path: pathlib.Path = DEFAULT_VOCAB,
        use_title: bool = True,
    ):
        self.use_title = use_title
        self.vocab = Vocab.load(vocab_path)
        state = torch.load(kv_path, map_location="cpu", weights_only=False)
        centre = state["state_dict"]["centre.weight"].detach().numpy().astype(np.float32)
        centre[self.vocab.pad_id] = 0.0
        self.centre = centre
        self.dim = centre.shape[1]
        self.skip_ids = {self.vocab.pad_id, self.vocab.unk_id}

        self.ids: list[str] = []
        self.embeddings: np.ndarray | None = None

    def _vector(self, text: str) -> np.ndarray:
        tokens = aggressive(text).split()
        ids = [self.vocab.stoi[t] for t in tokens if t in self.vocab.stoi]
        ids = [i for i in ids if i not in self.skip_ids]
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        vec = self.centre[ids].mean(axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    def _document(self, chunk: Chunk) -> str:
        parts = []
        if self.use_title and chunk.provision_title_bn:
            parts.append(chunk.provision_title_bn)
        parts.append(chunk.text_bn)
        return " ".join(parts)

    def index(self, chunks: list[Chunk]) -> None:
        self.ids = [c.chunk_id for c in chunks]
        self.embeddings = np.stack(
            [self._vector(self._document(c)) for c in chunks]
        ).astype(np.float32)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self.embeddings is None:
            raise RuntimeError("index() must be called before search()")
        vector = self._vector(query)
        scores = self.embeddings @ vector
        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(self.ids[i], float(scores[i])) for i in top]
