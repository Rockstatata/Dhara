"""Rung 1 — BM25 over aggressively-normalized provisions.

Not a toy. BM25 routinely beats a poorly-tuned dense model, and an untuned BM25
that loses to our fine-tuned encoder would prove nothing — so `k1` and `b` are
tuned on dev, never on gold.

Two index fields are supported because it is a real ablation and a results-table
row, not a preference: provision titles are dense with exactly the formal legal
terminology the register gap is about, so `title + text` is expected to help on
formal queries and help less on colloquial ones. That contrast is the point.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from ..normalize import content_tokens
from ..schema import Chunk
from .base import Retriever


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_title: bool = True):
        self.k1 = k1
        self.b = b
        self.use_title = use_title
        self.ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def _document(self, chunk: Chunk) -> str:
        if self.use_title and chunk.provision_title_bn:
            return f"{chunk.provision_title_bn} {chunk.text_bn}"
        return chunk.text_bn

    def index(self, chunks: list[Chunk]) -> None:
        self.ids = [c.chunk_id for c in chunks]
        tokenized = [content_tokens(self._document(c)) for c in chunks]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("index() must be called before search()")
        tokens = content_tokens(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argpartition(scores, -min(k, len(scores)))[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(self.ids[i], float(scores[i])) for i in top]
