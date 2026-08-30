"""Topic 5 — dense retrieval with a pretrained multilingual sentence encoder.

A **bi-encoder** embeds the question and each provision separately and compares
the two vectors by cosine similarity. Because provisions are embedded once and
cached, a query costs one forward pass plus a matrix multiply — brute force over
47,000 vectors is milliseconds, so no ANN index is needed at this scale and
adding one would be complexity with no measurable benefit.

Why *multilingual* rather than a Bangla-only model: 44% of this corpus is English,
including the acts that govern inheritance, land registration and arrest, and 60%
of the gold answers live in those English-only Acts. A Bangla-only encoder cannot
represent those provisions at all, so a Bangla question about inheritance could
never reach the Succession Act. This is the single constraint that decides the
checkpoint.

*Which* multilingual encoder is not a free choice either. multilingual-e5-base
reached 8% recall@10 on the English-gold slice — close enough to BM25's 0% that
the cross-language wall looked unbridgeable without translating the corpus.
BAAI/bge-m3 gets 22% on the same slice of the full 39,484-chunk corpus (0.32
overall), so the wall is bridgeable and the default moved on 2026-08-28. BGE-m3
is 568M params / 1024-dim against e5-base's 278M / 768, which is why the corpus
embed wants a T4.

**The prefix footgun.** Checkpoints in the E5 family were trained with
`"query: "` before questions and `"passage: "` before documents. Omitting them
degrades performance substantially and silently. They must be applied at index
time and query time, and in *both* the zero-shot and fine-tuned runs — otherwise
the comparison is between a crippled baseline and a working model, which is not a
comparison. The prefixes are chosen from the checkpoint name, not hand-set, so
they cannot drift apart between the two runs. BGE-m3 dense retrieval takes no
prefix at all; bge-v1.5 (non-m3) takes one on the query side only. Same function,
same guarantee.
"""

from __future__ import annotations

import pathlib

import numpy as np

from ..normalize import light
from ..schema import Chunk
from .base import Retriever


def prefixes_for(checkpoint: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) implied by the checkpoint name."""
    name = checkpoint.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge" in name and "m3" not in name:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


class BiEncoderRetriever(Retriever):
    name = "biencoder"

    def __init__(
        self,
        checkpoint: str = "BAAI/bge-m3",
        max_seq_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
        use_title: bool = True,
    ):
        self.checkpoint = checkpoint
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self.use_title = use_title
        self.query_prefix, self.passage_prefix = prefixes_for(checkpoint)
        self.ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._model = None

    @property
    def model(self):
        # Loaded once, lazily. Reloading a transformer per request is the single
        # most common way to make a demo feel broken.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.checkpoint, device=self.device)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def _document(self, chunk: Chunk) -> str:
        # Light normalization only. Aggressive normalization would move the input
        # off the distribution this tokenizer was trained on.
        parts = []
        if self.use_title and chunk.provision_title_bn:
            parts.append(chunk.provision_title_bn)
        parts.append(chunk.text_bn)
        return self.passage_prefix + light(" ".join(parts))

    def index(self, chunks: list[Chunk]) -> None:
        self.ids = [c.chunk_id for c in chunks]
        texts = [self._document(c) for c in chunks]
        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self.embeddings is None:
            raise RuntimeError("index() must be called before search()")
        vector = self.model.encode(
            [self.query_prefix + light(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        scores = self.embeddings @ vector
        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, directory: pathlib.Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "chunk_ids.json").write_text(
            __import__("json").dumps(self.ids), encoding="utf-8"
        )

    def load(self, directory: pathlib.Path) -> None:
        self.embeddings = np.load(directory / "embeddings.npy")
        self.ids = __import__("json").loads(
            (directory / "chunk_ids.json").read_text(encoding="utf-8")
        )
