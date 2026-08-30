"""The Retriever contract. Frozen — every rung implements exactly this.

`evaluate.py` never learns about specific models, so adding a rung is one file
plus one config line. Do not add methods here to suit one retriever; if a rung
needs something extra, it belongs on that rung's own class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import Chunk


class Retriever(ABC):
    name: str

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build whatever this retriever needs to answer queries."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return [(chunk_id, score), ...] sorted descending, length <= k."""
