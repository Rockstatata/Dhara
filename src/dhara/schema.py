"""The frozen chunk contract. Imported everywhere; never redefined locally.

The metadata *is* the citation. A chunk that cannot be cited is useless for this
application, so schema completeness is a correctness requirement rather than
bookkeeping — see CONTEXT.md, "Citation".
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Optional

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_BN_TO_ASCII = {ord(d): str(i) for i, d in enumerate(BN_DIGITS)}


def to_ascii_digits(text: str) -> str:
    """১২৩ -> 123. Load-bearing: provision numbers appear in both forms, in both
    queries and text, and a user writing "১০৩ ধারা" must reach the same chunk as
    one writing "section 103"."""
    return text.translate(_BN_TO_ASCII)


@dataclass
class Chunk:
    chunk_id: str            # "labour_2006_s103" or "labour_2006_s103_p1"
    provision_id: str        # "labour_2006_s103" — parent, for dedup to display
    act_id: str              # "labour_2006"
    act_title_bn: str
    act_title_en: str
    act_year: Optional[int]
    act_no: str
    chapter_bn: Optional[str]
    provision_kind: str      # "section" (ধারা) | "article" (অনুচ্ছেদ)
    provision_no_bn: str     # "১০৩" — as printed
    provision_no_ascii: str  # "103" — normalized, for lookup
    provision_title_bn: Optional[str]
    text_bn: str             # light-normalized body — what models consume
    text_raw: str            # untouched — the only thing ever shown to a reader
    domain: str
    risk_tier: str
    sub_idx: int             # 0 when the provision was not split
    n_sub: int               # 1 when the provision was not split
    source_url: str
    source_dataset: str      # "blad" | "bdlaws"
    crawl_date: str
    n_words: int
    language: str
    footnote_markers: list[str] = field(default_factory=list)

    def citation(self) -> str:
        kind = "অনুচ্ছেদ" if self.provision_kind == "article" else "ধারা"
        return f"{self.act_title_bn}, {kind} {self.provision_no_bn}"


def write_jsonl(chunks: Iterable[Chunk], dest: pathlib.Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(src: pathlib.Path) -> Iterator[Chunk]:
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Chunk(**json.loads(line))
