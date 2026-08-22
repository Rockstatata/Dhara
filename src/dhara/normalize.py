"""Two normalization levels, and they are not interchangeable.

`light` feeds transformers: their tokenizers were trained on natural text, so
anything more than fixing encoding inconsistencies moves the input off the
distribution the model learned.

`aggressive` feeds BM25, Word2Vec and TF-IDF: these match on surface forms, so
every unnormalized variant is a missed match.

Applying `aggressive` to transformer input is a real and commonly-made mistake
that costs measurable performance. Do not add a third level, and do not write a
local variant — divergent normalization produces irreproducible numbers.
"""

from __future__ import annotations

import re
import unicodedata

ZWNJ, ZWJ = "‌", "‍"
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
DIGIT_MAP = {d: str(i) for i, d in enumerate(BN_DIGITS)}

# Bangla has two sentence terminators in live use: the standard danda U+0964 and
# U+09F7, which appears in older typesetting. bdlaws uses both, sometimes in the
# same Act, so they are unified before anything else looks at the text.
DANDA = "।"
DANDA_VARIANTS = "৷"

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")
_NON_TEXT = re.compile(r"[^ঀ-৿0-9a-zA-Z।\s]")


def light(text: str) -> str:
    """For transformer inputs. Fix encoding inconsistencies only."""
    text = unicodedata.normalize("NFC", text)          # rule 1
    text = text.replace(ZWNJ, "")                      # rule 2 — ZWNJ only
    text = text.replace(DANDA_VARIANTS, DANDA)         # rule 5
    text = text.replace("\xa0", " ")                   # rule 6
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def aggressive(text: str) -> str:
    """For lexical models. Collapse every variant that means the same thing."""
    text = light(text)
    text = text.replace(ZWJ, "")                       # rule 2b — see note below
    text = "".join(DIGIT_MAP.get(c, c) for c in text)  # rule 4
    text = text.replace(DANDA, f" {DANDA} ")           # rule 5
    text = _NON_TEXT.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ZWJ is stripped only at the aggressive level because it is load-bearing in some
# conjuncts — র‍্য is the usual example. Stripping it for BM25 costs nothing,
# because both spellings then collapse to the same token; stripping it before a
# transformer would hand the tokenizer a string no human writes.
