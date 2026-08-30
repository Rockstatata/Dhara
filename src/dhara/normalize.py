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


# --------------------------------------------------------------------------
# Stopwords, for lexical models only
# --------------------------------------------------------------------------
#
# Without these, BM25 scores a citizen question mostly on আমি / কি / ও / দিতে and
# ranks long documents first: a question about যৌতুক ও দেনমোহর returned
# ভূমি উন্নয়ন কর আইন as its top hit while the Dowry Prohibition Act sat at rank 3.
#
# Two deliberate omissions:
#
#   - **Negation stays.** না, নেই, নয় are frequent enough to look like stopwords,
#     but "করা যাইবে না" (shall not) and "করা যাইবে" (shall) are opposite legal
#     rules. A bag of words cannot exploit the difference, yet discarding it
#     guarantees the loss. Keeping them costs a little precision and no
#     correctness.
#   - **Nothing domain-specific.** ধারা, আইন and আদালত are common but they are
#     what a citizen searching for a provision actually types.
#
# Transformers never see this list — their tokenizers were trained on natural
# text that contains these words.
STOPWORDS: frozenset[str] = frozenset(
    """
    আমি আমার আমাকে আমরা আমাদের তুমি তোমার তোমাকে আপনি আপনার আপনাকে
    সে তার তাকে তারা তাদের তিনি তাঁর তাঁকে এটি এটা ইহা উহা
    এই ঐ ওই সেই কোন কোনো যে যা যাহা যাহার তাহা তাহার
    কি কী কে কাকে কার কেন কোথায় কখন কিভাবে কীভাবে
    এবং ও বা কিংবা অথবা কিন্তু তবে তবু যদি তাহলে তাই সুতরাং
    করে করা করতে করিতে করিয়া হবে হয় হয়েছে হইবে হইয়া ছিল আছে থাকে থাকিবে
    এর ের থেকে হতে হইতে জন্য সাথে সঙ্গে দিয়া দিয়ে নিয়ে লইয়া দ্বারা মাধ্যমে
    পর আগে পূর্বে মধ্যে ভিতরে উপর উপরে নিচে নিম্নে প্রতি বিষয়ে সম্পর্কে
    মত মতো ন্যায় অনুযায়ী অনুসারে
    খুব অনেক বেশি কম সব সকল সমস্ত প্রত্যেক একটি একটা টি টা টির
    এখন তখন যখন এখনও আবার আরও ইত্যাদি
    """.split()
)


# `aggressive()` pads the danda into its own token on purpose: it is a sentence
# boundary and Word2Vec's context window needs to see it. A bag-of-words ranker
# must not. The danda occurs in 3,980 of 6,359 corpus chunks, so leaving it in
# gave every long provision a free term match and let document length, rather
# than topic, drive the BM25 ranking. Dropped here rather than in `aggressive()`
# so the Word2Vec contract is unchanged.
_PUNCT_TOKENS: frozenset[str] = frozenset({DANDA, ",", ".", ";", ":", "-", "—"})


def content_tokens(text: str) -> list[str]:
    """Aggressively normalize, then drop stopwords and punctuation.

    For BM25/TF-IDF/Word2Vec ranking. Not for Word2Vec *training*, which wants
    the danda as a boundary marker.
    """
    return [
        t
        for t in aggressive(text).split()
        if t not in STOPWORDS and t not in _PUNCT_TOKENS
    ]


# ZWJ is stripped only at the aggressive level because it is load-bearing in some
# conjuncts — র‍্য is the usual example. Stripping it for BM25 costs nothing,
# because both spellings then collapse to the same token; stripping it before a
# transformer would hand the tokenizer a string no human writes.
