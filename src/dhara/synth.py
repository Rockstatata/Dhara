"""Synthetic question generation, and the lexical-overlap audit that keeps it honest.

Read §5.3 of the implementation guide before changing anything here.

## Why this file is template-driven and not LLM-driven

The obvious way to make question→provision pairs is to hand a section to an LLM
and ask for a citizen question about it. Every such pipeline has the same defect,
and the guide calls it the single biggest threat to the project's validity: a
model asked to write a question about a passage reuses that passage's vocabulary
no matter what the prompt says. Train and evaluate on those pairs and you have
built a lexical-overlap task, your fine-tuned model posts excellent numbers, and
the premise under test — that colloquial and legal Bangla are lexically
separated — was never tested at all.

This project's thesis makes that failure mode *fatal* rather than merely
embarrassing, so the generator inverts the usual control flow. Instead of writing
questions from provisions and then measuring how much vocabulary leaked, the
questions are written first, in citizen register, from a hand-authored topic
inventory — and then *matched* to provisions by a topic signature. A template
literally cannot leak a provision's phrasing, because the template was written
without seeing the provision. Overlap becomes a property we can drive toward the
real-question distribution rather than an artifact we discover afterwards.

The cost is honest and worth stating: templates are less diverse than an LLM's
output, and topic-matched positives are *weakly* labelled — no human confirmed
that this section is the best answer to this question. Both facts are recorded on
every generated row (`generator`, `quality_tier`, `match_score`, `label_source`).
These pairs are training data only. They never touch the gold set, which is
human-adjudicated, and the headline claim is measured there.

## What "citizen register" means operationally

The corpus says বিবাহ বিচ্ছেদ; a citizen says তালাক or ছাড়াছাড়ি. The corpus says
পাওনা আদায়; a citizen says টাকা ফেরত পাচ্ছি না. Each topic therefore carries a
`banned` list — terms drawn from the provisions it matches — and any rendered
question containing one is discarded. That is what makes the register gap real in
the training data instead of asserted in the paper.

## The three quality tiers

- `tier_a` — hand-authored scenario templates for a citizen topic, matched to
  provisions by signature. Highest register fidelity, narrowest coverage.
- `tier_b` — provision-title paraphrases: the title's legal phrasing rewritten
  through the register lexicon into a question. Broad coverage, weaker register.
- `tier_c` — title-derived questions with no lexicon rewrite available. Kept for
  coverage of the long tail, and the first thing to drop if training data quality
  turns out to be the binding constraint.

Every consumer can filter on the tier, and every reported number says which tiers
went in.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

from .normalize import _PUNCT_TOKENS, aggressive

# --------------------------------------------------------------------------
# The audit (§5.3). These two functions are the reason this module exists.
# --------------------------------------------------------------------------

# Function words carry no topical signal, so leaving them in makes every pair
# look more similar than it is and hides exactly the effect we are measuring.
BN_STOPWORDS = set(
    """
    আমি আমার আমাকে আমরা আমাদের তুমি তোমার তোমাকে আপনি আপনার আপনাকে সে তার তাকে
    তারা তাদের তিনি তাহার উনি এই ঐ ওই সেই কোন কোনো যে যা যাহা যাহারা কি কী কে
    কেন কীভাবে কিভাবে কোথায় কখন কতটুকু কত এবং ও বা কিংবা অথবা কিন্তু তবে তবু যদি
    তাহলে তাই সুতরাং করে করা করতে করিতে করব করবো করবে করলে হবে হয় হয়েছে হইবে
    হইয়াছে ছিল ছিলাম আছে আছি থাকে থাকা রাখা এর ের থেকে হতে জন্য সাথে সঙ্গে দিয়ে
    নিয়ে দ্বারা মাধ্যমে পর আগে পূর্বে মধ্যে উপর প্রতি বিষয়ে সম্পর্কে অনুযায়ী
    অনুসারে খুব অনেক বেশি কম সব সকল সমস্ত প্রত্যেক একটি একটা একজন এখন তখন যখন
    আবার আরও আরো ইত্যাদি না নেই নয় নাই কেউ কিছু মত মতো ভাবে দিন বছর টাকা
    """.split()
)

EN_STOPWORDS = set(
    """
    a an the and or but if then so of in on at to for by with from as is are was
    were be been being do does did have has had i my me we our you your he she it
    they them their this that these those which who whom what when where why how
    any some all each every no not nor can could shall should will would may might
    must such other than there here upon under over into out about
    """.split()
)

STOPWORDS = BN_STOPWORDS | EN_STOPWORDS


def jaccard(q: str, p: str) -> float:
    """Symmetric token overlap between a question and a passage."""
    qs = set(aggressive(q).split())
    ps = set(aggressive(p).split())
    return len(qs & ps) / max(len(qs | ps), 1)


# Wholesale English loanwords that a citizen and a statute say identically -
# there is no Bangla synonym in circulation for these, unlike everything in
# register_map.json (which exists precisely because a *different* word is
# available). Matching on one of these is not evidence of copying, so it
# carries no signal for the overlap gate. This is deliberately narrow: common
# native Bangla words that merely recur across many provisions (তথ্য, সরকার,
# আদালত, মামলা, ...) are NOT here, because the human baseline shows real
# citizens fail the gate on exactly these words 7-11% of the time - exempting
# them would decouple the gate from that calibration rather than fix a bug.
# Basic case-suffix inflections are listed explicitly since aggressive() does
# not stem; this list is not exhaustive and is meant to be extended as new
# loanwords are found, the same way register_map.json grows.
NO_GAP_LOANWORDS: frozenset[str] = frozenset(
    """
    সাইবার সার্টিফিকেট সার্টিফিকেটের ডেভেলপার ডেভেলপারের ফিটনেস
    রেজিস্ট্রেশন রেজিস্ট্রেশনের লাইসেন্স লাইসেন্সের পারমিট পারমিটের
    টোকেন টোকেনের ডিজিটাল নেটওয়ার্ক নেটওয়ার্কে সিস্টেম সিস্টেমে
    ট্রাইব্যুনাল ট্রাইব্যুনালে ট্রাইব্যুনালের চেয়ারম্যান চেয়ারম্যানের
    নোটিশ নোটিশে কমিশন কমিশনের কমিশনকে এনআইডি এনআইডির
    """.split()
)


def content_overlap(q: str, p: str, stopwords: set = STOPWORDS) -> float:
    """Fraction of the question's content words that appear in the passage.

    More diagnostic than raw Jaccard: Jaccard is dominated by passage length (a
    long section shares few of its tokens with any short question, so every pair
    scores low and the metric flattens), whereas this asks the question that
    actually matters — did the question writer copy from the passage.

    Bug fix (2026-09-17): `aggressive()` deliberately pads the danda (।) into
    its own token — normalize.py's own comment explains why, for Word2Vec's
    context window — but that makes it a "content word" here too, and it
    appears in nearly every Bangla sentence on both sides. Every overlap
    computed by this function before this fix carried a spurious +1 on both
    numerator and denominator from matching on punctuation, not content.
    `normalize.content_tokens()` already excludes it correctly; this function
    was a local reimplementation that missed that exclusion, which is exactly
    the divergence normalize.py's own docstring warns against. Excluding the
    same `_PUNCT_TOKENS` set here closes that gap rather than working around it.
    """
    qs = set(aggressive(q).split()) - stopwords - _PUNCT_TOKENS - NO_GAP_LOANWORDS
    ps = set(aggressive(p).split()) - stopwords - _PUNCT_TOKENS - NO_GAP_LOANWORDS
    return len(qs & ps) / max(len(qs), 1)


# --------------------------------------------------------------------------
# Topics
# --------------------------------------------------------------------------


@dataclass
class Topic:
    """A thing citizens actually get into trouble about.

    `signature_bn` / `signature_en` are matched against provision text to find
    candidate positives. Both are needed because 37% of the corpus is English-only
    statute and those Acts — Succession, Penal Code, CrPC, Contract, TPA — govern
    inheritance, arrest and contracts, i.e. most of what people ask about. A topic
    with no English signature silently cannot reach them.
    """

    topic_id: str
    domain: str
    risk_tier: str
    signature_bn: list[str] = field(default_factory=list)
    signature_en: list[str] = field(default_factory=list)
    # A provision must contain at least one of these to match at all. Used to stop
    # e.g. a dowry topic from matching every section that merely says "নারী".
    require_bn: list[str] = field(default_factory=list)
    require_en: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    slots: dict[str, list[str]] = field(default_factory=dict)
    # Statute vocabulary that must never appear in a rendered question.
    banned: list[str] = field(default_factory=list)

    def render_all(self, max_per_template: int = 6, seed: int = 0) -> list[str]:
        """Expand every template against its slot fillers.

        Deterministic: the same topic always yields the same questions in the
        same order, so a regenerated dataset is byte-identical and a result can
        be traced back to the exact question that produced it.
        """
        rng = random.Random(f"{self.topic_id}:{seed}")
        out: list[str] = []
        for template in self.templates:
            names = re.findall(r"\{(\w+)\}", template)
            if not names:
                out.append(template)
                continue
            pools = [self.slots.get(n, [""]) for n in names]
            combos = list(itertools.product(*pools))
            rng.shuffle(combos)
            for combo in combos[:max_per_template]:
                out.append(template.format(**dict(zip(names, combo))))
        # Dedupe while preserving order; templates can collide on shared slots.
        seen: set[str] = set()
        unique = []
        for q in out:
            key = re.sub(r"\s+", " ", q).strip()
            if key not in seen:
                seen.add(key)
                unique.append(key)
        return unique

    def violates_register(self, question: str) -> Optional[str]:
        """Return the banned statute term the question leaked, if any."""
        norm = aggressive(question)
        for term in self.banned:
            if aggressive(term) and aggressive(term) in norm:
                return term
        return None


@dataclass
class Narrative:
    """Scaffolding that turns a one-line ask into a citizen's letter.

    ## The mismatch this fixes

    Real mined questions are newspaper legal-advice letters: median 74 words,
    p10-p90 spanning 11 to 167, standard deviation 56.3. The first template
    generation produced one-liners: median 11 words, p10-p90 of 7 to 16,
    standard deviation 3.41. Roughly seven times shorter with a sixteenth of
    the variance, which is a train/test distribution gap severe enough to
    matter on its own.

    Note what this is *not* claiming. Measured against the base BGE-m3 encoder,
    the short templates were no more concentrated than the real questions
    (0.6983 vs 0.6990 mean cosine to centroid), so length was never the cause of
    attempt 2's query-space collapse — that came from the training dynamics.
    These are two separate defects and this class addresses only the first.

    Composition is `opener + backgrounds + core + closing`, where `core` is the
    hand-authored topic template. Nothing here mentions a statute, names an Act,
    or states a legal conclusion: scaffolding sets a scene, and the topic's
    `banned` list is still applied to the finished question, so context cannot
    smuggle in legal register through the back door.
    """

    openers: dict[str, list[str]] = field(default_factory=dict)
    backgrounds: dict[str, list[str]] = field(default_factory=dict)
    closings: list[str] = field(default_factory=list)
    background_count_weights: dict[int, int] = field(default_factory=dict)
    # Domains where the emotionally-loaded `sensitive` clauses are plausible.
    # "I stayed silent out of shame" belongs in a domestic-violence letter, not
    # in a question about a mistyped land area.
    sensitive_domains: list[str] = field(default_factory=list)

    # Speaker cues. A core that says "my husband" has a female speaker, so an
    # opener saying "I am a farmer" (কৃষক reads male) or a background clause
    # saying "my wife is at her father's house" would produce a persona no real
    # letter has. Cheap surface matching is enough: these are the only gendered
    # relations the topic inventory actually uses.
    FEMALE_CUES = ("আমার স্বামী", "স্বামীর", "স্বামীকে", "স্বামী ")
    MALE_CUES = ("আমার স্ত্রী", "স্ত্রীর", "স্ত্রীকে", "স্ত্রী ")

    @classmethod
    def from_yaml(cls, path) -> "Narrative":
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        openers = raw.get("openers", {})
        if isinstance(openers, list):        # tolerate the pre-grouping format
            openers = {"neutral": openers}
        return cls(
            openers=openers,
            backgrounds=raw.get("backgrounds", {}),
            closings=raw.get("closings", []),
            background_count_weights={int(k): int(v)
                                      for k, v in (raw.get("background_count_weights") or {}).items()},
            sensitive_domains=raw.get("sensitive_domains", []),
        )

    @classmethod
    def speaker_of(cls, text: str) -> Optional[str]:
        """'female' | 'male' | None — whom the text implies is writing."""
        has_female = any(c in text for c in cls.FEMALE_CUES)
        has_male = any(c in text for c in cls.MALE_CUES)
        if has_female and not has_male:
            return "female"
        if has_male and not has_female:
            return "male"
        return None

    def _compatible(self, candidates: list[str], speaker: Optional[str]) -> list[str]:
        """Drop candidates that contradict the speaker the core established."""
        if speaker is None:
            return candidates
        opposite = "male" if speaker == "female" else "female"
        return [c for c in candidates if self.speaker_of(c) != opposite]

    def _counts(self) -> tuple[list[int], list[int]]:
        items = sorted(self.background_count_weights.items())
        return [k for k, _ in items], [w for _, w in items]

    def compose(self, core: str, domain: str, rng: random.Random) -> str:
        """Wrap one core ask in a letter. `rng` carries the determinism."""
        speaker = self.speaker_of(core)

        pool = list(self.backgrounds.get("generic", []))
        pool += self.backgrounds.get(domain, [])
        if domain in self.sensitive_domains:
            pool += self.backgrounds.get("sensitive", [])
        pool = self._compatible(pool, speaker)

        counts, weights = self._counts()
        n_background = rng.choices(counts, weights=weights, k=1)[0] if counts else 0
        n_background = min(n_background, len(pool))

        opener_pool = list(self.openers.get("neutral", []))
        if speaker:
            opener_pool += self.openers.get(speaker, [])

        parts: list[str] = []
        if opener_pool:
            opener = rng.choice(opener_pool).strip()
            if opener:
                parts.append(opener)
        if n_background:
            parts.extend(rng.sample(pool, n_background))
        parts.append(core.strip())
        if self.closings:
            closing = rng.choice(self.closings).strip()
            if closing:
                parts.append(closing)

        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def variants(self, core: str, domain: str, topic_id: str, n: int, seed: int) -> list[str]:
        """`n` distinct letters around the same core ask.

        Distinct is enforced rather than hoped for: identical scaffolding drawn
        twice would inflate the question count without adding any diversity,
        which is precisely the failure being corrected.
        """
        out: list[str] = []
        seen: set[str] = set()
        for i in range(n * 4):          # oversample; duplicates are discarded
            rng = random.Random(f"{topic_id}:{core}:{seed}:{i}")
            composed = self.compose(core, domain, rng)
            if composed not in seen:
                seen.add(composed)
                out.append(composed)
            if len(out) >= n:
                break
        return out


@dataclass
class Haystack:
    """A chunk's normalized text, computed once.

    Matching 45 topics against 39,484 chunks calls `aggressive()` 1.8M times if
    the normalization lives inside the scoring loop, which takes minutes. The
    normalized forms depend only on the chunk, so they are computed once here and
    the topic signatures are pre-normalized on the Topic side. Same numbers,
    two orders of magnitude faster.
    """

    body: str
    title: str


def prepare(chunks: Iterable) -> list[Haystack]:
    return [
        Haystack(
            body=aggressive(f"{c.provision_title_bn or ''} {c.text_bn}"),
            title=aggressive(c.provision_title_bn or ""),
        )
        for c in chunks
    ]


def normalized_signature(topic: Topic) -> tuple[list[str], list[str]]:
    """(signature terms, require terms), normalized once per topic."""
    signature = [aggressive(t) for t in topic.signature_bn + topic.signature_en]
    require = [aggressive(t) for t in topic.require_bn + topic.require_en]
    return [s for s in signature if s], [r for r in require if r]


def match_score_pre(
    hay: Haystack, signature: Sequence[str], require: Sequence[str]
) -> float:
    """How strongly a provision belongs to a topic, on pre-normalized inputs.

    Title hits count double: a section titled তালাকের নোটিশ is about divorce
    notice, whereas a section that merely mentions তালাক once in a proviso is
    usually about something else. Score is normalised by signature length so
    topics with many signature terms are not automatically stronger.
    """
    if require and not any(r in hay.body for r in require):
        return 0.0
    score = 0.0
    for term in signature:
        if term in hay.title:
            score += 2.0
        elif term in hay.body:
            score += 1.0
    return score / max(len(signature), 1)


def match_score(topic: Topic, text: str, title: str = "") -> float:
    """Convenience wrapper for one-off scoring; the batch path uses `prepare`."""
    signature, require = normalized_signature(topic)
    hay = Haystack(body=aggressive(f"{title} {text}"), title=aggressive(title))
    return match_score_pre(hay, signature, require)


# --------------------------------------------------------------------------
# Row construction
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# Question-level positive selection
#
# A topic's anchors cover everything that topic is about; one question asks for
# one thing. Pairing every question with every anchor -- which is what tier A did
# until 2026-09-04 -- gives a median of 6 "correct" provisions per question, of
# which typically one or two actually answer it. Measured example: "কম বয়সে বিয়ে
# দিলে বাবা-মায়ের কি শাস্তি হয়?" carried three positives (prohibition,
# registrar's duty, compensation) and none of them was the parents' punishment
# section.
#
# Under MultipleNegativesRankingLoss every one of those is pulled toward the
# question with equal force, so the four wrong ones outvote the right one. This
# is the same defect as the gold set's, arrived at from the generator side.
#
# The fix is a cue match between what the question ASKS FOR and what the
# provision's own title SAYS IT IS. Both sides are hand-written strings, so this
# is auditable in a way a model-scored selection would not be -- and it must be,
# because using the retriever to pick training labels for the retriever is
# circular.
#
# Coverage is deliberately incomplete. When a question expresses no recognised
# intent, or no anchor matches the one it expresses, every anchor is kept and the
# row is counted under `intent_fallback`. A wrong-but-plausible positive is
# better than no positive; a silently narrowed training set is not.

INTENT_CUES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # intent: (cues in the question, cues in the provision title)
    "penalty": (
        ("শাস্তি", "দণ্ড", "সাজা", "জেল", "কারাদণ্ড", "জরিমানা", "অপরাধ"),
        ("দণ্ড", "শাস্তি", "অপরাধ", "penalt", "punish", "offence", "offense"),
    ),
    "procedure": (
        ("মামলা", "অভিযোগ", "কোথায়", "আবেদন", "থানা", "আদালতে", "কীভাবে", "কি করব",
         "কী করব", "করণীয়", "প্রতিকার"),
        ("অভিযোগ", "মামলা", "দায়ের", "আদালত", "কার্যক্রম", "পদ্ধতি", "আবেদন",
         "procedure", "application", "complaint", "cognizance", "trial", "suit"),
    ),
    "time_limit": (
        ("কতদিন", "কত দিন", "সময়সীমা", "মেয়াদ", "তামাদি", "কতদিনের"),
        ("সময়", "মেয়াদ", "তামাদি", "limitation", "period", "time"),
    ),
    "compensation": (
        ("ক্ষতিপূরণ", "টাকা ফেরত", "ফেরত পাব", "লোকসান"),
        ("ক্ষতিপূরণ", "ফেরত", "compensation", "damages", "refund"),
    ),
    "registration": (
        ("নিবন্ধন", "রেজিস্ট্রি", "রেজিস্ট্রেশন", "কাবিন", "দলিল"),
        ("নিবন্ধন", "রেজিস্ট্রি", "registration", "register", "record"),
    ),
    "entitlement": (
        ("অধিকার", "প্রাপ্য", "ভাগ", "পাব", "পাবে", "হকদার", "অংশ"),
        ("অধিকার", "প্রাপ্য", "ভাগ", "অংশ", "right", "share", "entitle", "succession"),
    ),
    "prohibition": (
        ("নিষেধ", "সীমা আছে", "বাড়াতে পারে", "পারে কি", "বৈধ", "অবৈধ"),
        ("নিষেধ", "বাধা", "নিষিদ্ধ", "সীমা", "restrict", "prohibit", "bar ", "void"),
    ),
    # The four below were added after sampling what fell through: 178 of 482
    # fallback questions expressed no recognised intent, and they clustered into
    # "is this valid", "who looks after the child", "where do I shelter", and
    # "must someone be told".
    "validity": (
        ("ঠিক আছে", "ঠিক হলো", "টিকবে", "নিয়মে আছে", "নিয়মমতো", "করা যায়",
         "চলবে", "গ্রহণযোগ্য", "কার্যকর"),
        ("কার্যকর", "বৈধ", "বাতিল", "valid", "void", "effect", "operation",
         "registration", "নিবন্ধন"),
    ),
    "guardianship": (
        ("দেখাশোনা", "অভিভাবক", "হেফাজত", "কে পাবে", "লালনপালন", "সন্তানের দায়িত্ব"),
        ("অভিভাবক", "হেফাজত", "জিম্মা", "guardian", "custody", "ward", "minor"),
    ),
    "protection": (
        ("আশ্রয়", "নিরাপত্তা", "সুরক্ষা", "বাঁচার উপায়", "রক্ষা"),
        ("সুরক্ষা", "নিরাপত্তা", "আশ্রয়", "protection", "safe", "shelter", "relief"),
    ),
    "notice": (
        ("জানাতে হয়", "নোটিশ", "জানানো", "জমা দিতে হয়", "জানাতে হবে"),
        ("নোটিশ", "জ্ঞাপন", "notice", "intimation", "inform", "submission"),
    ),
}


def intents_of(question: str) -> set:
    """Which recognised intents a question expresses. May be empty."""
    text = question.lower()
    return {name for name, (q_cues, _t_cues) in INTENT_CUES.items()
            if any(cue.lower() in text for cue in q_cues)}


def _title_matches(chunk, intents: set) -> bool:
    """Match on the provision title, falling back to its opening words.

    Roughly one anchor in six has no `provision_title_bn` at all -- the scraper
    found a section with no printed marginal note. Those would never match on
    title and would drag their question into the fallback path, so the first 120
    characters of the body stand in. Only when the title is missing: a body is
    long enough that cue matching over all of it would match nearly anything,
    which would make the selection look precise while being arbitrary.
    """
    title = (getattr(chunk, "provision_title_bn", "") or "").strip().lower()
    haystack = title or " ".join((getattr(chunk, "text_bn", "") or "").split())[:120].lower()
    if not haystack:
        return False
    return any(cue.lower() in haystack
               for name in intents
               for cue in INTENT_CUES[name][1])


def dedupe_by_provision(chunks: Sequence) -> list:
    """One chunk per provision, preferring the first sub-chunk.

    A long section is split into overlapping sub-chunks that share a citation.
    Treating three of them as three separate positives for one question triples
    that provision's pull without adding information, and it was happening for
    198 of 826 questions.
    """
    best: dict[str, object] = {}
    for chunk in chunks:
        current = best.get(chunk.provision_id)
        if current is None or getattr(chunk, "sub_idx", 0) < getattr(current, "sub_idx", 0):
            best[chunk.provision_id] = chunk
    return [best[pid] for pid in sorted(best)]


def select_positives(question: str, anchored: Sequence, max_positives: int = 3) -> tuple[list, str]:
    """The anchors that answer THIS question, and how they were chosen.

    Returns `(chunks, mode)` where mode is one of:
      `intent`   - the question's intent matched at least one anchor title;
      `fallback` - no intent recognised, or none matched: every anchor is kept.
    """
    unique = dedupe_by_provision(anchored)
    intents = intents_of(question)
    if intents:
        matched = [c for c in unique if _title_matches(c, intents)]
        if matched:
            return matched[:max_positives], "intent"
    return unique, "fallback"

def qid_for(question: str, chunk_id: str) -> str:
    """Stable id from content, so regeneration does not renumber the dataset."""
    digest = hashlib.sha1(f"{question}||{chunk_id}".encode("utf-8")).hexdigest()
    return f"syn_{digest[:12]}"


def make_row(
    question: str,
    chunk,
    *,
    topic_id: str,
    generator: str,
    quality_tier: str,
    match_score_value: float,
) -> dict:
    passage = f"{chunk.provision_title_bn or ''} {chunk.text_bn}".strip()
    return {
        "qid": qid_for(question, chunk.chunk_id),
        "question_bn": question,
        "gold_chunk_ids": [chunk.chunk_id],
        "gold_provision_ids": [chunk.provision_id],
        "act_id": chunk.act_id,
        "domain": chunk.domain,
        "risk_tier": chunk.risk_tier,
        "lang_tag": chunk.language,
        "topic_id": topic_id,
        "generator": generator,
        "quality_tier": quality_tier,
        "match_score": round(match_score_value, 4),
        # Never "human". These are weak labels and every consumer must be able
        # to see that without reading the generator's source.
        "label_source": "topic_signature" if generator == "tier_a" else "provision_title",
        "content_overlap": round(content_overlap(question, passage), 4),
        "jaccard": round(jaccard(question, passage), 4),
        "source": "synthetic",
        "annotation_mode": "generated",
    }


# --------------------------------------------------------------------------
# Provisions that no citizen will ever ask about
# --------------------------------------------------------------------------

# 33,307 of 39,484 chunks are domain "other", and the corpus survey shows what
# that is: definitions, rule-making powers, budgets, university syndicate duties,
# and amendments to other Acts. Generating citizen questions for those would
# flood training with pairs no user will ever issue, and would make the training
# distribution diverge from the evaluation distribution in a way that shows up as
# a null result rather than as an error.
BOILERPLATE_TITLE = re.compile(
    r"^\s*\[?\s*(omitted|repealed)"
    r"|^\s*(সংজ্ঞা|definitions?)\s*$"
    r"|সংক্ষিপ্ত\s*শিরোনাম|short\s*title"
    r"|বিধি\s*প্রণয়ন|প্রবিধান\s*প্রণয়ন|power\s+to\s+make\s+(rules|regulations)"
    r"|রহিতকরণ|repeal\s+and\s+savings?"
    r"|ইংরেজিতে\s*অনূদিত|অসুবিধা\s*দূরীকরণ|removal\s+of\s+difficult"
    r"|ক্ষমতা\s*অর্পণ|delegation\s+of\s+power"
    r"|^\s*(বাজেট|তহবিল|budget|fund)\s*$"
    r"|হিসাব\s*রক্ষণ|হিসাবরক্ষণ|accounts?\s+and\s+audit"
    r"|বার্ষিক\s*প্রতিবেদন|annual\s+report"
    r"|সংশোধন|amendment\s+of|amendments\s+of"
    r"|আইনের\s*প্রাধান্য|act\s+to\s+override",
    re.IGNORECASE,
)

REPEALED_BODY = re.compile(r"^\s*\[?\s*(omitted|repealed|রহিত করা হইয়াছে|বিলুপ্ত)", re.IGNORECASE)


def is_repealed(chunk) -> bool:
    """Retrieving a repealed section and showing it to a citizen is real harm.

    They are still present in the frozen corpus_v1 (887 of them) and must be
    excluded at index time until corpus_v2 drops them properly.
    """
    title = (chunk.provision_title_bn or "").strip()
    return bool(REPEALED_BODY.match(title) or REPEALED_BODY.match(chunk.text_bn.strip()))


def is_boilerplate(chunk) -> bool:
    title = (chunk.provision_title_bn or "").strip()
    if not title:
        return False
    return bool(BOILERPLATE_TITLE.search(title))


def is_askable(chunk, min_words: int = 25) -> bool:
    """Could a citizen plausibly have a question this provision answers."""
    if is_repealed(chunk) or is_boilerplate(chunk):
        return False
    if chunk.n_words < min_words:
        return False
    return True


# --------------------------------------------------------------------------
# Audit reporting
# --------------------------------------------------------------------------


def overlap_summary(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        return round(ordered[min(int(p * n), n - 1)], 4)

    return {
        "n": n,
        "mean": round(sum(ordered) / n, 4),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "median": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "frac_over_0.3": round(sum(1 for v in ordered if v > 0.3) / n, 4),
        "frac_over_0.5": round(sum(1 for v in ordered if v > 0.5) / n, 4),
    }
