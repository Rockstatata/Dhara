"""Query -> cited provisions, or an honest refusal to answer.

The one place retrieval, abstention and risk-tier framing meet. The Gradio app in
`src/app/app.py` is a thin shell over this; a CLI or an API would be another.

    from dhara.service import Dhara
    dhara = Dhara.load()                 # ~30s: model + index, once per process
    answer = dhara.search("পুলিশ আমাকে ধরে নিয়ে গেছে, আমার কী অধিকার?")

## Three things this file exists to enforce

**The model loads once.** `Dhara.load()` at module level in the app, never inside
a request handler. Reloading BGE-m3 per request is the "demo feels frozen" trap
in the guide's known-traps table.

**It can say no.** Below a score threshold the answer is `abstained`, with no
provisions attached. The threshold is calibrated by
`scripts/46_calibrate_abstention.py` against the gold questions marked
unanswerable, and the calibration deliberately prefers false abstentions to false
confidence: telling someone "I have no confident match" wastes a minute, and
confidently citing the wrong section of criminal law can send them to a police
station with a false belief about their rights.

**Risk tier changes the framing, never the ranking.** For high-tier domains the
answer carries `legal_aid_referral = True`, which the UI must render *above* the
provisions, and the abstention threshold is raised. Ranking is identical across
tiers -- filtering or reordering by risk would silently hide provisions from the
people who most need them.

## What it returns, and why `text_raw`

Every hit carries the citation fields (Act title, section number, source URL,
crawl date) and `text_raw` -- the law as printed. The normalized `text_bn` is
what the model consumes; showing it to a Bangla reader displays mangled
punctuation and stripped joiners. Two fields, deliberately, per §4.4.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field, asdict

import numpy as np

from . import schema
from .normalize import light

# The full-corpus index (39,484 chunks) needs a GPU to build
# (scripts/76_build_final_demo_index.py). A smaller demo-scope index (covered
# domains only, 'other' bucket excluded) is a CPU-buildable fallback for local
# runs -- falls back to it automatically if the full index isn't present yet.
_FULL_INDEX = pathlib.Path("models/index_bge_m3_finetuned_v6")
_DEMOSCOPE_INDEX = pathlib.Path("models/index_bge_m3_finetuned_v6_demoscope")
DEFAULT_INDEX = _FULL_INDEX if _FULL_INDEX.exists() else _DEMOSCOPE_INDEX
DEFAULT_CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
DEFAULT_THRESHOLDS = pathlib.Path("configs/abstention.json")
EXCLUSIONS = pathlib.Path("data/processed/excluded_repealed_v1.jsonl")

# Used only until `scripts/29_calibrate_abstention.py` has been run. Deliberately
# conservative: it is better for the demo to abstain on a few answerable
# questions than to look confident on unanswerable ones before anyone has
# measured where the line should sit.
FALLBACK_THRESHOLDS = {"default": 0.52, "high": 0.58}

# Risk-tier detection always looks at this many top candidates, regardless of
# the caller's k, so the same question gets the same tier (and therefore the
# same abstention threshold) no matter how many results the UI asks to show.
TIER_LOOKAHEAD = 5

HIGH_RISK_NOTICE_BN = (
    "এই বিষয়টি জরুরি হতে পারে। আইনি সহায়তার জন্য জাতীয় আইনগত সহায়তা সংস্থা "
    "(লিগ্যাল এইড) — হটলাইন ১৬৪৩০ — এ যোগাযোগ করুন। নিচের ধারাগুলো তথ্যের জন্য, "
    "পরামর্শের বিকল্প নয়।"
)
DISCLAIMER_BN = "এটি আইনি পরামর্শ নয়। এটি একটি তথ্য অনুসন্ধান সরঞ্জাম।"


@dataclass
class Hit:
    """One retrieved provision, with everything needed to cite it."""

    chunk_id: str
    provision_id: str
    score: float
    act_title: str
    act_year: str
    provision_kind: str
    provision_no: str
    provision_title: str
    text_raw: str
    domain: str
    risk_tier: str
    source_url: str
    crawl_date: str


@dataclass
class Answer:
    query: str
    abstained: bool
    hits: list[Hit] = field(default_factory=list)
    top_score: float | None = None
    threshold: float | None = None
    risk_tier: str = "low"
    legal_aid_referral: bool = False
    notice_bn: str = ""
    disclaimer_bn: str = DISCLAIMER_BN
    message_bn: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Dhara:
    """Loaded index + encoder. Construct once, call `search` many times."""

    def __init__(self, model, embeddings, chunk_ids, chunks, manifest, thresholds):
        self.model = model
        self.C = embeddings
        self.chunk_ids = chunk_ids
        self.chunks = chunks
        self.manifest = manifest
        self.thresholds = thresholds

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        index: pathlib.Path = DEFAULT_INDEX,
        corpus: pathlib.Path = DEFAULT_CORPUS,
        thresholds: pathlib.Path = DEFAULT_THRESHOLDS,
        device: str | None = None,
    ) -> "Dhara":
        from sentence_transformers import SentenceTransformer

        manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
        embeddings = np.load(index / "embeddings.npy").astype(np.float32)
        chunk_ids = json.loads((index / "chunk_ids.json").read_text(encoding="utf-8"))

        by_id = {c.chunk_id: c for c in schema.read_jsonl(corpus)}

        # Repealed provisions are dropped from the corpus build, but an index can
        # outlive an exclusion list, so the serving path re-checks. Retrieving a
        # repealed section is the one failure that does active harm.
        excluded: set[str] = set()
        if EXCLUSIONS.exists():
            with EXCLUSIONS.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        excluded.add(json.loads(line)["chunk_id"])

        keep = [i for i, cid in enumerate(chunk_ids) if cid in by_id and cid not in excluded]
        if len(keep) != len(chunk_ids):
            embeddings = embeddings[keep]
            chunk_ids = [chunk_ids[i] for i in keep]

        loaded_thresholds = FALLBACK_THRESHOLDS
        if thresholds.exists():
            loaded_thresholds = json.loads(thresholds.read_text(encoding="utf-8"))["thresholds"]

        model = SentenceTransformer(manifest["checkpoint"], device=device)
        model.max_seq_length = manifest["max_seq_length"]

        return cls(model, embeddings, chunk_ids, by_id, manifest, loaded_thresholds)

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 5, show_all: bool = False) -> Answer:
        """`show_all=True` still returns `hits` even when the score would
        normally abstain -- for a demo audience that wants to see the raw
        top-k regardless of confidence. `abstained`/`top_score`/`threshold`
        stay accurate either way, so the caller can still render a low-
        confidence warning instead of pretending the system was sure. Actual
        evaluation code (scripts/13_eval_dense_index.py etc.) never sets this
        -- abstention must stay real for every number CLAUDE.md governs."""
        query = (query or "").strip()
        if not query:
            return Answer(query=query, abstained=True,
                          message_bn="একটি প্রশ্ন লিখুন।")

        prefix = self.manifest.get("query_prefix", "")
        q = self.model.encode([prefix + light(query)], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)[0]
        scores = self.C @ q

        # Over-fetch, then collapse sub-chunks: a long section is split into
        # overlapping pieces that share one citation, and three of them would
        # spend three of the five slots re-citing the same provision.
        # Fetch enough candidates for TIER_LOOKAHEAD even if the caller's k is
        # smaller -- see the tier-detection note below for why.
        collect_n = max(k, TIER_LOOKAHEAD)
        top = np.argpartition(-scores, min(collect_n * 8, len(scores) - 1))[: collect_n * 8]
        top = top[np.argsort(-scores[top])]

        all_hits: list[Hit] = []
        seen: set[str] = set()
        for i in top:
            chunk = self.chunks[self.chunk_ids[i]]
            if chunk.provision_id in seen:
                continue
            seen.add(chunk.provision_id)
            all_hits.append(Hit(
                chunk_id=chunk.chunk_id,
                provision_id=chunk.provision_id,
                score=float(scores[i]),
                act_title=chunk.act_title_bn or chunk.act_title_en or chunk.act_id,
                act_year=str(chunk.act_year or ""),
                provision_kind=chunk.provision_kind or "section",
                provision_no=str(chunk.provision_no_bn or ""),
                provision_title=chunk.provision_title_bn or "",
                text_raw=chunk.text_raw or chunk.text_bn,
                domain=chunk.domain or "other",
                risk_tier=chunk.risk_tier or "low",
                source_url=chunk.source_url or "",
                crawl_date=str(chunk.crawl_date or ""),
            ))
            if len(all_hits) == collect_n:
                break
        hits = all_hits[:k]

        # Risk tier comes from what was retrieved, since the query carries no
        # label. Any high-tier provision in the top hits raises the bar and turns
        # on the referral -- erring toward showing the referral, because the cost
        # of an unnecessary legal-aid banner is nil.
        #
        # Deliberately checked over a FIXED lookahead (TIER_LOOKAHEAD), not over
        # `hits` (which is truncated to the caller's `k`). Using `hits` directly
        # made the tier -- and therefore the abstention threshold, and therefore
        # whether the top-1 result is shown at all -- depend on the UI's k
        # slider: raising k could pull a high-risk hit into view further down
        # the list, silently hiding an already-good top-1 result at a higher k
        # than at a lower one for the identical query. That is a real bug, not
        # intended behavior -- the risk framing must be stable for a given
        # question regardless of how many results the caller asked to see.
        tier = "low"
        for hit in all_hits[:TIER_LOOKAHEAD]:
            if hit.risk_tier == "high":
                tier = "high"
                break
            if hit.risk_tier == "medium" and tier != "high":
                tier = "medium"

        threshold = self.thresholds.get(tier, self.thresholds.get("default", 0.52))
        top_score = hits[0].score if hits else None

        if top_score is None or top_score < threshold:
            return Answer(
                query=query, abstained=True, hits=(hits if show_all else []), top_score=top_score,
                threshold=threshold, risk_tier=tier,
                legal_aid_referral=(tier == "high"),
                notice_bn=HIGH_RISK_NOTICE_BN if tier == "high" else "",
                message_bn=(
                    "আপনার প্রশ্নের সাথে নিশ্চিতভাবে মেলে এমন কোনো ধারা পাওয়া যায়নি। "
                    "নিচে সর্বোচ্চ স্কোরের ফলাফলগুলো দেখানো হলো, তবে এগুলোর কোনোটিই আত্মবিশ্বাসের "
                    "থ্রেশহোল্ড অতিক্রম করেনি।" if show_all else
                    "আপনার প্রশ্নের সাথে নিশ্চিতভাবে মেলে এমন কোনো ধারা পাওয়া যায়নি। "
                    "প্রশ্নটি অন্যভাবে লিখে দেখুন, অথবা একজন আইনজীবীর পরামর্শ নিন।"
                ),
            )

        return Answer(
            query=query, abstained=False, hits=hits, top_score=top_score,
            threshold=threshold, risk_tier=tier,
            legal_aid_referral=(tier == "high"),
            notice_bn=HIGH_RISK_NOTICE_BN if tier == "high" else "",
        )


def cite(hit: Hit) -> str:
    """One-line citation, the way it should appear under a result."""
    kind = "অনুচ্ছেদ" if hit.provision_kind == "article" else "ধারা"
    year = f", {hit.act_year}" if hit.act_year and hit.act_year not in hit.act_title else ""
    return f"{hit.act_title}{year} — {kind} {hit.provision_no}"


if __name__ == "__main__":            # smoke test: python -m dhara.service
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    dhara = Dhara.load()
    for q in ("পুলিশ আমাকে ধরে নিয়ে গেছে, আমার কী অধিকার?",
              "আমার স্বামী মুখে তালাক দিয়েছে, এখন কী করব?",
              "আজকে ঢাকার আবহাওয়া কেমন?"):
        answer = dhara.search(q, k=3)
        print("\nQ:", q)
        if answer.abstained:
            print(f"   ABSTAINED (top score {answer.top_score:.3f} < {answer.threshold})")
            print("  ", answer.message_bn)
            continue
        for hit in answer.hits:
            print(f"   {hit.score:.3f}  {cite(hit)}")
            print(f"          {hit.provision_title[:70]}")
