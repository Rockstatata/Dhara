"""Query insights for the demo: domain classification (Topic 1) and sequence
tagging (Topic 2), run alongside retrieval and shown next to it.

These are the two syllabus components that were built and evaluated as parallel
tracks. They are surfaced in the demo so a reader can see what they actually do,
and they are **display-only**: nothing here filters, reorders, boosts or gates
retrieval. Intent-based candidate filtering is an ablation, never the default
path (CLAUDE.md) -- a wrong intent prediction would silently destroy retrieval,
and this classifier is wrong more often than right on real questions.

## What is served, and why these two models

**Domain classifier -- Logistic Regression on TF-IDF, human-only training set.**
The best of the four classifiers in `results/runs/classify_v6.json` (macro-F1
0.343, accuracy 0.475 on 200 held-out human questions), and it beats every model
trained on 12x more (mostly synthetic) rows. It is re-fit here from the same
files, the same rows and `build_tfidf_classifiers` as
`scripts/42_train_classifiers.py` (with C pinned to the recorded 1.0, see `fit`)
rather than pickled: the vectorizer holds a lambda tokenizer, which cannot be
pickled, and refitting 352 rows takes about a second. `tests/test_insights.py` checks the refit reproduces the recorded
per-question predictions of the reported run.

**Sequence tagger -- `BiRNNSequenceLabeler`, checkpoint `models/tagger.pt`.**
Trained on weak labels (a section-number regex plus the register-map gazetteer),
never hand-corrected. Its recorded scores (`results/runs/tagger_v3.json`) are
self-consistency with those rules on a 58-question internal slice, NOT accuracy
against human ground truth: ACT F1 0.857 and PARTY F1 0.627 are real learning
of the rules; SECTION_NO and LEGAL_TERM are 0.0 -- not learned (2 and 155
weak-labelled tokens in the whole training set). The demo therefore reports the
regex-found section numbers separately from the neural tags rather than letting
the model take credit for what a rule found.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

import torch

from .normalize import aggressive
from .vocab import Vocab

GOLD = pathlib.Path("data/processed/gold_verified_v3.jsonl")
SPLIT = pathlib.Path("data/processed/test_split_v1.json")
VOCAB = pathlib.Path("data/processed/vocab.json")
MATRIX = pathlib.Path("models/embedding_matrix.pt")
TAGGER = pathlib.Path("models/tagger.pt")
DOMAINS_CFG = pathlib.Path("configs/domains.yaml")
CLASSIFY_RUN = pathlib.Path("results/runs/classify_v6.json")
TAGGER_RUN = pathlib.Path("results/runs/tagger_v3.json")

# The weak labeller's rule (scripts/43_train_tagger.py): a digit token within two
# positions of a section marker -- the rule the tagger was trained to imitate.
# The marker match here is deliberately looser than the labeller's exact-token
# match: Bangla inflects the noun (ধারায়, ধারার, ধারাটি), and "৩২৩ ধারায়" is how
# people write it. This is a display-side rule, reported as a rule; it is not
# what the tagger's SECTION_NO score measures.
SECTION_MARKERS = ("ধারা", "অনুচ্ছেদ", "section")
SECTION_NO_RE = re.compile(r"^\d+[a-z]?$")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def domain_names() -> dict[str, tuple[str, str]]:
    """domain id -> (Bangla name, English name), from configs/domains.yaml."""
    import yaml

    cfg = yaml.safe_load(DOMAINS_CFG.read_text(encoding="utf-8"))
    names = {d["id"]: (d.get("name_bn", d["id"]), d.get("name_en", d["id"])) for d in cfg["domains"]}
    names["other"] = ("অন্যান্য", "Other (outside the covered domains)")
    return names


# ------------------------------------------------------------------ classifier

class DomainClassifier:
    """Logistic Regression over TF-IDF: question -> legal domain."""

    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
        self.classes: list[str] = [str(c) for c in model.classes_]

    @classmethod
    def fit(cls, gold: pathlib.Path = GOLD, split: pathlib.Path = SPLIT) -> "DomainClassifier":
        from dhara.classify import build_tfidf_classifiers

        train_qids = set(json.loads(split.read_text(encoding="utf-8"))["train_qids"])
        by_qid = {r["qid"]: r for r in _read_jsonl(gold)}
        # Same rows and features as scripts/42_train_classifiers.py::main, but C is
        # pinned to 1.0 instead of CV-selected. That script builds its rows by
        # iterating a *set* of qids, so row order -- and with it the 3-fold CV
        # folds that pick C -- changes with PYTHONHASHSEED. The CV curve is flat
        # within noise (macro-F1 0.17-0.26 across C in {0.3, 1, 3}, the winner
        # flipping with row order), so the choice is not a real selection. The
        # recorded run (classify_v6.json) selected C=1.0; pinning it makes this
        # model deterministic and identical to the one whose numbers are reported.
        rows = [by_qid[q] for q in sorted(train_qids) if q in by_qid]
        vectorizer, _nb, lr = build_tfidf_classifiers(
            [r["question_bn"] for r in rows], [r["domain"] for r in rows], tune_lr=False
        )
        return cls(vectorizer, lr)

    def predict(self, query: str, top_n: int = 3) -> list[tuple[str, float]]:
        """[(domain_id, probability), ...] best first."""
        probs = self.model.predict_proba(self.vectorizer.transform([query]))[0]
        order = probs.argsort()[::-1][:top_n]
        return [(self.classes[i], float(probs[i])) for i in order]

    def predict_label(self, query: str) -> str:
        return str(self.model.predict(self.vectorizer.transform([query]))[0])

    @staticmethod
    def recorded_f1(run: pathlib.Path = CLASSIFY_RUN) -> dict[str, float]:
        """Per-domain F1 of this exact model on the 200 held-out human questions,
        read from the recorded run (never recomputed here) so the demo can show
        how much to trust each predicted domain."""
        data = json.loads(run.read_text(encoding="utf-8"))
        return data["human_only_fine_grained"]["models"]["logistic_regression"]["per_class_f1"]


# ---------------------------------------------------------------------- tagger

@dataclass
class TaggedQuery:
    tokens: list[str]
    tags: list[str]
    # Section numbers found by the regex rule, independent of the neural tags.
    rule_section_numbers: list[str] = field(default_factory=list)

    def entities(self) -> dict[str, list[str]]:
        """tag -> ordered unique surface tokens, entity tags only."""
        out: dict[str, list[str]] = {}
        for tok, tag in zip(self.tokens, self.tags):
            if tag != "O" and tok not in out.setdefault(tag, []):
                out[tag].append(tok)
        return out


def rule_section_numbers(tokens: list[str]) -> list[str]:
    found = []
    for i, tok in enumerate(tokens):
        if SECTION_NO_RE.match(tok):
            window = tokens[max(0, i - 2): i] + tokens[i + 1: i + 3]
            if any(w.startswith(SECTION_MARKERS) for w in window):
                found.append(tok)
    return found


class QueryTagger:
    """BiRNN tagger over the shared Topic-0 vocabulary and embedding matrix."""

    def __init__(self, model, vocab: Vocab, tags: list[str]):
        self.model = model
        self.vocab = vocab
        self.tags = tags

    @classmethod
    def load(cls, path: pathlib.Path = TAGGER, vocab: pathlib.Path = VOCAB,
             matrix: pathlib.Path = MATRIX) -> "QueryTagger":
        from dhara.models import tagger as tagger_mod

        embedding_matrix = torch.load(matrix, map_location="cpu", weights_only=False)
        model = tagger_mod.load(path, embedding_matrix)
        return cls(model, Vocab.load(vocab), list(tagger_mod.TAGS))

    @torch.no_grad()
    def tag(self, query: str) -> TaggedQuery:
        tokens = aggressive(query).split()
        if not tokens:
            return TaggedQuery([], [])
        ids = torch.tensor([[self.vocab.stoi.get(t, self.vocab.unk_id) for t in tokens]], dtype=torch.long)
        pred = self.model(ids).argmax(dim=-1).squeeze(0).tolist()
        return TaggedQuery(tokens, [self.tags[p] for p in pred], rule_section_numbers(tokens))
