"""Score the two required non-neural summarization baselines: first-sentence
extraction and TextRank.

    python scripts/72_summary_extractive_baselines.py

Both run on CPU with no training, so they exist before any GPU time is spent -
CLAUDE.md's rule for BM25 applies the same way here: a neural summarizer with
no non-neural floor is unpublishable, and the floor has to be tuned/real, not
a strawman. Scored against `summaries_test_v1.jsonl` (the frozen 10% split,
19 rows at time of writing) with ROUGE-1/2/L and chrF - the two
language-agnostic, no-download metrics available offline. BERTScore and
human-usefulness scoring are not run here; they need a downloaded scoring
model / an annotator respectively and are the fine-tuned model's job to beat
against, not this floor's.

Sentence splitting on `।` (Bangla full stop) with a fallback on `.`/`?`/`!`
handles the mixed Bangla/English source provisions in this corpus without an
external tokenizer.
"""

from __future__ import annotations

import json
import pathlib
import re

import networkx as nx
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

TEST = pathlib.Path("data/processed/summaries_test_v1.jsonl")
REPORT = pathlib.Path("results/runs/summary_baselines_v1.json")

SENT_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


def first_sentence_baseline(source: str, max_sentences: int = 2) -> str:
    sentences = split_sentences(source)
    return " ".join(sentences[:max_sentences])


def textrank_baseline(source: str, max_sentences: int = 3) -> str:
    """Extractive TextRank over Jaccard-similarity sentence graph.

    A from-scratch TextRank (no `sumy`/`gensim.summarization`, both either
    unavailable or removed upstream) so this stays dependency-light: build a
    sentence-similarity graph from bag-of-word Jaccard overlap, rank by
    PageRank, keep the top sentences in their original order so the summary
    still reads as connected prose rather than a shuffled bag.
    """
    sentences = split_sentences(source)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    tokenized = [set(s.split()) for s in sentences]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            union = tokenized[i] | tokenized[j]
            if not union:
                continue
            similarity = len(tokenized[i] & tokenized[j]) / len(union)
            if similarity > 0:
                graph.add_edge(i, j, weight=similarity)

    try:
        scores = nx.pagerank(graph, weight="weight")
    except nx.PowerIterationFailedConvergence:
        scores = {i: 1.0 for i in range(len(sentences))}

    top = sorted(scores, key=scores.get, reverse=True)[:max_sentences]
    return " ".join(sentences[i] for i in sorted(top))


class WhitespaceTokenizer:
    """rouge_score's built-in tokenizer strips non-ASCII-alphanumeric text,
    which silently zeroes every score on Bangla (confirmed: identical Bangla
    strings scored ROUGE-1 0.0 against themselves). Whitespace splitting is
    the right level for a space-delimited script like Bangla.
    """

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def score(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=WhitespaceTokenizer()
    )
    rouge = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in rouge:
            rouge[key].append(result[key].fmeasure)
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    return {
        "rouge1_f": float(np.mean(rouge["rouge1"])),
        "rouge2_f": float(np.mean(rouge["rouge2"])),
        "rougeL_f": float(np.mean(rouge["rougeL"])),
        "chrf": float(chrf),
    }


def main() -> None:
    rows = read_jsonl(TEST)
    references = [row["summary_bn"] for row in rows]

    per_query = []
    first_preds, textrank_preds = [], []
    for row in rows:
        fs = first_sentence_baseline(row["source_text"])
        tr = textrank_baseline(row["source_text"])
        first_preds.append(fs)
        textrank_preds.append(tr)
        per_query.append({
            "id": row["id"], "provision_id": row["provision_id"], "domain": row["domain"],
            "reference": row["summary_bn"], "first_sentence": fs, "textrank": tr,
        })

    report = {
        "test_rows": len(rows),
        "note": "Non-neural floor only. 19 test rows at time of writing is too few for a "
                "stable population estimate on its own - report alongside the fine-tuned "
                "model's number on the same rows, not in isolation. Every test-split "
                "provision is a single sentence by this splitter's definition (no Bangla "
                "'।' inside them), so first-sentence and TextRank both collapse to "
                "'return the whole source text' here and score identically - this is the "
                "real behavior of both baselines on short provisions, not a bug; it will "
                "diverge once longer, multi-sentence provisions are in the test split. "
                "BERTScore and human usefulness scoring are not computed here (need a "
                "downloaded model / an annotator).",
        "first_sentence": score(first_preds, references),
        "textrank": score(textrank_preds, references),
        "per_query": per_query,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
