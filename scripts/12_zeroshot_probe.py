"""Rank a retriever against a small hand-verified probe set. No training involved.

    python scripts/12_zeroshot_probe.py --retriever bm25
    python scripts/12_zeroshot_probe.py --retriever biencoder --index models/index_v1

Why this exists, separately from the real gold set: building 150-200 properly
adjudicated gold questions is Pass 1's job and takes real annotator time. This
script answers a narrower, cheaper question first — "does this checkpoint /
setting cross the language gap at all" — against 7 pairs verified by reading the
actual provision text (not just its title), so a checkpoint swap or a config
change can be sanity-checked in seconds instead of waiting on annotation.

**This is not the evaluation.** n=7 has no confidence interval worth quoting and
never substitutes for `09_run_eval.py` against the real gold set. Its only job is
catching a broken run — wrong checkpoint, backwards prefixes, a corpus
mismatch — before spending an annotator's afternoon on it, and giving a same-day
answer when comparing two checkpoints (see DECISIONS.md, 2026-08-27: the 2-example
spot check said zero-shot was "real, but not yet good"; this same-shaped 7-example
probe showed 14% Recall@50 and an English/Bengali split. Small samples mislead in
both directions — this script exists so the *next* small sample is at least the
same, reusable one instead of a fresh ad hoc pick each time.)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.retrievers.bm25 import BM25Retriever  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
PROBES = pathlib.Path("data/interim/zeroshot_probe_pairs.jsonl")
CUTOFFS = (5, 10, 20, 50, 100)


def load_probes(path: pathlib.Path) -> list[dict]:
    import json

    return [
        json.loads(line) for line in path.open(encoding="utf-8") if line.strip()
    ]


def rank_of(retriever, question: str, target: str, n_total: int) -> int | None:
    for i, (chunk_id, _score) in enumerate(retriever.search(question, k=n_total), 1):
        if chunk_id == target:
            return i
    return None


def build_retriever(name: str, chunks, index_dir: pathlib.Path | None, checkpoint: str):
    if name == "bm25":
        r = BM25Retriever()
        r.index(chunks)
        return r
    if name == "biencoder":
        from dhara.retrievers.biencoder import BiEncoderRetriever

        if index_dir is None:
            raise SystemExit("--index is required for --retriever biencoder")
        r = BiEncoderRetriever(checkpoint=checkpoint)
        r.load(index_dir)
        return r
    raise SystemExit(f"unknown retriever: {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", choices=["bm25", "biencoder"], default="bm25")
    ap.add_argument("--index", type=pathlib.Path, help="directory from BiEncoderRetriever.save()")
    ap.add_argument("--checkpoint", default="intfloat/multilingual-e5-base")
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--probes", type=pathlib.Path, default=PROBES)
    args = ap.parse_args()

    chunks = list(schema.read_jsonl(args.corpus))
    probes = load_probes(args.probes)
    by_id = {c.chunk_id: c for c in chunks}
    missing = [p["chunk_id"] for p in probes if p["chunk_id"] not in by_id]
    if missing:
        print(f"WARNING: {len(missing)} probe target(s) not in this corpus: {missing}")
        probes = [p for p in probes if p["chunk_id"] not in missing]

    retriever = build_retriever(args.retriever, chunks, args.index, args.checkpoint)
    label = args.checkpoint if args.retriever == "biencoder" else "bm25"
    print(f"retriever: {args.retriever}  ({label})\n")

    ranks: dict[str, int | None] = {}
    print(f"{'qid':10s} {'lang':8s} {'rank':>8s}  question")
    for p in probes:
        rank = rank_of(retriever, p["question_bn"], p["chunk_id"], len(chunks))
        ranks[p["qid"]] = rank
        print(f"{p['qid']:10s} {p['language']:8s} {rank!s:>8s}  {p['question_bn'][:50]}")

    print(f"\n{'cutoff':10s} {'overall':>10s} {'english':>10s} {'bengali':>10s}")
    by_lang = defaultdict(list)
    for p in probes:
        by_lang[p["language"]].append(ranks[p["qid"]])
    all_ranks = list(ranks.values())
    for k in CUTOFFS:
        def hit_rate(rs):
            rs = [r for r in rs if r is not None]
            return f"{sum(1 for r in rs if r <= k)}/{len(rs)}" if rs else "-"
        print(
            f"Recall@{k:<4d}"
            f"{hit_rate(all_ranks):>10s}"
            f"{hit_rate(by_lang.get('english', [])):>10s}"
            f"{hit_rate(by_lang.get('bengali', [])):>10s}"
        )

    print(
        "\nn=7 — a same-day sanity signal, not a reportable statistic. "
        "The real number comes from 09_run_eval.py against the adjudicated gold set."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
