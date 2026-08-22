"""Turn mined articles into individual citizen questions.

    python scripts/05_split_questions.py
    python scripts/05_split_questions.py --sample 5      # eyeball the output

Writes data/interim/questions_pool.jsonl and prints a per-source breakdown with
split confidence, so a bad parser shows up as a count rather than as silence.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import questions as Q  # noqa: E402

IN = pathlib.Path("data/interim/mined_articles.jsonl")
OUT = pathlib.Path("data/interim/questions_pool.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=pathlib.Path, default=IN)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--sample", type=int, default=0, help="print N random questions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    articles = [json.loads(line) for line in args.src.open(encoding="utf-8")]
    pool = Q.split_all(articles)
    Q.write_jsonl(pool, args.out)

    print(f"{len(articles)} articles -> {len(pool)} questions -> {args.out}\n")
    by_source = Counter(q.source for q in pool)
    print(f"{'source':34s} {'qs':>5s} {'high':>6s} {'med':>5s} {'low':>5s} {'wds':>5s}")
    for source, count in sorted(by_source.items()):
        rows = [q for q in pool if q.source == source]
        conf = Counter(q.split_confidence for q in rows)
        median = sorted(q.n_words for q in rows)[len(rows) // 2]
        print(
            f"{source:34s} {count:5d} {conf['high']:6d} {conf['medium']:5d} "
            f"{conf['low']:5d} {median:5d}"
        )

    if args.sample:
        random.seed(args.seed)
        print("\n--- sample ---")
        for q in random.sample(pool, min(args.sample, len(pool))):
            print(f"\n[{q.qid}] {q.source} ({q.split_confidence}, {q.n_words}w)")
            print(q.question_bn[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
