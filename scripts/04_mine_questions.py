"""Fetch citizen legal questions from the cleared public sources.

    python scripts/04_mine_questions.py                      # all sources
    python scripts/04_mine_questions.py --source ajkerpatrika_aini_poramorsho
    python scripts/04_mine_questions.py --list

Re-running is cheap: every response is archived under data/raw/questions/ and
read from there on the next pass. Delete an archive directory to force a refetch.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import mine  # noqa: E402

OUT = pathlib.Path("data/interim/mined_articles.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", choices=sorted(mine.SOURCES))
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--list", action="store_true", help="list sources and exit")
    args = ap.parse_args()

    if args.list:
        for name in sorted(mine.SOURCES):
            print(name)
        return 0

    # The Prothom Alo column has run under several names and the searches
    # overlap heavily (one query's results were a strict subset of another's),
    # so the same article arrives more than once. First writer wins.
    items, seen = [], set()
    for item in mine.mine(args.source):
        if item.source_url in seen:
            continue
        seen.add(item.source_url)
        items.append(item)
    n = mine.write_jsonl(iter(items), args.out)

    per_source = Counter(i.source for i in items)
    print(f"{n} articles -> {args.out}")
    for source, count in sorted(per_source.items()):
        chars = sum(len(i.text) for i in items if i.source == source)
        print(f"  {source:36s} {count:4d} articles  {chars // max(count, 1):6d} avg chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
