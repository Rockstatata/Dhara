"""Fetch from bdlaws what BLAD cannot supply.

    python scripts/02_scrape.py --gaps          # the 203 acts BLAD leaves empty
    python scripts/02_scrape.py --acts 1457 1261
    python scripts/02_scrape.py --gaps --limit 5

Resumable: everything is archived under data/raw/bdlaws/ and re-read from there,
so an interrupted crawl costs only the pages it had not reached.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import blad, scrape  # noqa: E402

OUT = pathlib.Path("data/interim/bdlaws_provisions.jsonl")


def gap_act_ids() -> list[str]:
    """Acts BLAD carries with zero provisions — its `language: unknown` rows."""
    ids = []
    for act in blad.acts():
        if act.get("sections"):
            continue
        act_id = blad._act_id(act.get("source_url", ""))
        if act_id:
            ids.append(act_id)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", nargs="*", default=None)
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    act_ids = list(args.acts) if args.acts else (gap_act_ids() if args.gaps else [])
    if not act_ids:
        ap.error("pass --gaps or --acts")
    if args.limit:
        act_ids = act_ids[: args.limit]

    print(f"fetching {len(act_ids)} acts from bdlaws\n")
    n = scrape.write_jsonl(scrape.acts(act_ids), args.out)
    print(f"\n{n} provisions -> {args.out}")

    if n:
        rows = [json.loads(line) for line in args.out.open(encoding="utf-8")]
        with_title = sum(1 for r in rows if r["title_bn"])
        print(f"  acts with content   {len({r['act_id'] for r in rows})}/{len(act_ids)}")
        print(f"  provisions w/ title {with_title} ({100 * with_title // len(rows)}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
