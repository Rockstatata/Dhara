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
import os
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
    ap.add_argument(
        "--force", action="store_true",
        help="ignore a stale lock left behind by a crawl that was killed",
    )
    args = ap.parse_args()

    # Strip whitespace: act ids often arrive from a file via the shell, and on
    # Windows a stray carriage return rides along and lands in the archive path.
    act_ids = [a.strip() for a in (args.acts or []) if a.strip()]
    if not act_ids and args.gaps:
        act_ids = gap_act_ids()
    if not act_ids:
        ap.error("pass --gaps or --acts")
    if args.limit:
        act_ids = act_ids[: args.limit]

    # A second crawler started against the same --out opens it in write mode and
    # truncates it while the first still holds a handle at a stale offset, so the
    # JSONL interleaves and its line count can go *down* between two reads. Worse,
    # both processes fetch the same pages, doubling the request rate on bdlaws
    # past the 1.5-2s the scraping-conduct rule commits us to. This happened on
    # 2026-08-25; see DECISIONS.md. Cheap to prevent, expensive to notice.
    lock = args.out.with_name(args.out.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and not args.force:
        print(
            f"{lock} exists - another crawl is already writing {args.out}. "
            f"If you are certain nothing is running, delete the lock or pass --force."
        )
        return 2
    lock.write_text(str(os.getpid()), encoding="utf-8")

    print(f"fetching {len(act_ids)} acts from bdlaws\n")
    try:
        n = scrape.write_jsonl(scrape.acts(act_ids), args.out)
    finally:
        lock.unlink(missing_ok=True)
    print(f"\n{n} provisions -> {args.out}")

    if n:
        rows = [json.loads(line) for line in args.out.open(encoding="utf-8")]
        with_title = sum(1 for r in rows if r["title_bn"])
        print(f"  acts with content   {len({r['act_id'] for r in rows})}/{len(act_ids)}")
        print(f"  provisions w/ title {with_title} ({100 * with_title // len(rows)}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
