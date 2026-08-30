"""Build the Dhara corpus from BLAD, and report what BLAD cannot supply.

    python scripts/03_build_corpus.py
    python scripts/03_build_corpus.py --gaps      # only list missing acts
    python scripts/03_build_corpus.py --qa        # corpus stats -> results/tables/

Writes data/processed/corpus_v1.jsonl. Freeze it once the QA gate passes; a
needed fix after that becomes corpus_v2.jsonl, never an edit in place.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import blad, schema  # noqa: E402

OUT = pathlib.Path("data/processed/corpus_v1.jsonl")
STATS = pathlib.Path("results/tables/corpus_stats.csv")


def report_gaps() -> None:
    gaps = blad.missing_acts()
    if not gaps:
        print("No gaps: BLAD supplies every act in configs/domains.yaml.")
        return
    print(f"\n{len(gaps)} acts in configs/domains.yaml that BLAD cannot supply:")
    print(f"{'domain':20s} {'reason':28s} act")
    for domain, fragment, reason in gaps:
        print(f"{domain:20s} {reason:28s} {fragment}")
    print("\nThese are the targets for the bdlaws pass.")


def qa(chunks: list[schema.Chunk]) -> None:
    STATS.parent.mkdir(parents=True, exist_ok=True)
    by_domain = Counter(c.domain for c in chunks)
    words = sorted(c.n_words for c in chunks)
    rows = [("metric", "value")]
    rows += [(f"chunks_{d}", str(n)) for d, n in sorted(by_domain.items())]
    rows += [
        ("chunks_total", str(len(chunks))),
        ("provisions_total", str(len({c.provision_id for c in chunks}))),
        ("acts_total", str(len({c.act_id for c in chunks}))),
        ("words_median", str(words[len(words) // 2] if words else 0)),
        ("words_p95", str(words[int(len(words) * 0.95)] if words else 0)),
        ("words_max", str(max(words) if words else 0)),
        ("chunks_under_20_words", str(sum(1 for w in words if w < 20))),
        ("chunks_over_400_words", str(sum(1 for w in words if w > 400))),
        ("chunks_missing_citation", str(sum(1 for c in chunks if not c.source_url))),
    ]
    STATS.write_text("\n".join(",".join(r) for r in rows), encoding="utf-8")
    print(f"\nwrote {STATS}")
    for key, value in rows[1:]:
        print(f"  {key:32s} {value}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument(
        "--all",
        action="store_true",
        help="ingest every act in BLAD, not just the everyday-life domains",
    )
    args = ap.parse_args()

    if args.gaps:
        report_gaps()
        return 0

    chunks, stats = blad.build(all_acts=args.all)
    # Fold in the acts BLAD leaves empty, fetched from bdlaws.
    extra, extra_stats = blad.build_from_bdlaws(all_acts=args.all)
    chunks += extra
    stats.update({f"bdlaws_{k}": v for k, v in extra_stats.items()})
    # BLAD has no provision titles; bdlaws does. Join them on.
    chunks, title_stats = blad.enrich_titles(chunks)
    stats.update({f"title_{k}": v for k, v in title_stats.items()})
    # Last, because it must see every chunk from every source: chunk_id is the
    # answer key the gold set points at and a duplicate makes an answer ambiguous.
    chunks, dedupe_stats = blad.dedupe_chunk_ids(chunks)
    stats.update(dedupe_stats)
    schema.write_jsonl(chunks, args.out)

    print(f"{len(chunks)} chunks -> {args.out}\n")
    for key, value in stats.items():
        print(f"  {key:38s} {value}")

    print(f"\n{'domain':22s} {'chunks':>7s} {'provisions':>11s}")
    by_domain = Counter(c.domain for c in chunks)
    for domain, count in sorted(by_domain.items(), key=lambda x: -x[1]):
        provisions = len({c.provision_id for c in chunks if c.domain == domain})
        print(f"{domain:22s} {count:7d} {provisions:11d}")

    report_gaps()
    if args.qa:
        qa(chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
