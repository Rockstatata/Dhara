"""Build the retrieval-time exclusion list for repealed and omitted provisions.

    python scripts/19_build_exclusions.py

Writes data/processed/excluded_repealed_v1.jsonl and
results/tables/exclusion_stats.csv.

## Why this is an ethics artifact and not a data-cleaning chore

The project's ethics constraints say repealed and omitted sections are dropped
from the corpus, because retrieving one and showing it to a citizen as the law
that governs their situation is actual harm — they would be reading a rule that
no longer exists, in a tool that presents itself as authoritative.

`corpus_v1.jsonl` was frozen with 561 of them still in it. The freeze rule says
corpus_v1 never changes after Week 3 and a needed fix becomes corpus_v2, so the
right move is not to quietly edit the frozen file. This script instead produces
an explicit exclusion list that the index build and the serving path both apply,
and records the count so it can be reported rather than discovered by a reviewer.

## What counts as repealed, and one thing that does not

A provision is repealed if its own title or body *begins* with a repeal marker:
`[Repealed]`, `[Omitted]`, `রহিত করা হইয়াছে`, `বিলুপ্ত`.

It is **not** repealed merely because the word রহিত appears in it. Sections
titled `রহিতকরণ ও হেফাজত` ("Repeal and savings") are the sections that repeal
*other* statutes; they are live, operative law. A first pass at this count used
a loose prefix match and reported 887 — 326 of those were repeal-and-savings
sections and constitutional provisions about the effect of repealed laws, all of
them perfectly valid retrieval targets. The stricter rule in `synth.is_repealed`
gives 561. The looser number is recorded below too, so the difference between
the two is visible rather than being a silently corrected mistake.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.synth import is_repealed  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
OUT = pathlib.Path("data/processed/excluded_repealed_v1.jsonl")
STATS = pathlib.Path("results/tables/exclusion_stats.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    gold_provisions = set()
    if PROBES.exists():
        with PROBES.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    gold_provisions.update(json.loads(line)["gold_provision_ids"])

    excluded = []
    by_domain: collections.Counter = collections.Counter()
    by_language: collections.Counter = collections.Counter()
    total = 0
    for chunk in schema.read_jsonl(args.corpus):
        total += 1
        if not is_repealed(chunk):
            continue
        excluded.append({
            "chunk_id": chunk.chunk_id,
            "provision_id": chunk.provision_id,
            "act_id": chunk.act_id,
            "act_title": chunk.act_title_bn or chunk.act_title_en,
            "provision_no": chunk.provision_no_ascii,
            "title": chunk.provision_title_bn,
            "reason": "repealed_or_omitted",
            "excerpt": chunk.text_bn[:120],
        })
        by_domain[chunk.domain] += 1
        by_language[chunk.language] += 1

    # If a gold answer were repealed, the gold set itself would be wrong and the
    # exclusion would silently make that question unanswerable. Check, loudly.
    collisions = [e for e in excluded if e["provision_id"] in gold_provisions]
    assert not collisions, (
        f"{len(collisions)} repealed provisions are gold answers, e.g. "
        f"{[c['chunk_id'] for c in collisions[:5]]} — the gold set needs fixing, not the index"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in excluded:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    pct = 100 * len(excluded) / max(total, 1)
    print(f"corpus            : {total} chunks")
    print(f"repealed/omitted  : {len(excluded)} ({pct:.2f}%)")
    print(f"gold collisions   : {len(collisions)} (asserted 0)")
    print(f"by domain         : {dict(by_domain.most_common(8))}")
    print(f"by language       : {dict(by_language)}")
    print(f"wrote {args.out}")

    STATS.parent.mkdir(parents=True, exist_ok=True)
    with STATS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["corpus_chunks", total])
        writer.writerow(["excluded_repealed_omitted", len(excluded)])
        writer.writerow(["excluded_pct", round(pct, 3)])
        writer.writerow(["gold_collisions", len(collisions)])
        for domain, n in by_domain.most_common():
            writer.writerow([f"domain_{domain}", n])
        for language, n in by_language.most_common():
            writer.writerow([f"language_{language}", n])
    print(f"wrote {STATS}")


if __name__ == "__main__":
    main()
