"""The §5.3 lexical-overlap audit. Decides which synthetic tiers are usable.

    python scripts/15_overlap_audit.py

Writes results/tables/overlap_audit.csv, results/runs/overlap_audit.json, and
results/figures/overlap_distributions.svg.

The question this answers is not "is the synthetic data good" in the abstract.
It is specific and falsifiable: **do synthetic questions sit at the same lexical
distance from their provision as real citizen questions do?**

If synthetic overlap is much higher than real overlap, then a model fine-tuned on
the synthetic set is being taught to match shared words — and the gold set, where
no such shared words exist, will not benefit. The fine-tuned model posts a good
training curve and a flat evaluation, and the project's premise goes untested.
The guide names this as the single most common cause of a null result here.

The real distribution comes from the 552 human-adjudicated probe questions
against their gold provisions. That is the distribution the training data has to
imitate, so it is the reference line in every comparison below.

Output is a per-tier verdict, not a single number, because the tiers were built
by different generators and there is no reason to expect them to pass or fail
together. A tier that fails is reported and excluded from the default training
recipe — not silently deleted, since "we generated 48,000 pairs and could only
use 6,000" is itself a finding about template-based generation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.synth import content_overlap, jaccard, overlap_summary  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
SYNTH = pathlib.Path("data/processed/synth_questions_v1.jsonl")
PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
OUT_TABLE = pathlib.Path("results/tables/overlap_audit.csv")
OUT_RUN = pathlib.Path("results/runs/overlap_audit.json")
OUT_FIG = pathlib.Path("results/figures/overlap_distributions.svg")

# A tier passes if its *median* pair is no more lexically overlapping than the
# most-overlapping decile of *real* pairs — i.e. tier median <= real p90.
#
# The first version of this test used real_median * (1 + tolerance), which
# degenerates here: the real median is exactly 0.0, so the threshold was 0.0 and
# the test was "is your median also exactly zero". That happened to give the
# right answer but for no defensible reason, and it would have failed a
# perfectly good generator whose median landed at 0.01. The p90 rule states the
# same intent in a form that survives a zero median.
TOLERANCE = 0.5  # retained as a documented knob; scales the p90 ceiling


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def passage_text(chunk) -> str:
    return f"{chunk.provision_title_bn or ''} {chunk.text_bn}".strip()


def histogram(values: list[float], bins: int = 20) -> list[int]:
    counts = [0] * bins
    for v in values:
        idx = min(int(v * bins), bins - 1)
        counts[idx] += 1
    return counts


def svg_figure(series: dict[str, list[float]], dest: pathlib.Path, bins: int = 20) -> None:
    """Overlap distributions on shared axes — the figure §5.3 asks to be in the report.

    Hand-rolled SVG rather than matplotlib: this is one chart with no runtime
    dependency worth adding, and an SVG drops straight into the paper.
    """
    width, height, pad = 720, 340, 48
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    colours = ["#1b6ac9", "#c9541b", "#2c8a4b", "#7a4bc9", "#a01b52"]

    normalized = {}
    for name, values in series.items():
        counts = histogram(values, bins)
        total = max(sum(counts), 1)
        normalized[name] = [c / total for c in counts]
    peak = max((max(v) for v in normalized.values()), default=1.0) or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui, sans-serif" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{pad}" y="22" font-size="13" font-weight="600">'
        f"Content-word overlap: question vs its provision</text>",
        f'<line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + plot_h}" stroke="#333"/>',
    ]
    for i in range(0, bins + 1, 4):
        x = pad + plot_w * i / bins
        parts.append(f'<line x1="{x:.1f}" y1="{pad + plot_h}" x2="{x:.1f}" y2="{pad + plot_h + 4}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{pad + plot_h + 17}" text-anchor="middle" fill="#444">{i / bins:.1f}</text>')
    parts.append(
        f'<text x="{pad + plot_w / 2:.0f}" y="{height - 6}" text-anchor="middle" fill="#444">'
        f"content overlap (fraction of question content words present in the provision)</text>"
    )

    for idx, (name, values) in enumerate(normalized.items()):
        colour = colours[idx % len(colours)]
        points = []
        for b, frac in enumerate(values):
            x = pad + plot_w * (b + 0.5) / bins
            y = pad + plot_h - plot_h * (frac / peak)
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{" ".join(points)}"/>'
        )
        ly = pad + 14 + idx * 16
        parts.append(f'<line x1="{pad + plot_w - 190}" y1="{ly - 4}" x2="{pad + plot_w - 170}" y2="{ly - 4}" stroke="{colour}" stroke-width="2"/>')
        parts.append(f'<text x="{pad + plot_w - 164}" y="{ly}" fill="#222">{name}</text>')

    parts.append("</svg>")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--synth", type=pathlib.Path, default=SYNTH)
    ap.add_argument("--probes", type=pathlib.Path, default=PROBES)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    # One pass over the corpus; it is 138 MB and we need two lookups from it.
    by_chunk: dict[str, str] = {}
    by_provision: dict[str, list[str]] = {}
    for chunk in schema.read_jsonl(args.corpus):
        text = passage_text(chunk)
        by_chunk[chunk.chunk_id] = text
        by_provision.setdefault(chunk.provision_id, []).append(text)

    # --- the reference distribution: real questions, human-adjudicated labels
    real_overlap, real_jaccard = [], []
    for probe in read_jsonl(args.probes):
        question = probe["question_bn"]
        # A provision may be several chunks; the question is compared against the
        # whole provision, since that is what the user would be shown.
        best = 0.0
        best_j = 0.0
        for provision_id in probe["gold_provision_ids"]:
            passage = " ".join(by_provision.get(provision_id, []))
            if not passage:
                continue
            best = max(best, content_overlap(question, passage))
            best_j = max(best_j, jaccard(question, passage))
        real_overlap.append(best)
        real_jaccard.append(best_j)

    real_median = statistics.median(real_overlap)
    print(f"real mined questions (n={len(real_overlap)}): {overlap_summary(real_overlap)}")

    # --- the synthetic tiers
    synth = read_jsonl(args.synth)
    by_tier: dict[str, list[float]] = collections.defaultdict(list)
    by_tier_j: dict[str, list[float]] = collections.defaultdict(list)
    for row in synth:
        # Recomputed here against the full provision, so the audit and the
        # generator cannot disagree about what was measured.
        passage = " ".join(by_provision.get(row["gold_provision_ids"][0], []))
        if not passage:
            continue
        by_tier[row["quality_tier"]].append(content_overlap(row["question_bn"], passage))
        by_tier_j[row["quality_tier"]].append(jaccard(row["question_bn"], passage))

    real_summary = overlap_summary(real_overlap)
    ceiling = real_summary["p90"] * (1 + args.tolerance)
    print(f"\npass threshold: median content overlap <= {ceiling:.4f} "
          f"(real p90 {real_summary['p90']:.4f} x {1 + args.tolerance:.2f}; "
          f"real median is {real_median:.4f})\n")

    verdicts = {}
    for tier in sorted(by_tier):
        summary = overlap_summary(by_tier[tier])
        passed = summary["median"] <= ceiling
        verdicts[tier] = {
            "content_overlap": summary,
            "jaccard": overlap_summary(by_tier_j[tier]),
            "median_vs_real": round(summary["median"] - real_median, 4),
            "verdict": "pass" if passed else "fail",
        }
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark}  {tier:18s} n={summary['n']:6d}  median={summary['median']:.4f}  "
              f"mean={summary['mean']:.4f}  p90={summary['p90']:.4f}")

    recommended = [t for t, v in verdicts.items() if v["verdict"] == "pass"]
    n_recommended = sum(verdicts[t]["content_overlap"]["n"] for t in recommended)
    print(f"\nrecommended training tiers: {recommended or '(none)'}  -> {n_recommended} pairs")

    svg_figure(
        {"real mined (gold)": real_overlap, **{t: by_tier[t] for t in sorted(by_tier)}},
        OUT_FIG,
    )
    print(f"wrote {OUT_FIG}")

    OUT_RUN.parent.mkdir(parents=True, exist_ok=True)
    OUT_RUN.write_text(
        json.dumps(
            {
                "run_id": "overlap_audit",
                "corpus": str(args.corpus),
                "synth": str(args.synth),
                "probes": str(args.probes),
                "tolerance": args.tolerance,
                "real": {
                    "content_overlap": overlap_summary(real_overlap),
                    "jaccard": overlap_summary(real_jaccard),
                },
                "pass_threshold_median": round(ceiling, 4),
                "tiers": verdicts,
                "recommended_tiers": recommended,
                "recommended_pairs": n_recommended,
                "per_query": [
                    {"qid": p["qid"], "content_overlap": round(o, 4)}
                    for p, o in zip(read_jsonl(args.probes), real_overlap)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT_RUN}")

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["set", "n", "mean", "p25", "median", "p75", "p90", "frac_over_0.3", "verdict"])
        r = overlap_summary(real_overlap)
        writer.writerow(["real_mined_gold", r["n"], r["mean"], r["p25"], r["median"], r["p75"], r["p90"], r["frac_over_0.3"], "reference"])
        for tier in sorted(by_tier):
            s = verdicts[tier]["content_overlap"]
            writer.writerow([tier, s["n"], s["mean"], s["p25"], s["median"], s["p75"], s["p90"], s["frac_over_0.3"], verdicts[tier]["verdict"]])
    print(f"wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
