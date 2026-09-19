"""Audit the gold set for rubber-stamping: how much of it is a human label?

    python scripts/24_gold_audit.py
    python scripts/24_gold_audit.py --sheets data/annotation

Writes results/runs/gold_audit.json and results/tables/gold_audit.csv.

## Why this exists

`scripts/09_autolabel_pass1.py` fills every sheet with a machine proposal so a
human verifies rather than authors from scratch. Its own docstring names the
risk that creates:

    If a retriever's proposals survive into the gold set unchanged, the
    evaluation compares a retriever against a retriever. `pass1_auto` exists so
    that the fraction of machine labels a human changed is measurable and
    reportable; if that fraction is near zero, the labels were rubber-stamped
    and the evaluation should be described accordingly.

Nothing was measuring it. This does.

## What it checks, and what each answer means

**Human-edit evidence.** `answer_source` filled, `verified` filled, `notes`
written by a person rather than carried over from the machine pass. All three
empty across a sheet means no recorded human action on those rows — which is not
proof that no thinking happened, but it is the only evidence the pipeline keeps,
and a paper cannot claim adjudication it cannot show.

**Independence of the double-annotated block.** The OVERLAP block exists to
yield an inter-annotator kappa. Kappa is only meaningful if the two annotators
could disagree. If the sheets were distributed with identical machine answers
already filled in and both were returned unchanged, agreement is 1.0 by
construction and reporting it as kappa would be a fabrication. This script
reports the disagreement count, and refuses to compute kappa when it is zero
across every pair — a zero there means the block measured nothing.

**How much a human changed** -- when `--verified` points at the sheets that came
back from the annotators and `--sheets` at the machine-filled copies they started
from, every differing `answer` cell is one label a person overrode. That fraction
is the number `09_autolabel_pass1.py` asked for, and it decides whether the
evaluation set is adjudicated or rubber-stamped.

**Agreement, on the labels where agreement is defined.** Cohen's kappa needs
categorical decisions, so it is computed on `verdict`, `register` and `domain`.
The provision set is open-ended -- an annotator may pick any subset of 39,484
chunks -- so kappa does not apply to it, and reporting one would be an abuse of
the statistic. Provision agreement is reported as exact-set match, at-least-one
overlap, and mean Jaccard instead.

**External validation where it exists.** These questions come from newspaper
legal-advice columns, so a lawyer's published answer sits alongside many of
them. Where that answer names a statute whose title appears in the corpus, the
labelled provision either belongs to that Act or it does not. Coverage is
partial and the matching is by title string, so this is a floor on quality
rather than a verdict — but it is the one independent signal available without
new human work.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.normalize import aggressive  # noqa: E402

GOLD = pathlib.Path("data/processed/gold_test_v1.jsonl")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
SHEETS = pathlib.Path("data/annotation")
OUT_RUN = pathlib.Path("results/runs/gold_audit.json")
OUT_TABLE = pathlib.Path("results/tables/gold_audit.csv")

# Act titles short enough to be ambiguous as substrings match half the corpus;
# a title has to be distinctive to count as the lawyer naming a law.
MIN_TITLE_WORDS = 2


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_sheets(dirpath: pathlib.Path, pattern: str = "annotate_*.csv") -> list[dict]:
    rows = []
    for path in sorted(dirpath.glob(pattern)):
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                row["_sheet"] = path.name
                rows.append(row)
    return rows


def edit_evidence(rows: list[dict]) -> dict:
    """Every trace a human action would have left in the sheet."""
    filled = lambda key: sum(1 for r in rows if (r.get(key) or "").strip())
    human_notes = sum(
        1 for r in rows
        if (r.get("notes") or "").strip() and not (r.get("notes") or "").startswith("[llm-assist]")
    )
    return {
        "rows": len(rows),
        "pass1_auto": sum(1 for r in rows if (r.get("pass1_auto") or "").strip() == "1"),
        "answer_source_filled": filled("answer_source"),
        "answer_source_values": dict(collections.Counter(
            (r.get("answer_source") or "").strip() or "(empty)" for r in rows)),
        "verified_filled": filled("verified"),
        "human_written_notes": human_notes,
        "paraphrase_filled": filled("paraphrase"),
        "formal_version_filled": filled("formal_version"),
        "empty_answers": sum(1 for r in rows if not (r.get("answer") or "").strip()),
    }


def overlap_independence(rows: list[dict]) -> dict:
    """Do the double-annotated rows actually differ anywhere?"""
    by_qid: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_qid[r["qid"]].append(r)
    shared = {q: rs for q, rs in by_qid.items() if len(rs) > 1}

    pairs = differing = 0
    examples = []
    for qid, rs in shared.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                pairs += 1
                a = (rs[i].get("answer") or "").strip()
                b = (rs[j].get("answer") or "").strip()
                if a != b:
                    differing += 1
                    if len(examples) < 5:
                        examples.append({"qid": qid,
                                         "a": rs[i]["annotator"], "a_answer": a[:60],
                                         "b": rs[j]["annotator"], "b_answer": b[:60]})
    return {
        "double_annotated_qids": len(shared),
        "annotator_pairs_compared": pairs,
        "pairs_with_differing_answers": differing,
        "blocks": dict(collections.Counter(rs[0].get("block", "?") for rs in shared.values())),
        "examples": examples,
        "kappa_computable": differing > 0,
        "note": (
            "Zero differing pairs means the sheets were returned exactly as the machine "
            "pass filled them. Agreement is then 1.0 by construction and kappa measures "
            "nothing; it must not be reported."
        ),
    }


def act_title_index(chunks) -> dict[str, str]:
    """Normalized Bangla act title -> act_id, for titles distinctive enough to match."""
    index: dict[str, str] = {}
    for chunk in chunks:
        title = (chunk.act_title_bn or "").strip()
        if not title:
            continue
        norm = aggressive(title)
        # Drop the year suffix so "দণ্ডবিধি, ১৮৬০" also matches a bare mention.
        norm = re.sub(r"[,\s]*\d{4}\s*$", "", norm).strip()
        if len(norm.split()) >= MIN_TITLE_WORDS:
            index.setdefault(norm, chunk.act_id)
    return index


def lawyer_check(rows: list[dict], gold: list[dict], titles: dict[str, str],
                 act_of: dict[str, str]) -> dict:
    """Where the lawyer's published answer names an Act, is the label in it?"""
    answer_by_qid = {r["qid"]: (r.get("lawyer_answer") or "") for r in rows}
    matched = mismatched = 0
    no_title_found = 0
    examples = []

    for row in gold:
        prose = answer_by_qid.get(row["qid"], "")
        if not prose.strip():
            continue
        norm_prose = aggressive(prose)
        named = {act_id for title, act_id in titles.items() if title in norm_prose}
        if not named:
            no_title_found += 1
            continue
        labelled_acts = {act_of[p] for p in row.get("relevant_provision_ids") or [] if p in act_of}
        if labelled_acts & named:
            matched += 1
        else:
            mismatched += 1
            if len(examples) < 5:
                examples.append({"qid": row["qid"],
                                 "lawyer_named_acts": sorted(named)[:3],
                                 "labelled_acts": sorted(labelled_acts)[:3]})
    checked = matched + mismatched
    return {
        "gold_rows_with_lawyer_answer": sum(1 for r in gold if answer_by_qid.get(r["qid"], "").strip()),
        "lawyer_answer_names_a_corpus_act": checked,
        "no_recognisable_act_title": no_title_found,
        "label_act_matches_lawyer": matched,
        "label_act_differs": mismatched,
        "agreement_rate": round(matched / checked, 4) if checked else None,
        "examples_of_mismatch": examples,
        "note": (
            "Title-substring matching over a corpus of 1,227 Acts, so coverage is partial "
            "and a miss is not necessarily a wrong label — the lawyer may name no law, or "
            "name it in words that are not its printed title. Read the agreement rate as a "
            "floor on label quality, not a verdict."
        ),
    }


def cohen_kappa(a, b):
    """Two raters, same items, categorical labels.

    Returns None when the raters used only one category between them: kappa is
    undefined there (expected agreement is 1, so the denominator vanishes), and a
    0.0 in that slot would read as total disagreement rather than "not
    applicable".
    """
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return None
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def parse_ids(cell):
    return {piece.strip() for piece in (cell or "").replace(";", ",").split(",") if piece.strip()}


def agreement(rows):
    """Pairwise agreement over every question two annotators both answered."""
    by_qid = collections.defaultdict(list)
    for r in rows:
        by_qid[r["qid"]].append(r)
    shared = {q: rs for q, rs in by_qid.items() if len(rs) > 1}
    if not shared:
        return {"double_annotated_qids": 0, "note": "no question was annotated twice"}

    pairs = collections.defaultdict(lambda: {"a": [], "b": []})
    for qid, rs in shared.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                key = tuple(sorted((rs[i]["annotator"], rs[j]["annotator"])))
                first, second = (rs[i], rs[j]) if rs[i]["annotator"] == key[0] else (rs[j], rs[i])
                pairs[key]["a"].append(first)
                pairs[key]["b"].append(second)

    out = {}
    for key, data in sorted(pairs.items()):
        a_rows, b_rows = data["a"], data["b"]
        entry = {"n": len(a_rows)}
        for field in ("verdict", "register", "domain"):
            a = [(r.get(field) or "").strip() for r in a_rows]
            b = [(r.get(field) or "").strip() for r in b_rows]
            entry[field + "_raw_agreement"] = round(
                sum(1 for x, y in zip(a, b) if x == y) / len(a), 4)
            entry[field + "_kappa"] = cohen_kappa(a, b)

        exact = overlap = 0
        jaccards = []
        for x, y in zip(a_rows, b_rows):
            sa, sb = parse_ids(x.get("answer", "")), parse_ids(y.get("answer", ""))
            if sa == sb:
                exact += 1
            if sa & sb:
                overlap += 1
            union = sa | sb
            jaccards.append(len(sa & sb) / len(union) if union else 1.0)
        entry["provision_exact_match"] = round(exact / len(a_rows), 4)
        entry["provision_any_overlap"] = round(overlap / len(a_rows), 4)
        entry["provision_mean_jaccard"] = round(sum(jaccards) / len(jaccards), 4)
        out["+".join(key)] = entry

    return {"double_annotated_qids": len(shared), "pairs": out}


def edit_rate(baseline, verified):
    """How often a human overrode the machine proposal, per annotator."""
    base = {(r["qid"], r["annotator"]): r for r in baseline}
    per_annotator = collections.defaultdict(lambda: {"compared": 0, "changed": 0, "examples": []})
    compared = changed = missing = 0

    for row in verified:
        key = (row["qid"], row["annotator"])
        if key not in base:
            missing += 1
            continue
        before = parse_ids(base[key].get("answer", ""))
        after = parse_ids(row.get("answer", ""))
        compared += 1
        stats = per_annotator[row["annotator"]]
        stats["compared"] += 1
        if before != after:
            changed += 1
            stats["changed"] += 1
            if len(stats["examples"]) < 3:
                stats["examples"].append({"qid": row["qid"],
                                          "machine": sorted(before)[:3],
                                          "human": sorted(after)[:3]})
    for stats in per_annotator.values():
        stats["edit_rate"] = round(stats["changed"] / stats["compared"], 4) if stats["compared"] else None

    return {
        "rows_compared": compared,
        "rows_changed": changed,
        "edit_rate": round(changed / compared, 4) if compared else None,
        "rows_with_no_baseline_match": missing,
        "per_annotator": dict(per_annotator),
        "note": (
            "The fraction of machine proposals a human overrode. Near zero means the "
            "labels were accepted as-is and the evaluation compares a retriever against a "
            "retriever; scripts/09_autolabel_pass1.py exists to make this measurable."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=pathlib.Path, default=GOLD)
    ap.add_argument("--sheets", type=pathlib.Path, default=SHEETS)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--verified", type=pathlib.Path, default=None,
                    help="directory of sheets returned by the annotators; compared against "
                         "--sheets (the machine-filled copies) to get the edit rate")
    ap.add_argument("--pattern", default="annotate_*.csv",
                    help="glob for the sheet files (exported Google Sheets rarely keep the name)")
    args = ap.parse_args()

    gold = read_jsonl(args.gold)
    rows = read_sheets(args.sheets, args.pattern)
    assert rows, f"no {args.pattern} found under {args.sheets}"

    verified_rows = []
    if args.verified:
        verified_rows = read_sheets(args.verified, args.pattern)
        assert verified_rows, f"no {args.pattern} found under {args.verified}"
        print(f"comparing {len(verified_rows)} returned rows against "
              f"{len(rows)} machine-filled rows\n")

    chunks = list(schema.read_jsonl(args.corpus))
    act_of = {c.provision_id: c.act_id for c in chunks}
    titles = act_title_index(chunks)

    report = {
        "run_id": "gold_audit",
        "gold_file": str(args.gold),
        "gold_rows": len(gold),
        "distinct_questions": len({r["qid"] for r in gold}),
        "annotators": dict(collections.Counter(r["annotator"] for r in gold)),
        "verdicts": dict(collections.Counter(r["verdict"] for r in gold)),
        "annotation_modes": dict(collections.Counter(r.get("annotation_mode") for r in gold)),
        "edit_evidence": edit_evidence(rows),
        "overlap": overlap_independence(rows),
        "lawyer_cross_check": lawyer_check(rows, gold, titles, act_of),
        "agreement": agreement(verified_rows or rows),
        "act_titles_indexed": len(titles),
    }

    ev = report["edit_evidence"]
    ov = report["overlap"]
    lw = report["lawyer_cross_check"]

    print(f"gold rows {report['gold_rows']} over {report['distinct_questions']} questions, "
          f"{len(report['annotators'])} annotators")
    print("\nhuman-edit evidence in the returned sheets")
    print(f"  machine-filled (pass1_auto=1)   : {ev['pass1_auto']}/{ev['rows']}")
    print(f"  answer_source filled by a human : {ev['answer_source_filled']}  {ev['answer_source_values']}")
    print(f"  verified column filled          : {ev['verified_filled']}")
    print(f"  notes written by a human        : {ev['human_written_notes']}")
    print(f"  paraphrase / formal_version     : {ev['paraphrase_filled']} / {ev['formal_version_filled']}")

    print("\ndouble-annotated block")
    print(f"  qids annotated more than once   : {ov['double_annotated_qids']}")
    print(f"  annotator pairs compared        : {ov['annotator_pairs_compared']}")
    print(f"  pairs whose answers differ      : {ov['pairs_with_differing_answers']}")
    print(f"  kappa computable                : {ov['kappa_computable']}")

    print("\ncross-check against the lawyer's published answer")
    print(f"  gold rows with a lawyer answer  : {lw['gold_rows_with_lawyer_answer']}")
    print(f"  answer names a corpus Act       : {lw['lawyer_answer_names_a_corpus_act']}")
    print(f"  label's Act matches             : {lw['label_act_matches_lawyer']}")
    print(f"  label's Act differs             : {lw['label_act_differs']}")
    if lw["agreement_rate"] is not None:
        print(f"  agreement rate                  : {lw['agreement_rate']:.3f}")

    verdicts = []
    if ev["answer_source_filled"] + ev["verified_filled"] + ev["human_written_notes"] == 0:
        verdicts.append(
            "NO recorded human edit anywhere in the sheets. The gold set is a machine "
            "proposal set. Every reported number is measured against it, so the paper "
            "must either describe it that way or the labels must be adjudicated."
        )
    if not ov["kappa_computable"]:
        verdicts.append(
            "Inter-annotator agreement is NOT measurable from these files: no double-"
            "annotated pair differs, which is what identical pre-filled answers produce."
        )
    if verdicts:
        print("\nFINDINGS")
        for v in verdicts:
            print(f"  - {v}")
    report["findings"] = verdicts

    OUT_RUN.parent.mkdir(parents=True, exist_ok=True)
    OUT_RUN.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, value in (
            ("gold_rows", report["gold_rows"]),
            ("distinct_questions", report["distinct_questions"]),
            ("pass1_auto_rows", ev["pass1_auto"]),
            ("answer_source_filled", ev["answer_source_filled"]),
            ("verified_filled", ev["verified_filled"]),
            ("human_written_notes", ev["human_written_notes"]),
            ("double_annotated_qids", ov["double_annotated_qids"]),
            ("pairs_with_differing_answers", ov["pairs_with_differing_answers"]),
            ("lawyer_checked", lw["lawyer_answer_names_a_corpus_act"]),
            ("lawyer_agreement_rate", lw["agreement_rate"]),
        ):
            writer.writerow([key, value])

    if verified_rows:
        report["edit_rate"] = edit_rate(rows, verified_rows)
        er = report["edit_rate"]
        print("\nhuman edits over the machine proposals")
        print(f"  rows compared                   : {er['rows_compared']}")
        rate = f"  ({er['edit_rate']:.1%})" if er["edit_rate"] is not None else ""
        print(f"  rows a human changed            : {er['rows_changed']}{rate}")
        for name, stats in sorted(er["per_annotator"].items()):
            shown = f"{stats['edit_rate']:.1%}" if stats["edit_rate"] is not None else "n/a"
            print(f"    {name:10s} {stats['changed']:4d}/{stats['compared']:4d}  {shown}")

    ag = report["agreement"]
    if ag.get("pairs"):
        print("\ninter-annotator agreement (double-annotated questions only)")
        for pair, e in ag["pairs"].items():
            kappa = e["verdict_kappa"]
            kappa_s = f"{kappa:+.3f}" if kappa is not None else "undefined (one category)"
            print(f"  {pair:22s} n={e['n']:3d}  verdict kappa {kappa_s}  "
                  f"provision exact {e['provision_exact_match']:.3f}  "
                  f"any-overlap {e['provision_any_overlap']:.3f}  "
                  f"Jaccard {e['provision_mean_jaccard']:.3f}")

    print(f"\nwrote {OUT_RUN}")
    print(f"wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
