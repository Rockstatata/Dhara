"""Prepare, validate, and apply Dhara Round-2 annotation corrections.

Run from the repository root:

    python scripts/31_finalize_annotation_round2.py --prepare
    python scripts/31_finalize_annotation_round2.py --dry-run
    python scripts/31_finalize_annotation_round2.py --apply

``--prepare`` writes the filled correction/review/adjudication CSVs but does
not touch working sheets or versioned outputs.  ``--dry-run`` proposes every
change in memory and prints all release gates.  ``--apply`` refuses to write if
any gate fails, makes a timestamped archive, applies edits by qid, and writes
the v2 JSONL and audit artifacts.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.annotation_round2 import Round2Finalizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--decisions",
        type=pathlib.Path,
        default=pathlib.Path("data/annotation/round2_llm_decisions.json"),
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    decisions = args.decisions if args.decisions.is_absolute() else root / args.decisions
    finalizer = Round2Finalizer(root, decisions)
    finalizer.apply_in_memory()

    if args.prepare:
        finalizer.write_support_files()
        print("wrote filled correction, review, and adjudication CSVs")
        return 0

    summary, metric_rows = finalizer.audit()
    failures = finalizer.gates(summary)
    finalizer.print_report(summary, failures)
    if args.dry_run:
        return 1 if failures else 0
    if failures:
        print("No files were changed because one or more release gates failed.")
        return 1

    archive = finalizer.backup_working_files()
    finalizer.write_support_files()
    finalizer.write_working_files()
    finalizer.write_release_artifacts(summary, metric_rows)
    print(f"archive: {archive}")
    print("wrote gold_verified_v2.jsonl, questions_release_v2.jsonl, and audit artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
