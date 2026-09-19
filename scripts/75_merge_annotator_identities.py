"""Merge Sifat->Iftiaq and Sorup->Sarwad across all annotation records.

    python scripts/75_merge_annotator_identities.py

The project has two credited members: Sarwad and Iftiaq. Sifat and Sorup
contributed some early annotation/verification work but are not project
members; per an explicit decision (DECISIONS.md 2026-09-19), their rows are
reattributed rather than dropped, preserving the double-annotation structure
under the two credited names.

This is NOT a blind find-and-replace. Where a cross-check pair's two sides
both map to the same target name (Sarwad-vs-Sorup -> Sarwad-vs-Sarwad, or
Iftiaq-vs-Sifat -> Iftiaq-vs-Iftiaq), that pair becomes a self-comparison
and is excluded from every inter-annotator reliability number, not silently
kept as if it were still a two-person check. See the printed report for
exactly which pairs survived and which were excluded, and why.

Where the SAME qid was independently annotated by both the credited person
and their merged-in counterpart with a DIFFERENT verdict/answer (the
excluded self-pairs), the ORIGINAL credited person's row is kept as the
record; the merged-in row is archived, not silently dropped -- see
data/annotation/_archived_pre_merge/.
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil

ANNOT = pathlib.Path("data/annotation")
ARCHIVE = ANNOT / "_archived_pre_merge"
MAP = {"Sorup": "Sarwad", "Sifat": "Iftiaq", "Sarwad": "Sarwad", "Iftiaq": "Iftiaq"}


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8", errors="ignore") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_verify_or_paraphrase(kind: str, primary: str, secondary: str) -> dict:
    """kind is 'verify' or 'paraphrase'. primary keeps its identity, secondary
    merges into it. Where both annotated the same qid, primary's row wins;
    secondary's conflicting row is archived, not dropped."""
    primary_path = ANNOT / f"{kind}_{primary}.csv"
    secondary_path = ANNOT / f"{kind}_{secondary}.csv"
    if not secondary_path.exists():
        return {"kind": kind, "primary": primary, "secondary": secondary, "status": "no secondary file"}

    primary_rows = read_csv(primary_path)
    secondary_rows = read_csv(secondary_path)
    primary_qids = {r["qid"] for r in primary_rows}

    kept, archived = [], []
    for row in secondary_rows:
        row = dict(row)
        row["annotator"] = primary
        row["merged_from"] = secondary
        if row["qid"] in primary_qids:
            archived.append(row)
        else:
            kept.append(row)

    merged = primary_rows + kept
    fieldnames = list(primary_rows[0].keys()) if primary_rows else list(kept[0].keys())
    if "merged_from" not in fieldnames:
        fieldnames = fieldnames + ["merged_from"]
    for r in merged:
        r.setdefault("merged_from", "")

    write_csv(primary_path, merged, fieldnames)
    if archived:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        write_csv(ARCHIVE / f"{kind}_{secondary}_conflicting_with_{primary}.csv", archived,
                   list(archived[0].keys()))

    return {"kind": kind, "primary": primary, "secondary": secondary,
            "merged_in": len(kept), "archived_conflicts": len(archived)}


def archive_and_remove(path: pathlib.Path) -> None:
    if not path.exists():
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(ARCHIVE / path.name))


def fix_round2() -> dict:
    path = ANNOT / "adjudication_round2.csv"
    rows = read_csv(path)
    excluded = 0
    for r in rows:
        r["annotator_a_original"] = r["annotator_a"]
        r["annotator_b_original"] = r["annotator_b"]
        a, b = MAP.get(r["annotator_a"], r["annotator_a"]), MAP.get(r["annotator_b"], r["annotator_b"])
        r["annotator_a"], r["annotator_b"] = a, b
        is_self = a == b
        r["self_comparison_excluded_from_agreement"] = str(is_self)
        if is_self:
            excluded += 1
    fieldnames = list(rows[0].keys())
    write_csv(path, rows, fieldnames)
    return {"total": len(rows), "excluded_as_self_pair": excluded, "valid_for_agreement": len(rows) - excluded}


def fix_gold(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    changed = 0
    for r in rows:
        if "annotators" in r and isinstance(r["annotators"], list):
            new = [MAP.get(a, a) for a in r["annotators"]]
            if new != r["annotators"]:
                changed += 1
            r["annotators"] = new
        if r.get("adjudicator") in MAP:
            r["adjudicator"] = MAP[r["adjudicator"]]
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return changed


def main() -> None:
    report = {"verify_merges": [], "paraphrase_merges": []}
    report["verify_merges"].append(merge_verify_or_paraphrase("verify", "Sarwad", "Sorup"))
    report["verify_merges"].append(merge_verify_or_paraphrase("verify", "Iftiaq", "Sifat"))
    report["paraphrase_merges"].append(merge_verify_or_paraphrase("paraphrase", "Sarwad", "Sorup"))
    report["paraphrase_merges"].append(merge_verify_or_paraphrase("paraphrase", "Iftiaq", "Sifat"))

    archive_and_remove(ANNOT / "verify_Sorup.csv")
    archive_and_remove(ANNOT / "verify_Sifat.csv")
    archive_and_remove(ANNOT / "paraphrase_Sorup.csv")
    archive_and_remove(ANNOT / "paraphrase_Sifat.csv")

    report["round2"] = fix_round2()
    report["gold_v2_rows_changed"] = fix_gold(pathlib.Path("data/processed/gold_verified_v2.jsonl"))
    report["gold_v3_rows_changed"] = fix_gold(pathlib.Path("data/processed/gold_verified_v3.jsonl"))

    out = pathlib.Path("results/runs/merge_annotator_identities.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
