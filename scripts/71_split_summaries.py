"""Split the provision-summarization dataset into train/validation/test,
by Act, per the user's spec: "Split by Act, not randomly... No provision,
near-duplicate, or same Act-family content should leak across splits."

    python scripts/71_split_summaries.py

Reads `data/processed/summaries_v1/merged.jsonl`, assigns every row's whole
Act to exactly one split (never splits one Act's rows across two files),
and writes `summaries_train_v1.jsonl` / `summaries_val_v1.jsonl` /
`summaries_test_v1.jsonl` plus a manifest recording which Act went where
and why - so a re-run is auditable rather than a re-shuffle.

Assignment is deterministic and reproducible (no RNG, no seed to lose):
Acts are sorted by row count descending and each is greedily placed into
whichever split is currently furthest below its 80/10/10 row-count target
(longest-processing-time-first bin balancing). This targets the row-count
ratio the spec asks for while still keeping every row of an Act on one
side of the split - the two goals trade off against each other with only
17 Acts, and this is the honest way to trade them off rather than picking
whichever split happens to hit round numbers.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

MERGED = pathlib.Path("data/processed/summaries_v1/merged.jsonl")
OUT_DIR = pathlib.Path("data/processed")
TARGET_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = read_jsonl(MERGED)
    by_act: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_act[row["act_id"]].append(row)

    total = len(rows)
    targets = {split: total * ratio for split, ratio in TARGET_RATIOS.items()}
    assigned: dict[str, int] = {split: 0 for split in TARGET_RATIOS}
    act_split: dict[str, str] = {}

    # Largest Act first: placing big Acts while every split still has slack
    # avoids the common failure mode where the biggest Act is forced into
    # whichever split is left over.
    for act_id, act_rows in sorted(by_act.items(), key=lambda kv: -len(kv[1])):
        split = max(TARGET_RATIOS, key=lambda s: targets[s] - assigned[s])
        act_split[act_id] = split
        assigned[split] += len(act_rows)

    split_rows: dict[str, list[dict]] = {split: [] for split in TARGET_RATIOS}
    for row in rows:
        split = act_split[row["act_id"]]
        row = {**row, "split": split}
        split_rows[split].append(row)

    for split, out_rows in split_rows.items():
        write_jsonl(OUT_DIR / f"summaries_{split}_v1.jsonl", out_rows)

    manifest = {
        "total_rows": total,
        "total_acts": len(by_act),
        "target_ratios": TARGET_RATIOS,
        "actual_row_counts": {s: len(r) for s, r in split_rows.items()},
        "actual_row_ratios": {s: round(len(r) / total, 4) for s, r in split_rows.items()},
        "acts_per_split": {
            split: sorted(a for a, s in act_split.items() if s == split)
            for split in TARGET_RATIOS
        },
    }
    (OUT_DIR / "summaries_split_manifest_v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
