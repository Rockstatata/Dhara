"""Verify the training artifacts are mutually consistent before a Colab run.

    python scripts/23_preflight.py
    python scripts/23_preflight.py --variant full

Exits non-zero on the first hard failure, so it can gate a run.

## Why

A GPU run costs ~30 minutes and every failure so far has been silent at the
point it mattered. Attempt 1 trained on contaminated negatives and looked fine
until the full eval. A later attempt was very nearly trained on stale uploads,
because the notebook's upload cell skips when a file of the same name is already
on the runtime's disk — same filename, different contents, no warning anywhere.

Every check here is one that has actually gone wrong, or is one line away from a
failure that has. None of them need a GPU.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

PROBES = pathlib.Path("data/processed/probe_questions.jsonl")
AUDIT = pathlib.Path("results/runs/overlap_audit.json")
CORPUS_GZ = pathlib.Path("data/processed/corpus_v1.jsonl.gz")

failures: list[str] = []
warnings: list[str] = []


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check(ok: bool, label: str, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {label}")
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL  {label}  --  {detail}")


def warn(ok: bool, label: str, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {label}")
    else:
        warnings.append(f"{label}: {detail}")
        print(f"  WARN  {label}  --  {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["strict", "full", "anchor"], default="strict")
    args = ap.parse_args()

    train_path = pathlib.Path(f"data/processed/train_{args.variant}_v1.jsonl")
    neg_path = pathlib.Path(f"data/processed/train_{args.variant}_v1_negatives.jsonl")
    dev_path = pathlib.Path(f"data/processed/dev_{args.variant}_v1.jsonl")

    print(f"preflight for variant '{args.variant}'\n")

    print("files exist")
    for p in (PROBES, train_path, neg_path, dev_path, CORPUS_GZ):
        check(p.exists(), str(p), "missing")
    if failures:
        print("\nmissing inputs; nothing else can be checked")
        raise SystemExit(1)

    probes = read_jsonl(PROBES)
    train = read_jsonl(train_path)
    negs = read_jsonl(neg_path)
    dev = read_jsonl(dev_path)

    print("\nnegatives file is current with the split")
    check(len(negs) == len(train),
          "negatives row count == train row count",
          f"{len(negs)} vs {len(train)} -- re-run scripts/17_mine_negatives.py")
    if len(negs) == len(train):
        same = all(a["question_bn"] == b["question_bn"] and
                   a["gold_chunk_ids"] == b["gold_chunk_ids"]
                   for a, b in zip(train, negs))
        check(same, "negatives rows align with train rows", "row order or content differs")

    print("\ngold isolation")
    probe_q = {p["question_bn"] for p in probes}
    for name, rows in (("train", train), ("dev", dev)):
        overlap = {r["question_bn"] for r in rows} & probe_q
        check(not overlap, f"no probe question appears in {name}",
              f"{len(overlap)} leaked, e.g. {sorted(overlap)[:2]}")

    gold_provs = {p for r in probes for p in r["gold_provision_ids"]}
    hits = {r["gold_provision_ids"][0] for r in train} & gold_provs
    if args.variant in ("strict", "anchor"):
        check(not hits, f"{args.variant}: no training positive is a gold provision",
              f"{len(hits)} found -- re-run scripts/16_split_training.py")
    else:
        warn(True, f"full: {len(hits)} training positives are gold provisions (expected for this variant)")

    print("\ntrain/dev separation")
    train_topics = {r["topic_id"] for r in train}
    dev_topics = {r["topic_id"] for r in dev}
    check(not (train_topics & dev_topics), "train and dev topics are disjoint",
          f"shared: {sorted(train_topics & dev_topics)[:3]}")
    shared_q = {r["question_bn"] for r in train} & {r["question_bn"] for r in dev}
    check(not shared_q, "no question in both train and dev", f"{len(shared_q)} shared")

    print("\nnegatives quality")
    all_positives = {r["gold_provision_ids"][0] for r in negs}
    contaminated = 0
    total_slots = 0
    empty = 0
    for r in negs:
        n = r.get("negative_chunk_ids", [])
        total_slots += len(n)
        if not n:
            empty += 1
    prov_of_chunk: dict[str, str] = {}
    for r in negs:
        prov_of_chunk[r["gold_chunk_ids"][0]] = r["gold_provision_ids"][0]
    for r in negs:
        for n in r.get("negative_chunk_ids", []):
            p = prov_of_chunk.get(n)
            if p and p in all_positives and p != r["gold_provision_ids"][0]:
                contaminated += 1
    check(empty == 0, "every pair has at least one negative", f"{empty} pairs have none")
    rate = contaminated / max(total_slots, 1)
    check(rate < 0.01,
          "cross-topic negative contamination under 1%",
          f"{contaminated}/{total_slots} = {rate:.1%} -- the attempt-1 bug; "
          f"scripts/17_mine_negatives.py must exclude ALL training positives")

    if args.variant == "anchor":
        print("\nlabel purity")
        sources = collections.Counter(r.get("label_source") for r in train)
        check(set(sources) == {"anchor"},
              "every training positive is hand-anchored",
              f"found {dict(sources)} -- keyword-matched labels are ~80% wrong at "
              f"section level (see scripts/16_split_training.py docstring)")
        print(f"        {len({r['gold_provision_ids'][0] for r in train})} distinct positive "
              f"provisions over {len({r['topic_id'] for r in train})} topics")

    print("\ntier selection matches the §5.3 audit")
    if AUDIT.exists():
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        recommended = set(audit["recommended_tiers"])
        used = {r["quality_tier"] for r in train}
        check(used <= recommended,
              "train uses only audit-approved tiers",
              f"used {sorted(used)}, approved {sorted(recommended)}")
    else:
        warn(False, "overlap audit present", f"{AUDIT} missing -- run scripts/15_overlap_audit.py")

    print("\nquestion shape vs the real questions")
    def words(rows, key="question_bn"):
        return [len(r[key].split()) for r in {r["question_bn"]: r for r in rows}.values()]
    real, synth = words(probes), words(train)
    rm, sm = statistics.median(real), statistics.median(synth)
    rsd = statistics.pstdev(real)
    ssd = statistics.pstdev(synth)
    print(f"        real  : median {rm:.0f} words, sd {rsd:.1f}")
    print(f"        train : median {sm:.0f} words, sd {ssd:.1f}, "
          f"{len({r['question_bn'] for r in train})} distinct questions")
    warn(sm >= rm * 0.4, "train median length within 2.5x of real",
         f"{sm:.0f} vs {rm:.0f} -- short one-liners against long narratives is a "
         f"train/test distribution gap (see DECISIONS.md 2026-09-04)")
    warn(ssd >= rsd * 0.35, "train length spread within 3x of real",
         f"sd {ssd:.1f} vs {rsd:.1f}")

    print("\nupload fingerprints -- check these against the Colab side if a run looks wrong")
    for p in (train_path, neg_path, dev_path, PROBES):
        digest = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
        print(f"        {digest}  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

    print()
    if failures:
        print(f"{len(failures)} HARD FAILURE(S) -- do not start a GPU run:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    if warnings:
        print(f"{len(warnings)} warning(s), none blocking:")
        for w in warnings:
            print(f"  - {w}")
    print("\npreflight clean -- safe to upload and run.")


if __name__ == "__main__":
    main()
