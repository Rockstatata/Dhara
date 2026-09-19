"""Rebuild human-aware retrieval splits (v4): shrink dev/test, grow train.

    python scripts/68_build_human_aware_retrieval_splits_v4.py

`51_build_human_aware_retrieval_splits_v3.py` reserved 50% of the 480
answerable gold-verified rows for train and 25% each for dev/test (251
human train / 117 dev / 112 test), and deliberately allowed a provision to
appear in both human train and human eval ("this evaluates generalization
to new formulations of a known legal provision" -- its own docstring).

`66_merge_authored_into_pool.py` then enforced a STRICTER, provision-level
train/eval exclusion when folding in the 972 authored questions -- a policy
v3's split never imposed. That mismatch retroactively dropped 199 of the
251 human train rows (kept only 52), which is the actual root cause of the
2026-09 flat fine-tune result (see DECISIONS.md 2026-09-19).

This script fixes the conflict at the source: it enforces the SAME
provision-level exclusion the merge step already enforces, at split time,
using connected components over shared provisions (a row citing two
provisions binds both provisions to the same split, transitively, so no
provision can ever end up a positive in two different splits). It also
reserves only 10% each for dev/test instead of 25% each. Shrinking eval
size is a deliberate tradeoff to grow the only source of real citizen
training questions this project has; the CI on dev/test at n~48 will be
wide as a result, and every comparison against these files must report
that CI (scripts/18_compare_runs.py), not read the point estimate alone.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
from typing import Any

GOLD = pathlib.Path("data/processed/gold_verified_v2.jsonl")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
TRAIN_OUT = pathlib.Path("data/processed/human_train_v4.jsonl")
DEV_OUT = pathlib.Path("data/processed/dev_retrieval_v4.jsonl")
TEST_OUT = pathlib.Path("data/processed/test_retrieval_v4.jsonl")
AUDIT_OUT = pathlib.Path("results/runs/retrieval_human_aware_splits_v4.json")

TRAIN_FRACTION = 0.8


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_question(question: str) -> str:
    import re

    return re.sub(r"\s+", " ", question.casefold()).strip()


def eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": row["qid"],
        "question": row["question_bn"],
        "positive_chunk_ids": row["relevant_chunk_ids"],
        "hard_negative_chunk_ids": [],
        "domain": row["domain"],
        "register": row["register"],
        "source": "human_adjudicated_v2",
        "annotation_mode": row["annotation_mode"],
        "answer_source": row["answer_source"],
    }


def train_row(row: dict[str, Any], chunk_to_row: dict[str, dict[str, Any]]) -> dict[str, Any]:
    positives = row["relevant_chunk_ids"]
    first = chunk_to_row[positives[0]]
    return {
        "qid": row["qid"],
        "question": row["question_bn"],
        "positive_chunk_ids": positives,
        "hard_negative_chunk_ids": [],
        "provision_id": first["provision_id"],
        "act_id": first["act_id"],
        "domain": row["domain"],
        "lang_tag": "bengali" if any("ঀ" <= char <= "৿" for char in row["question_bn"]) else "english",
        "source": "human_adjudicated_v2",
        "annotation_mode": row["annotation_mode"],
        "answer_source": row["answer_source"],
        "label_source": "human_adjudicated_provision",
        "review_status": "approved",
        "eligible_for_headline_training": True,
    }


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main() -> None:
    corpus = read_jsonl(CORPUS)
    chunk_to_row = {row["chunk_id"]: row for row in corpus}
    gold = [row for row in read_jsonl(GOLD) if row.get("answerable") and row.get("relevant_chunk_ids")]
    assert len(gold) == 480, f"expected 480 answerable human rows, found {len(gold)}"
    assert all(cid in chunk_to_row for row in gold for cid in row["relevant_chunk_ids"])

    # Connected components over shared provisions: a row citing provisions
    # {A, B} binds A and B to the same split forever. This is the invariant
    # the downstream authored-merge step needs and the v3 split never gave it.
    uf = UnionFind()
    for row in gold:
        provisions = [chunk_to_row[cid]["provision_id"] for cid in row["relevant_chunk_ids"]]
        for p in provisions[1:]:
            uf.union(provisions[0], p)

    components: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in gold:
        provisions = [chunk_to_row[cid]["provision_id"] for cid in row["relevant_chunk_ids"]]
        root = uf.find(provisions[0])
        components[root].append(row)

    # Stratify by the component's own domain (majority vote across its rows,
    # ties broken by a stable hash) so a component spanning shared provisions
    # across two domains doesn't get lost in whichever domain happens first.
    # A single global 80/10/10 split-by-size (no domain awareness) put 65% of
    # dev/test into the "other" domain and left family/constitutional/
    # cybercrime almost unrepresented -- exactly the domains CLAUDE.md flags
    # as high-risk and requiring evaluation. Stratifying per domain first
    # fixes that: each domain gets its own 80/10/10 allocation.
    def component_domain(rows: list[dict[str, Any]]) -> str:
        counts = collections.Counter(r["domain"] for r in rows)
        best = counts.most_common()
        top_count = best[0][1]
        tied = sorted(name for name, count in best if count == top_count)
        return tied[0]

    bucketed: dict[str, list[list[dict[str, Any]]]] = collections.defaultdict(list)
    for component_rows in components.values():
        bucketed[component_domain(component_rows)].append(component_rows)

    # A component citing several shared provisions can be large purely because
    # several real citizens asked correlated variants of the same popular
    # question -- one component alone accounts for 386 of the 480 rows in
    # this pool. Letting *any* component land in dev/test regardless of size
    # means a single 5-9 row cluster can dominate a 35-36 row eval split,
    # making that split behave like a much smaller, correlated sample (this
    # is exactly what produced the dev-vs-test instability flagged
    # 2026-09-19). Large clusters are fine in train -- more repetition of a
    # popular provision only helps there -- so anything above
    # MAX_EVAL_COMPONENT_SIZE is routed to train unconditionally, and only
    # small, largely-independent components compete for dev/test slots.
    #
    # A first version enforced this per-domain (each domain gets its own
    # 80/10/10 quota, like the domain stratification above). That starved
    # dev/test whenever a domain's own small-component pool couldn't fill its
    # local quota -- e.g. the domain holding the 386-row mega-component had
    # almost no small components left over, so its ~40-row dev/test quota
    # went unmet and nothing compensated from other domains. Domain balance
    # is still tracked (round-robin ordering below) but the fill target is
    # global, so one domain's thin small-pool doesn't cap the whole split.
    counts = {"train": 0, "dev": 0, "test": 0}
    assigned: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    MAX_EVAL_COMPONENT_SIZE = 4
    small_by_domain: dict[str, list[list[dict[str, Any]]]] = {}
    for domain in sorted(bucketed):
        for component_rows in bucketed[domain]:
            if len(component_rows) > MAX_EVAL_COMPONENT_SIZE:
                assigned["train"].extend(component_rows)
                counts["train"] += len(component_rows)
            else:
                small_by_domain.setdefault(domain, []).append(component_rows)
        if domain in small_by_domain:
            small_by_domain[domain].sort(
                key=lambda rows: (
                    -len(rows),
                    hashlib.sha1(f"human-aware-v4:{rows[0]['qid']}".encode("utf-8")).hexdigest(),
                ),
            )

    # Round-robin across domains (largest-first within each) so dev/test draw
    # from many domains instead of exhausting one domain's pool before
    # touching the next.
    total = len(gold)
    targets = {
        "train": total * TRAIN_FRACTION,
        "dev": total * (1 - TRAIN_FRACTION) / 2,
        "test": total * (1 - TRAIN_FRACTION) / 2,
    }
    domains_cycle = sorted(small_by_domain)
    cursors = {domain: 0 for domain in domains_cycle}
    remaining = sum(len(pool) for pool in small_by_domain.values())
    while remaining:
        for domain in domains_cycle:
            pool = small_by_domain[domain]
            cursor = cursors[domain]
            if cursor >= len(pool):
                continue
            component_rows = pool[cursor]
            cursors[domain] += 1
            remaining -= 1
            deficit = {name: targets[name] - counts[name] for name in counts}
            split = max(deficit, key=lambda name: deficit[name])
            assigned[split].extend(component_rows)
            counts[split] += len(component_rows)

    human_train_rows = [train_row(row, chunk_to_row) for row in assigned["train"]]
    dev_rows = [eval_row(row) for row in assigned["dev"]]
    test_rows = [eval_row(row) for row in assigned["test"]]

    split_sets = {
        "train": {normalize_question(row["question"]) for row in human_train_rows},
        "dev": {normalize_question(row["question"]) for row in dev_rows},
        "test": {normalize_question(row["question"]) for row in test_rows},
    }
    assert not (split_sets["train"] & split_sets["dev"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["dev"] & split_sets["test"])

    train_provisions = {row["provision_id"] for row in human_train_rows}
    eval_provisions = {
        chunk_to_row[cid]["provision_id"]
        for row in [*dev_rows, *test_rows] for cid in row["positive_chunk_ids"]
    }
    assert not (train_provisions & eval_provisions), "provision leaked across human train/eval"

    write_jsonl(TRAIN_OUT, human_train_rows)
    write_jsonl(DEV_OUT, dev_rows)
    write_jsonl(TEST_OUT, test_rows)
    audit = {
        "version": "v4",
        "split_policy": "provision-connected-component, best-fit-decreasing, 80/10/10 by row count",
        "human_adjudicated": {"train": len(human_train_rows), "dev": len(dev_rows), "test": len(test_rows)},
        "components": len(components),
        "largest_component_rows": max(len(v) for v in components.values()),
        "note": "dev/test shrunk from v3's 117/112 to grow real human train signal; CI at this n is wide, report it",
        "human_train_out": str(TRAIN_OUT),
        "frozen_test": str(TEST_OUT),
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
