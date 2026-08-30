"""Retrieval metrics and the paired bootstrap.

Two rules from the guide are enforced by the shape of this module rather than by
discipline:

**Provision-level, not chunk-level.** A long section is split into several
chunks, all of which cite the same provision. Retrieving chunk 2 when the gold
label names chunk 1 is a correct answer to the user's question, and scoring it
as a miss would understate every rung equally but not identically — longer
provisions would be penalised more. `first_hit` therefore matches on
`provision_id`.

**Nothing is aggregated without keeping the per-query record.** Every function
here takes or returns the per-query ranks, because error analysis and
significance testing both need them and regenerating them later means re-running
the whole experiment. At n=552 the 95% CI on Recall@5 is roughly +/-4 points, so
adjacent rungs will sometimes be indistinguishable; a difference reported without
a CI on the *difference* is not a result.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence

CUTOFFS = (1, 5, 10, 20, 50, 100)


def first_hit(ranked_provision_ids: Iterable[str], gold: set[str]) -> Optional[int]:
    """1-based rank of the first ranked item whose provision is gold, else None."""
    for rank, provision_id in enumerate(ranked_provision_ids, 1):
        if provision_id in gold:
            return rank
    return None


def recall_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """Fraction of queries whose first gold hit landed at rank <= k.

    With one gold provision per query this is recall; with several it is
    hit-rate@k, which is the quantity the UI actually cares about (does the top-k
    the user sees contain a right answer at all).
    """
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: Sequence[Optional[int]], cutoff: int = 10) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None and r <= cutoff) / len(ranks)


def summarize(ranks: Sequence[Optional[int]], cutoffs: Sequence[int] = CUTOFFS) -> dict:
    return {
        "n": len(ranks),
        **{f"R@{k}": round(recall_at_k(ranks, k), 4) for k in cutoffs},
        "MRR@10": round(mrr(ranks, 10), 4),
    }


def paired_bootstrap(
    ranks_a: Sequence[Optional[int]],
    ranks_b: Sequence[Optional[int]],
    k: int = 5,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """95% CI on (metric_a - metric_b) at cutoff k, resampling queries in pairs.

    Paired because both rungs answer the same queries: the query-difficulty
    variance is shared and cancels, which is the only reason a 552-query
    comparison has any power at all. `significant` is CI-excludes-zero, and a
    pair honestly reported as within noise reads as more credible than a
    narrated bar chart.
    """
    if len(ranks_a) != len(ranks_b):
        raise ValueError("paired bootstrap needs the same queries in the same order")
    n = len(ranks_a)
    hit_a = [1.0 if r is not None and r <= k else 0.0 for r in ranks_a]
    hit_b = [1.0 if r is not None and r <= k else 0.0 for r in ranks_b]
    observed = (sum(hit_a) - sum(hit_b)) / n

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            total += hit_a[i] - hit_b[i]
        deltas.append(total / n)
    deltas.sort()
    lo = deltas[int((alpha / 2) * n_resamples)]
    hi = deltas[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return {
        "k": k,
        "delta": round(observed, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "significant": bool(lo > 0 or hi < 0),
        "n_resamples": n_resamples,
    }
