"""Topic 3 — perplexity as a second, independent measure of the register gap.

    python scripts/40_measure_perplexity.py

The LSTM language model (models/language_model.pt) is trained only on
formal legal Bangla (corpus text_bn) -- it never sees a citizen question
during training. If colloquial questions are measurably more surprising to
it than formal ones, that is the register gap, measured by an instrument
that never touches a retriever.

Run on all 552 gold questions (not just the 480 answerable/probe rows): this
is not retrieval training or tuning, so the leakage rule that keeps the 200
frozen test qids out of every trained model does not apply to a read-only
perplexity measurement, and the extra n materially helps the underpowered
formal slice (41 formal questions is already thin; restricting to train-split
only would cut it further for no reason).
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.models.language_model import load, perplexity  # noqa: E402
from dhara.vocab import Vocab  # noqa: E402

GOLD = pathlib.Path("data/processed/gold_verified_v2.jsonl")
LM = pathlib.Path("models/language_model.pt")
VOCAB = pathlib.Path("data/processed/vocab.json")
MATRIX = pathlib.Path("models/embedding_matrix.pt")
OUT = pathlib.Path("results/runs/lm_perplexity_v2.json")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    rows = read_jsonl(GOLD)
    vocab = Vocab.load(VOCAB)
    embedding_matrix = torch.load(MATRIX, map_location="cpu", weights_only=False)
    model = load(LM, embedding_matrix)
    print(f"{len(rows)} gold questions, LM loaded from {LM}")

    per_query = []
    by_register: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        ppl = perplexity(model, vocab, r["question_bn"])
        if ppl is None or not (ppl == ppl) or ppl == float("inf"):  # NaN/inf guard
            continue
        per_query.append({"qid": r["qid"], "register": r["register"], "domain": r["domain"], "perplexity": round(ppl, 2)})
        by_register[r["register"]].append(ppl)

    def summary(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        return {
            "n": n,
            "mean": round(sum(vals_sorted) / n, 2),
            "median": round(vals_sorted[n // 2], 2),
            "min": round(vals_sorted[0], 2),
            "max": round(vals_sorted[-1], 2),
        }

    stats = {reg: summary(vals) for reg, vals in by_register.items()}
    print("\nperplexity by register (lower = less surprising to the legal-Bangla LM):")
    for reg, s in stats.items():
        print(f"  {reg:12s} {s}")

    if "colloquial" in stats and "formal" in stats and stats["colloquial"]["n"] and stats["formal"]["n"]:
        gap = stats["colloquial"]["mean"] - stats["formal"]["mean"]
        print(f"\nmean gap (colloquial - formal): {gap:+.2f}  "
              f"(positive = colloquial questions are more surprising to the legal-Bangla model, "
              f"as the register-gap hypothesis predicts)")

    results = {
        "run_id": "lm_perplexity_v2",
        "gold": str(GOLD),
        "checkpoint": str(LM),
        "n_scored": len(per_query),
        "n_total": len(rows),
        "by_register": stats,
        "per_query": per_query,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
