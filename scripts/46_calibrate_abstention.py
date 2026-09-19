"""Calibrate the abstention threshold on the deliberately-included unanswerable gold.

    python scripts/46_calibrate_abstention.py --run results/runs/bge_m3_acttitle_v2.json \
        --index models/index_bge_m3_acttitle_v1

Every output must be able to say "no confident match" (CLAUDE.md's
non-negotiable rule). `service.py` shipped with a guessed fallback threshold
(`{"default": 0.52, "high": 0.58}`) pending this calibration. This script
sweeps a threshold against two populations:

  - the 480 **answerable** probe questions' top-1 score (from an already-scored
    run's `per_query`, so no re-encoding needed for these)
  - the 72 **answerable == False** gold rows (`abstain_calibration_v2.jsonl`),
    live-encoded against the same index -- there is no cached score for them
    because they were never fed to a retriever before now

and picks the threshold that minimises `2 * false_confidence + false_abstain`
(the ethics-driven asymmetry: a confidently wrong answer is worse than an
unnecessary "I don't know" -- CLAUDE.md, service.py's own docstring), split by
risk tier via `configs/domains.yaml`'s risk_tier per domain (high vs.
everything else), so the high-tier threshold can sit higher independently.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

sys.stdout.reconfigure(encoding="utf-8")

ABSTAIN = pathlib.Path("data/processed/abstain_calibration_v2.jsonl")
DOMAINS_CFG = pathlib.Path("configs/domains.yaml")
OUT = pathlib.Path("configs/abstention.json")
THRESHOLD_GRID = np.arange(0.0, 0.80, 0.01)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def risk_tier_of_domain() -> dict[str, str]:
    cfg = yaml.safe_load(DOMAINS_CFG.read_text(encoding="utf-8"))
    domains = cfg["domains"]  # list of {id, name_bn, name_en, risk_tier, ...}
    return {d["id"]: d.get("risk_tier", "low") for d in domains}


def sweep(answerable_scores: list[float], unanswerable_scores: list[float]) -> dict:
    best = None
    curve = []
    for tau in THRESHOLD_GRID:
        false_abstain = sum(1 for s in answerable_scores if s < tau)
        false_confidence = sum(1 for s in unanswerable_scores if s >= tau)
        cost = 2 * false_confidence + false_abstain
        curve.append({
            "tau": round(float(tau), 3), "false_abstain": false_abstain,
            "false_confidence": false_confidence, "cost": cost,
        })
        if best is None or cost < best["cost"]:
            best = curve[-1]
    return {"selected": best, "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True,
                    help="a results/runs/*.json for the index the demo will actually serve, "
                         "already scored on probe_questions_v2.jsonl")
    ap.add_argument("--index", type=pathlib.Path, required=True)
    args = ap.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    risk_tier = risk_tier_of_domain()

    abstain_rows = read_jsonl(ABSTAIN)
    print(f"{len(abstain_rows)} unanswerable rows to live-encode against {args.index}")

    from sentence_transformers import SentenceTransformer

    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    corpus_emb = np.load(args.index / "embeddings.npy").astype(np.float32)
    model = SentenceTransformer(manifest["checkpoint"], device="cpu")
    q_emb = model.encode([r["question_bn"] for r in abstain_rows], batch_size=32,
                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    top_scores = (corpus_emb @ q_emb.T).max(axis=0)
    for r, s in zip(abstain_rows, top_scores):
        r["top_score"] = float(s)

    results: dict = {"run": str(args.run), "index": str(args.index), "tiers": {}}
    for tier_name, tier_set in (("high", {"high"}), ("default", {"low", "medium"})):
        ans_scores = [
            r["top_score"] for r in run["per_query"]
            if r.get("top_score") is not None and risk_tier.get(r.get("block"), "low") in tier_set
        ]
        unans_scores = [r["top_score"] for r in abstain_rows if risk_tier.get(r.get("domain"), "low") in tier_set]
        print(f"\n{tier_name}: {len(ans_scores)} answerable, {len(unans_scores)} unanswerable")
        if unans_scores:
            print(f"  unanswerable top_score range: {min(unans_scores):.3f} - {max(unans_scores):.3f}")
        result = sweep(ans_scores, unans_scores)
        result["n_unanswerable"] = len(unans_scores)
        result["small_sample_warning"] = len(unans_scores) < 15

        if result["small_sample_warning"] and unans_scores:
            # The cost-minimising sweep is not trustworthy on this few examples
            # (n=5 for 'high' here) -- CLAUDE.md's rule is that abstention must be
            # CONSERVATIVE for high-risk tiers, never data-fit down to whatever a
            # handful of examples happens to allow. Override: sit just above the
            # highest observed unanswerable score, so at least every unanswerable
            # example seen so far would correctly abstain, and accept the (higher)
            # false-abstain cost on answerable questions as the safe tradeoff --
            # exactly the asymmetry service.py's own docstring states.
            safe_tau = round(min(0.79, max(unans_scores) + 0.02), 2)
            fa = sum(1 for s in ans_scores if s < safe_tau)
            fc = sum(1 for s in unans_scores if s >= safe_tau)
            result["selected"] = {"tau": safe_tau, "false_abstain": fa, "false_confidence": fc,
                                  "cost": 2 * fc + fa, "overridden_for_small_sample": True}
        results["tiers"][tier_name] = result
        print(f"  selected tau={result['selected']['tau']}  "
              f"false_abstain={result['selected']['false_abstain']}/{len(ans_scores)}  "
              f"false_confidence={result['selected']['false_confidence']}/{len(unans_scores)}"
              + ("  ** small sample (n<15): overridden to sit above the observed "
                 "unanswerable max rather than trust a cost-minimising fit **"
                 if result["small_sample_warning"] else ""))

    thresholds = {t: results["tiers"][t]["selected"]["tau"] for t in ("default", "high")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"thresholds": thresholds, "calibration": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {OUT}: {thresholds}")


if __name__ == "__main__":
    main()
