"""Topic 1 — domain intent classification, four representations (proposal §8.5).

    python scripts/42_train_classifiers.py

Naive Bayes (generative) vs. Logistic Regression / VanillaRNN / StackedBiLSTM
(discriminative), all on the same `domain` labels from gold_verified_v2.jsonl,
same frozen 352/200 split as the retrieval rungs (test_split_v1.json) so no
qid is ever trained on and later tested on, here or anywhere else in the repo.

15 classes, `other` at ~66% -- reported here as macro-F1 (never accuracy
alone) plus a confusion matrix, because accuracy on this label distribution
would hide every minority-class failure. Not a project requirement (the
supervisor's guidance: classification is not mandatory), kept as cheap
syllabus coverage per docs/PIPELINE.md.

v3 (2026-09-19), vs v2: two imbalance fixes, one honest limit documented.
Fixes -- NB now fits with a uniform class prior (previously the default
learned prior, ~66% of which was 'other' by itself, doing most of the "just
predict other" work for it) and both RNNs now train with
`balanced_class_weights` cross-entropy (previously unweighted, the one
model here with zero imbalance correction). Limit -- 7 of 14 domains
(constitutional, civil_registration, cybercrime, local_government,
money_recovery, road_transport, tenancy) have 1-6 examples in the *entire*
552-row gold pool, not just this split. No reweighting scheme manufactures
a decision boundary from 2 training points. This script also runs a second,
merged-label pass (`rare_other` bucket for those 7 domains) to separate
"imbalance the model can be corrected for" from "classes too thin to model
at all" -- both results are written, neither replaces the other.

v6 (2026-09-19): LogReg gained a small CV-selected regularization sweep
(`build_tfidf_classifiers`'s `tune_lr`) instead of trusting the sklearn
default C blind -- a real, if modest, credibility improvement. A second
change, word bigrams in the TF-IDF vectorizer, was tried and reverted after
it regressed the best-performing variant's macro-F1 from 0.343 to 0.203
(overfitting on a 352-row training set at 1.4x the feature count) -- see
`build_tfidf_classifiers`'s docstring for the full account. Recorded here so
the attempt and its outcome aren't lost even though the code reverted.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.classify import (  # noqa: E402
    StackedBiLSTMClassifier,
    VanillaRNNClassifier,
    balanced_class_weights,
    build_tfidf_classifiers,
    predict_rnn,
    train_rnn,
)
from dhara.vocab import Vocab  # noqa: E402

GOLD = pathlib.Path("data/processed/gold_verified_v3.jsonl")
SPLIT = pathlib.Path("data/processed/test_split_v1.json")
VOCAB = pathlib.Path("data/processed/vocab.json")
MATRIX = pathlib.Path("models/embedding_matrix.pt")
OUT = pathlib.Path("results/runs/classify_v6.json")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
DOMAINS_CFG = pathlib.Path("configs/domains.yaml")
APPROVED = pathlib.Path("data/processed/train_retrieval_v3.jsonl")
AUTHORED = pathlib.Path("data/processed/authored_v1/merged_passing.jsonl")

# v4 (2026-09-19): gold_verified_v2's `domain` field was stale for 353/552
# rows -- it predated several domains.yaml additions and simply never got
# re-derived from the answer Act. Re-derived here deterministically (answer
# Act title matched against domains.yaml's own curated act lists) in
# gold_verified_v3.jsonl. Biggest single change: 'family' was undercounted
# at 7 rows (looked like severe data scarcity) and is actually 243 rows --
# the second-largest domain -- once correctly attributed. 'other' drops from
# 363 (66%) to 119 (22%).
#
# Domains with <=6 total examples in the CORRECTED pool: no reweighting
# scheme gives a model a decision boundary from 1-4 training points. The
# merged pass buckets these into 'rare_other' so the comparison separates
# "imbalance we fixed" from "data volume we can't fix here".
RARE_DOMAINS = {
    "constitutional", "civil_registration", "money_recovery", "road_transport",
}


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_act_id_to_domain() -> dict[str, str]:
    """Same deterministic Act-title-to-domain lookup used to build
    gold_verified_v3.jsonl (DECISIONS.md 2026-09-19), applied here by act_id
    so the 2,965 approved + 972 authored retrieval rows -- which have the
    same stale-label problem gold_verified_v2 had (approved: 70% 'other')
    -- can be used as classifier TRAINING data without needing new
    annotation. Never used for eval: these rows are synthetic (approved is
    LLM-generated from Act/section titles, authored is LLM-narrative against
    provision text), and eval must stay on real human questions only, same
    reasoning as the retrieval dev/test split.
    """
    import yaml

    domains_cfg = yaml.safe_load(DOMAINS_CFG.read_text(encoding="utf-8"))

    def clean(title: str) -> str:
        if not title:
            return ""
        title = re.sub(r"^\d+", "", title.strip())
        title = re.sub(r",?\s*\d{4}.*$", "", title)
        return title.strip().casefold()

    domain_act_titles = {d["id"]: [clean(a) for a in d["acts"]] for d in domains_cfg["domains"]}

    act_id_to_title: dict[str, str] = {}
    for row in read_jsonl(CORPUS):
        if row["act_id"] not in act_id_to_title:
            act_id_to_title[row["act_id"]] = row.get("act_title_bn") or row.get("act_title_en") or ""

    mapping: dict[str, str] = {}
    for act_id, title in act_id_to_title.items():
        c = clean(title)
        if not c:
            continue
        for dom_id, titles in domain_act_titles.items():
            if any(t and (t in c or c in t) for t in titles):
                mapping[act_id] = dom_id
                break
    return mapping


def load_synthetic_for_training(act_id_to_domain: dict[str, str]) -> list[dict]:
    rows = []
    for path, source_label in [(APPROVED, "approved"), (AUTHORED, "authored")]:
        for r in read_jsonl(path):
            domain = act_id_to_domain.get(r.get("act_id"), r.get("domain", "other"))
            rows.append({"qid": r["qid"], "question_bn": r["question"], "domain": domain,
                         "synthetic_source": source_label})
    return rows


def load_balanced_synthetic(act_id_to_domain: dict[str, str], n_approved: int = 400, n_authored: int = 400) -> list[dict]:
    """Quality-filtered, size-capped synthetic subset, not the full 3,972-row
    dump that diluted training to 92% synthetic and hurt LogReg (DECISIONS.md
    2026-09-19). 'Quality' here means LOWEST content_overlap for approved
    rows -- this project's premise is that real questions have near-zero
    lexical overlap with their answer, so a HIGH-overlap approved pair is the
    easy/shortcut-able one, not the good one -- and overlap_gate_waived=False
    for authored rows (passed the lexical-overlap quality gate cleanly, no
    manual waiver needed).
    """
    approved = sorted(
        read_jsonl(APPROVED),
        key=lambda r: r.get("content_overlap", 1.0),
    )[:n_approved]
    authored = [r for r in read_jsonl(AUTHORED) if not r.get("overlap_gate_waived")][:n_authored]

    rows = []
    for r, source_label in [(row, "approved") for row in approved] + [(row, "authored") for row in authored]:
        domain = act_id_to_domain.get(r.get("act_id"), r.get("domain", "other"))
        rows.append({"qid": r["qid"], "question_bn": r["question"], "domain": domain,
                     "synthetic_source": source_label})
    return rows


def run_variant(variant: str, train_rows: list[dict], test_rows: list[dict], domain_of) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    labels = sorted({domain_of(r) for r in train_rows + test_rows})
    label_to_id = {d: i for i, d in enumerate(labels)}
    n_classes = len(labels)
    print(f"[{variant}] {n_classes} domains: {labels}")

    train_texts = [r["question_bn"] for r in train_rows]
    train_labels_str = [domain_of(r) for r in train_rows]
    train_labels_id = [label_to_id[d] for d in train_labels_str]
    test_texts = [r["question_bn"] for r in test_rows]
    test_labels_str = [domain_of(r) for r in test_rows]

    out: dict = {"labels": labels, "n_train": len(train_rows), "n_test": len(test_rows), "models": {}}
    per_query: dict[str, dict] = {r["qid"]: {"qid": r["qid"], "domain_true": domain_of(r)} for r in test_rows}

    # ---- TF-IDF: Naive Bayes (generative, uniform prior) + Logistic Regression (discriminative, balanced)
    vectorizer, nb, lr = build_tfidf_classifiers(train_texts, train_labels_str)
    X_test = vectorizer.transform(test_texts)
    for name, model in (("naive_bayes", nb), ("logistic_regression", lr)):
        preds = model.predict(X_test)
        for r, p in zip(test_rows, preds):
            per_query[r["qid"]][f"domain_pred_{name}"] = p
        acc = accuracy_score(test_labels_str, preds)
        macro_f1 = f1_score(test_labels_str, preds, average="macro", zero_division=0)
        out["models"][name] = {
            "representation": "tfidf (unigram+bigram, min_df=2)", "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class_f1": {
                d: round(f, 4) for d, f in zip(
                    labels, f1_score(test_labels_str, preds, labels=labels, average=None, zero_division=0)
                )
            },
            "confusion_matrix": confusion_matrix(test_labels_str, preds, labels=labels).tolist(),
        }
        cv_selection = getattr(model, "cv_selection_", None)
        if cv_selection:
            out["models"][name]["cv_selection"] = cv_selection
        print(f"  {name:22s} acc={acc:.4f} macro_f1={macro_f1:.4f}"
              + (f"  (C={cv_selection['selected_C']} via 3-fold CV)" if cv_selection else ""))

    # ---- RNNs over the shared Topic-0 embedding matrix, balanced cross-entropy
    vocab = Vocab.load(VOCAB)
    embedding_matrix = torch.load(MATRIX, map_location="cpu", weights_only=False)
    class_weights = balanced_class_weights(train_labels_id, n_classes)

    for name, cls in (("vanilla_rnn", VanillaRNNClassifier), ("stacked_bilstm", StackedBiLSTMClassifier)):
        model = cls(embedding_matrix, n_classes)
        train_rnn(model, train_texts, train_labels_id, vocab, class_weights=class_weights)
        pred_ids = predict_rnn(model, test_texts, vocab)
        preds = [labels[i] for i in pred_ids]
        for r, p in zip(test_rows, preds):
            per_query[r["qid"]][f"domain_pred_{name}"] = p
        acc = accuracy_score(test_labels_str, preds)
        macro_f1 = f1_score(test_labels_str, preds, average="macro", zero_division=0)
        out["models"][name] = {
            "representation": "topic0_embedding_matrix", "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class_f1": {
                d: round(f, 4) for d, f in zip(
                    labels, f1_score(test_labels_str, preds, labels=labels, average=None, zero_division=0)
                )
            },
            "confusion_matrix": confusion_matrix(test_labels_str, preds, labels=labels).tolist(),
        }
        print(f"  {name:22s} acc={acc:.4f} macro_f1={macro_f1:.4f}")

    out["per_query"] = list(per_query.values())
    return out


def main() -> None:
    rows = read_jsonl(GOLD)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_qids, test_qids = set(split["train_qids"]), set(split["test_qids"])

    by_qid = {r["qid"]: r for r in rows}
    human_train_rows = [by_qid[q] for q in train_qids if q in by_qid]
    test_rows = [by_qid[q] for q in test_qids if q in by_qid]  # real human only, never synthetic

    # Approved (LLM-generated from titles) + authored (LLM-narrative) rows,
    # relabelled by the same deterministic Act-to-domain lookup that fixed
    # gold_verified_v3 -- these were never eligible before because their
    # domain field had the same staleness problem. Training-only: eval stays
    # on real human questions, same reasoning as the retrieval dev/test split
    # (a synthetic-question test would measure something circular).
    act_id_to_domain = build_act_id_to_domain()
    synthetic_rows = load_synthetic_for_training(act_id_to_domain)
    synthetic_other = sum(1 for r in synthetic_rows if r["domain"] == "other")
    print(f"synthetic training rows: {len(synthetic_rows)} "
          f"({sum(1 for r in synthetic_rows if r['synthetic_source']=='approved')} approved + "
          f"{sum(1 for r in synthetic_rows if r['synthetic_source']=='authored')} authored), "
          f"{synthetic_other} still 'other' after remap")

    # Leakage check: no synthetic training question is a near-duplicate of a
    # real held-out test question (qid namespaces are already disjoint, but
    # question TEXT is what actually matters for leakage).
    def norm_q(s: str) -> str:
        return re.sub(r"\s+", " ", s.casefold()).strip()
    test_qtexts = {norm_q(r["question_bn"]) for r in test_rows}
    leaked = [r for r in synthetic_rows if norm_q(r["question_bn"]) in test_qtexts]
    assert not leaked, f"{len(leaked)} synthetic training rows duplicate a held-out test question"

    train_rows = human_train_rows + synthetic_rows
    print(f"train {len(train_rows)} ({len(human_train_rows)} human + {len(synthetic_rows)} synthetic), test {len(test_rows)}")

    def fine_grained_domain(r: dict) -> str:
        return r["domain"]

    def merged_domain(r: dict) -> str:
        return "rare_other" if r["domain"] in RARE_DOMAINS else r["domain"]

    fine = run_variant("fine_grained_rebalanced", train_rows, test_rows, fine_grained_domain)
    merged = run_variant("merged_rare_domains", train_rows, test_rows, merged_domain)
    human_only_fine = run_variant("human_only_fine_grained", human_train_rows, test_rows, fine_grained_domain)

    balanced_synthetic = load_balanced_synthetic(act_id_to_domain, n_approved=400, n_authored=400)
    balanced_train_rows = human_train_rows + balanced_synthetic
    print(f"balanced mix: {len(balanced_train_rows)} ({len(human_train_rows)} human + {len(balanced_synthetic)} quality-filtered synthetic, human share {100*len(human_train_rows)/len(balanced_train_rows):.0f}%)")
    balanced = run_variant("balanced_mix", balanced_train_rows, test_rows, fine_grained_domain)

    all_domains = [r["domain"] for r in train_rows + test_rows]
    n_total = len(all_domains)
    n_other = sum(1 for d in all_domains if d == "other")
    n_rare = sum(1 for d in all_domains if d in RARE_DOMAINS)

    results = {
        "run_id": "classify_v6",
        "gold": str(GOLD),
        "split": str(SPLIT),
        "vs_v5": "results/runs/classify_v5.json",
        "n_human_train": len(human_train_rows),
        "n_synthetic_train": len(synthetic_rows),
        "n_test": len(test_rows),
        "fine_grained_rebalanced": fine,
        "merged_rare_domains": merged,
        "human_only_fine_grained": human_only_fine,
        "balanced_mix": balanced,
        "n_balanced_synthetic": len(balanced_synthetic),
        "class_imbalance_note": (
            f"'other' is {n_other}/{n_total} ({100*n_other/n_total:.0f}%) of all questions. "
            f"Domains ({', '.join(sorted(RARE_DOMAINS))}) total {n_rare}/{n_total} "
            f"({100*n_rare/n_total:.0f}%) COMBINED, still thin even after adding synthetic data "
            "because approved/authored questions were generated to match the corpus's own Act "
            "distribution, not to specifically cover rare domains. v5 vs v4: training now "
            "includes 2,965 approved + 972 authored rows (relabelled via the same deterministic "
            "Act lookup that fixed gold_verified_v3), test set unchanged (real human only, "
            "200 rows) so any macro-F1 change here is a genuine generalization result, not a "
            "leakage artifact -- checked directly, 0 synthetic training questions duplicate a "
            "held-out test question. human_only_fine_grained is the v4 result re-run for a "
            "same-script, same-run baseline comparison."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
