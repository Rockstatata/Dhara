"""The demo's classifier and tagger must be the models the results tables describe."""

import json
import pathlib

import pytest

from dhara import insights

needs_data = pytest.mark.skipif(
    not (insights.GOLD.exists() and insights.SPLIT.exists() and insights.CLASSIFY_RUN.exists()),
    reason="gold/split/run artifacts not present",
)


def test_rule_section_numbers_handles_inflected_marker() -> None:
    # aggressive() has already turned ৩২৩ into 323 by the time tokens reach the rule.
    assert insights.rule_section_numbers(["দণ্ডবিধির", "323", "ধারায়", "কী", "শাস্তি"]) == ["323"]
    assert insights.rule_section_numbers(["ধারা", "103", "কী"]) == ["103"]


def test_rule_section_numbers_ignores_bare_numbers() -> None:
    assert insights.rule_section_numbers(["আমার", "বয়স", "18", "বছর"]) == []


@needs_data
def test_refit_classifier_reproduces_recorded_predictions() -> None:
    """Refitting must give the same predictions `classify_v6.json` recorded for
    the human-only Logistic Regression -- otherwise the demo shows a different
    model from the one whose macro-F1 it prints beside the prediction."""
    run = json.loads(insights.CLASSIFY_RUN.read_text(encoding="utf-8"))
    recorded = {q["qid"]: q["domain_pred_logistic_regression"]
                for q in run["human_only_fine_grained"]["per_query"]}
    questions = {r["qid"]: r["question_bn"] for r in insights._read_jsonl(insights.GOLD)}

    clf = insights.DomainClassifier.fit()
    mismatches = [q for q, want in recorded.items() if clf.predict_label(questions[q]) != want]
    assert not mismatches, f"{len(mismatches)}/{len(recorded)} predictions differ from the recorded run"


@pytest.mark.skipif(not (insights.TAGGER.exists() and insights.MATRIX.exists()),
                    reason="tagger checkpoint not present")
def test_tagger_outputs_one_valid_tag_per_token() -> None:
    tagger = insights.QueryTagger.load()
    result = tagger.tag("আমার স্বামী মুখে তালাক দিয়েছে, এখন কী করব?")
    assert result.tokens and len(result.tokens) == len(result.tags)
    assert set(result.tags) <= set(tagger.tags)
    assert insights.QueryTagger.tag(tagger, "   ").tokens == []
