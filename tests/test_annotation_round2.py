from dhara.annotation_round2 import (
    answers_agree,
    normalized_text,
    resolve_answer,
    selected_outside_candidates,
    validate_answer_row,
)


CORPUS = {
    "act_1_s8": {"provision_id": "act_1_s8"},
    "act_1_s54_p0": {"provision_id": "act_1_s54"},
    "act_1_s54_p1": {"provision_id": "act_1_s54"},
    "act_2_s3": {"provision_id": "act_2_s3"},
}


def row(answer: str, *, source: str = "own_search", verdict: str = "answered") -> dict[str, str]:
    return {
        "answer": answer,
        "answer_source": source,
        "verdict": verdict,
        "candidates": "1. <act_1_s8> citation\n2. <act_1_s54_p0> citation",
    }


def test_normalized_text_collapses_unicode_and_whitespace() -> None:
    assert normalized_text("  প্রশ্ন\n  এক  ") == normalized_text("প্রশ্ন এক")


def test_candidate_number_resolves_to_chunk_and_provision() -> None:
    resolved = resolve_answer(row("1", source="candidate"), CORPUS)
    assert resolved.chunk_ids == ("act_1_s8",)
    assert resolved.provision_ids == ("act_1_s8",)


def test_different_subchunks_agree_at_parent_provision_level() -> None:
    left = resolve_answer(row("act_1_s54_p0"), CORPUS)
    right = resolve_answer(row("act_1_s54_p1"), CORPUS)
    assert answers_agree(left, right)


def test_candidate_source_detects_own_search_provision() -> None:
    selected = row("act_2_s3", source="candidate")
    assert selected_outside_candidates(selected, CORPUS)
    assert "candidate source used for a provision outside candidate list" in validate_answer_row(
        selected, CORPUS
    )


def test_none_requires_matching_verdict_and_source() -> None:
    valid = row("none", source="none", verdict="not_found")
    invalid = row("none", source="candidate", verdict="answered")
    assert validate_answer_row(valid, CORPUS) == []
    assert validate_answer_row(invalid, CORPUS)
