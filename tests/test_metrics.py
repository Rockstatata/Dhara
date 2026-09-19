from dhara.metrics import first_hit


def test_first_hit_deduplicates_chunks_of_one_provision() -> None:
    """A long section split into chunks occupies one user-visible rank."""
    ranked_chunks_as_provisions = ["act_s1", "act_s1", "act_s2", "act_s3"]
    assert first_hit(ranked_chunks_as_provisions, {"act_s3"}) == 3
