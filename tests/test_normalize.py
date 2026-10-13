"""Normalisation bugs are invisible: the text looks fine and retrieval is quietly worse.
These pin the two levels' behaviour on real inputs (CLAUDE.md: unit-test normalize.py)."""

from dhara.normalize import aggressive, light


def test_light_strips_zwnj_and_keeps_bangla_digits() -> None:
    assert light("১০৩ ধারা অনুযায়ী, তালাক‌ দেওয়া হলো।") == "১০৩ ধারা অনুযায়ী, তালাক দেওয়া হলো।"


def test_light_does_not_lowercase_or_strip_punctuation() -> None:
    text = 'Section 103 of the Penal Code, 1860 — "Talaq"'
    assert light(text) == text


def test_aggressive_maps_bangla_digits_to_ascii() -> None:
    # A user writing "১০৩ ধারা" must reach the same chunk as one writing "section 103".
    assert "103" in aggressive("১০৩ ধারা").split()
    assert aggressive("১০৩") == aggressive("103")


def test_aggressive_pads_danda_and_lowercases() -> None:
    assert aggressive("১০৩ ধারা অনুযায়ী, তালাক‌ দেওয়া হলো।") == "103 ধারা অনুযায়ী তালাক দেওয়া হলো ।"
    assert aggressive('Section 103 of the Penal Code, 1860 — "Talaq"') == "section 103 of the penal code 1860 talaq"


def test_both_levels_are_idempotent() -> None:
    text = "আমার স্বামী তালাক দিয়েছে। কী করব?"
    assert light(light(text)) == light(text)
    assert aggressive(aggressive(text)) == aggressive(text)
