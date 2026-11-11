import json
from pathlib import Path

from schelling.augment.paraphrases import (
    ParaphraseStore,
    build_messages,
    filter_candidates,
    question_key,
    spend_usd,
)


def test_prompt_keeps_only_the_question_stem() -> None:
    msgs = build_messages("Would you take $50 now?", 4)
    text = " ".join(m["content"] for m in msgs)
    assert "Would you take $50 now?" in text and "4" in text


def test_filter_drops_copies_duplicates_and_changed_numbers() -> None:
    src = "Would you take $50 now or $60 in 30 days?"
    cands = [
        src,  # identical
        "Would you rather get $50 today or $60 in 30 days?",  # good
        "would you rather get $50 today or $60 in 30 days?",  # duplicate (case)
        "Would you rather get $50 today or $65 in 30 days?",  # numbers changed
        "Yes?",  # far too short
        "x" * 500,  # far too long
    ]
    assert filter_candidates(src, cands) == ["Would you rather get $50 today or $60 in 30 days?"]


def test_key_is_stable_and_text_sensitive() -> None:
    assert question_key("a b") == question_key("a b")
    assert question_key("a b") != question_key("a c")


def test_store_roundtrip_and_lookup_by_text(tmp_path: Path) -> None:
    store = ParaphraseStore(tmp_path / "p.jsonl", generator="m@v1")
    store.add("Q one?", ["Q 1?", "Question one?"])
    again = ParaphraseStore(tmp_path / "p.jsonl", generator="m@v1")
    assert again.get("Q one?") == ["Q 1?", "Question one?"]
    assert again.get("never seen") == []
    assert json.loads((tmp_path / "p.jsonl").read_text().splitlines()[0])["generator"] == "m@v1"


def test_store_ignores_rows_from_another_generator(tmp_path: Path) -> None:
    ParaphraseStore(tmp_path / "p.jsonl", generator="old").add("Q?", ["R?"])
    assert ParaphraseStore(tmp_path / "p.jsonl", generator="new").get("Q?") == []


def test_spend_uses_the_price_table() -> None:
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert abs(spend_usd(usage, in_per_m=0.15, out_per_m=0.60) - 0.75) < 1e-9


def test_spelled_out_numbers_count_as_the_same_number() -> None:
    src = "Cars are 8 times more likely to cause harm."
    assert filter_candidates(src, ["Cars are eight times as likely to cause harm."]) != []
    assert filter_candidates(src, ["Cars are nine times as likely to cause harm."]) == []
