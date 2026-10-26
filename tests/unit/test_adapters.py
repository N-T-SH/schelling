"""Adapter tests (SPEC "Data adapters and splits")."""

import json
from pathlib import Path

import pytest

from schelling.data.adapters import psych101, twin2k
from schelling.data.schema import DecisionRecord

pytestmark = pytest.mark.unit


# ---- Psych-101 parser on hand-written trial strings -------------------------------------


def test_parse_single_choice() -> None:
    spans = psych101.parse_choices("D: a, P: b, H: c. You press <<D>>.\nD: x. You press <<P>>.")
    assert [s.choice for s in spans] == ["D", "P"]


def test_parse_multiple_choices_in_one_trial() -> None:
    text = "You pick <<A>> and then <<B>>, finally <<12>>."
    assert [s.choice for s in psych101.parse_choices(text)] == ["A", "B", "12"]


def test_parse_spans_point_at_the_choice_text() -> None:
    text = "x <<abc>> y"
    (span,) = psych101.parse_choices(text)
    assert text[span.start : span.end] == "<<abc>>"


def test_parse_ignores_unclosed_marker() -> None:
    assert psych101.parse_choices("press << never closed") == []


def test_records_use_only_earlier_text_as_state() -> None:
    row = {"text": "T1 <<A>>\nT2 <<B>>\nT3 <<A>>", "experiment": "e/x.csv", "participant": "7"}
    recs = psych101.records_from_rows([row] * 1 + [dict(row, participant="8")])
    first = [r for r in recs if r.person_id.endswith("|7")]
    assert [r.trial_idx for r in first] == [0, 1, 2]
    assert "<<A>>" not in first[0].state.split("T1")[-1]  # the target itself is masked
    assert "T3" not in first[1].state


def test_records_options_are_observed_choice_set() -> None:
    rows = [
        {"text": "q <<A>> q <<B>>", "experiment": "e", "participant": "1"},
        {"text": "q <<B>> q <<C>>", "experiment": "e", "participant": "2"},
    ]
    recs = psych101.records_from_rows(rows)
    assert all(r.options == ["A", "B", "C"] for r in recs)
    assert [r.response for r in recs] == [0, 1, 1, 2]


def test_experiments_with_too_many_choices_are_dropped_and_counted() -> None:
    rows = [
        {"text": " ".join(f"<<{i}>>" for i in range(30)), "experiment": "e", "participant": "1"}
    ]
    report = psych101.AdapterReport()
    assert psych101.records_from_rows(rows, report=report) == []
    assert report.dropped_experiments == {"e": "more than 20 distinct choices"}


def test_psych_fixture_records_validate_and_ids_are_stable(fixtures_dir: Path) -> None:
    rows = [json.loads(line) for line in (fixtures_dir / "psych101_rows.jsonl").open()]
    a = psych101.records_from_rows(rows)
    b = psych101.records_from_rows(rows)
    assert len(a) > 100
    assert [r.record_id for r in a] == [r.record_id for r in b]
    assert len({r.record_id for r in a}) == len(a)
    for r in a:
        DecisionRecord.model_validate(r.model_dump())
        assert r.options is not None and r.response is not None
        assert 0 <= r.response < len(r.options)
        assert r.source == "psych101"


# ---- Twin-2K-500 ------------------------------------------------------------------------


def test_twin_fixture_records_validate_and_map_types(fixtures_dir: Path) -> None:
    out = twin2k.build(fixtures_dir / "twin2k")
    recs, persons = out.records, out.persons
    assert {r.qtype for r in recs} == {"noul", "choice", "score"}
    for r in recs:
        DecisionRecord.model_validate(r.model_dump())
        n = len(r.levels or r.options or [0, 1])
        assert r.response is not None and 0 <= r.response < n
        assert r.source == "twin2k"
    assert out.report.unmapped_columns  # free text / sliders / multi-select are logged, not silent
    assert set(persons["person_id"]) == {r.person_id for r in recs}


def test_twin_ids_stable_across_runs(fixtures_dir: Path) -> None:
    a = twin2k.build(fixtures_dir / "twin2k").records
    b = twin2k.build(fixtures_dir / "twin2k").records
    assert [r.record_id for r in a] == [r.record_id for r in b]


def test_twin_wave4_records_marked_and_repeat_items_share_item_id(fixtures_dir: Path) -> None:
    recs = twin2k.build(fixtures_dir / "twin2k").records
    w4 = [r for r in recs if r.wave == 4]
    w1 = {(r.person_id, r.item_id) for r in recs if r.wave == 1}
    assert w4 and all((r.person_id, r.item_id) in w1 for r in w4)


def test_twin_demographics_become_attributes_not_records(fixtures_dir: Path) -> None:
    out = twin2k.build(fixtures_dir / "twin2k")
    assert "Which part of the United States" not in " ".join(r.question for r in out.records)
    assert any(c.startswith("QID11") or c == "region" for c in out.persons.columns)
