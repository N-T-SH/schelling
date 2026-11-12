"""B5 / fork D external-model baseline: prompts, parsing, cache replay. No network."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from schelling.baselines import make_baseline
from schelling.baselines.llm_prompt import (
    CacheMissError,
    LLMPromptBaseline,
    LLMSpec,
    ParseError,
    apply_temperature,
    build_messages,
    parse_probs,
)
from schelling.llm.fireworks import FireworksClient

pytestmark = pytest.mark.unit

ATTRS = {"sex": "Female", "age_band": "30-49", "region": "Midwest"}


def _frame() -> pd.DataFrame:
    rows = []
    for person in ("p1", "p2", "p3"):
        rows.append(
            {
                "item_id": "c1",
                "person_id": person,
                "qtype": "choice",
                "question": "Which fruit do you prefer?",
                "options": ["pear", "apple", "fig"],
                "levels": None,
            }
        )
        rows.append(
            {
                "item_id": "s1",
                "person_id": person,
                "qtype": "score",
                "question": "I am talkative.",
                "options": None,
                "levels": ["Disagree", "Neutral", "Agree"],
            }
        )
        rows.append(
            {
                "item_id": "n1",
                "person_id": person,
                "qtype": "noul",
                "question": "Do you own a car?",
                "options": None,
                "levels": None,
            }
        )
    return pd.DataFrame(rows)


def _persons() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"person_id": "p1", **ATTRS},
            {"person_id": "p2", **ATTRS, "sex": "Male"},
            {"person_id": "p3", **ATTRS},
        ]
    )


class FakeLLM:
    """Answers with the option count encoded, so tests can check the mapping back to rows."""

    def __init__(self, reply: Any = None) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.reply = reply

    def __call__(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body["messages"])
        user = body["messages"][-1]["content"]
        n = user.count("\n- ")
        text = self.reply or json.dumps({"probs": [float(i + 1) for i in range(n)]})
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _spec(tmp_path: Path, **kw: Any) -> LLMSpec:
    kw.setdefault("form", "population")
    return LLMSpec(name="e-test", model="m", cache_dir=str(tmp_path), **kw)


def _online(spec: LLMSpec, llm: FakeLLM) -> LLMPromptBaseline:
    client = FireworksClient(cache_dir=Path(spec.cache_dir), transport=llm, api_key="k", retries=1)
    return LLMPromptBaseline(spec, client=client)


# ---- prompt -------------------------------------------------------------------------------


def test_population_prompt_has_question_and_options_but_no_person() -> None:
    msgs = build_messages("Which fruit?", ["apple", "fig"], None, wording=0)
    user = msgs[-1]["content"]
    assert "Which fruit?" in user and "\n- apple" in user and "\n- fig" in user
    assert "Female" not in user


def test_persona_prompt_lists_only_the_given_attributes() -> None:
    user = build_messages("Which fruit?", ["apple", "fig"], ATTRS, wording=0)[-1]["content"]
    for value in ATTRS.values():
        assert value in user
    assert "person_id" not in user


def test_wordings_differ_and_unknown_wording_is_rejected() -> None:
    a = build_messages("Q?", ["x", "y"], None, wording=0)
    b = build_messages("Q?", ["x", "y"], None, wording=1)
    assert a != b
    with pytest.raises(ValueError):
        build_messages("Q?", ["x", "y"], None, wording=99)


# ---- parser -------------------------------------------------------------------------------


def test_parse_renormalises_and_floors() -> None:
    p = parse_probs('noise {"probs": [2, 1, 1, 0]} trailing', 4, eps=0.01)
    assert p.sum() == pytest.approx(1.0)
    assert p[1] == pytest.approx(p[2])
    assert p[0] > p[1] > p[3] > 0
    assert p.min() >= 0.01 / 1.04 - 1e-12


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"probs": [0.5, 0.5]}',  # wrong length
        '{"probs": [0.5, -0.1, 0.6]}',  # negative
        '{"probs": [0, 0, 0]}',  # no mass
        '{"probs": [0.5, "a", 0.5]}',  # not numbers
        '{"other": [0.2, 0.3, 0.5]}',  # wrong key
    ],
)
def test_parse_rejects_malformed(text: str) -> None:
    with pytest.raises(ParseError):
        parse_probs(text, 3, eps=1e-4)


def test_temperature_one_is_identity_and_high_flattens() -> None:
    p = np.array([0.7, 0.2, 0.1])
    assert apply_temperature(p, 1.0) == pytest.approx(p)
    flat = apply_temperature(p, 4.0)
    assert flat.sum() == pytest.approx(1.0)
    assert flat.max() < p.max() and flat.min() > p.min()


# ---- baseline -----------------------------------------------------------------------------


def test_population_form_makes_one_call_per_item_and_maps_back(tmp_path: Path) -> None:
    llm = FakeLLM()
    base = _online(_spec(tmp_path), llm).fit(pd.DataFrame(), _persons())
    probs = base.predict_proba(_frame())
    assert len(llm.calls) == 3  # three items, nine rows
    assert [len(p) for p in probs[:3]] == [3, 3, 2]
    for p in probs:
        assert p.sum() == pytest.approx(1.0)
    # Score levels keep their order: the fake gives increasing mass by position.
    assert probs[1][0] < probs[1][1] < probs[1][2]


def test_choice_prediction_follows_the_label_under_option_permutation(tmp_path: Path) -> None:
    llm = FakeLLM()
    base = _online(_spec(tmp_path), llm).fit(pd.DataFrame(), _persons())
    frame = _frame().iloc[[0]].reset_index(drop=True)
    first = dict(zip(frame["options"][0], base.predict_proba(frame)[0], strict=True))
    shuffled = frame.copy()
    shuffled.at[0, "options"] = ["fig", "pear", "apple"]
    second = dict(zip(shuffled["options"][0], base.predict_proba(shuffled)[0], strict=True))
    assert first == pytest.approx(second)
    assert len(llm.calls) == 1  # the permuted row hits the cache


def test_persona_form_calls_once_per_distinct_profile(tmp_path: Path) -> None:
    llm = FakeLLM()
    spec = _spec(tmp_path, form="persona")
    base = _online(spec, llm).fit(pd.DataFrame(), _persons())
    base.predict_proba(_frame())
    assert len(llm.calls) == 6  # 3 items x 2 distinct profiles (p1 and p3 share one)
    assert any("Male" in c[-1]["content"] for c in llm.calls)


def test_offline_replay_needs_no_network_and_a_miss_raises(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    online = _online(spec, FakeLLM()).fit(pd.DataFrame(), _persons())
    want = online.predict_proba(_frame())
    offline = LLMPromptBaseline(spec).fit(pd.DataFrame(), _persons())
    got = offline.predict_proba(_frame())
    for a, b in zip(want, got, strict=True):
        assert a == pytest.approx(b)
    other = _frame().iloc[[0]].copy()
    other["question"] = "A question never asked?"
    with pytest.raises(CacheMissError):
        offline.predict_proba(other)


def test_unparsable_reply_falls_back_to_uniform_and_is_counted(tmp_path: Path) -> None:
    base = _online(_spec(tmp_path), FakeLLM(reply="sorry, I cannot")).fit(
        pd.DataFrame(), _persons()
    )
    probs = base.predict_proba(_frame().iloc[[0]])
    assert probs[0] == pytest.approx(np.full(3, 1 / 3))
    assert base.stats == {"requests": 1, "parse_failures": 1, "fallback_rows": 0}


def test_subsample_rows_use_persona_and_other_rows_use_the_fallback(tmp_path: Path) -> None:
    llm = FakeLLM()
    spec = _spec(tmp_path, form="persona", person_subsample=1, subsample_seed=0)
    base = _online(spec, llm).fit(pd.DataFrame(), _persons())
    chosen = base.subsample(_frame())
    assert len(chosen) == 1
    base.predict_proba(_frame())
    persona_calls = [c for c in llm.calls if "Midwest" in c[-1]["content"]]
    assert len(persona_calls) == 3  # one person x three items
    assert len(llm.calls) == 6  # plus the three population fallbacks
    assert base.stats["fallback_rows"] == 6


def test_make_baseline_loads_a_spec_file(tmp_path: Path) -> None:
    path = tmp_path / "e-test.json"
    path.write_text(_spec(tmp_path).model_dump_json())
    base = make_baseline(str(path))
    assert isinstance(base, LLMPromptBaseline)
    assert base.name == "e-test"
