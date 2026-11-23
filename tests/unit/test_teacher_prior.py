"""A1 (fork D): a cached teacher distribution per question, added to the two-tower item prior as a
logit offset with a learned weight per task type."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.helpers import tiny_tokenizer, tiny_tt_model

from schelling.infer.baseline_adapter import frame_items, person_attributes, predict_items
from schelling.infer.predictor import Predictor
from schelling.infer.teacher import (
    load_teacher,
    save_teacher,
    teacher_key,
    teacher_lookup,
    to_canonical,
)
from schelling.model.heads import (
    choice_probs,
    noul_probs,
    score_conditionals,
    score_probs,
    teacher_offset,
)
from schelling.model.heads_tt import _last
from schelling.train.cached_trainer import (
    CachedData,
    build_cache,
    cached_probs,
    cached_val_nll,
    fit_combiner,
)

pytestmark = pytest.mark.unit
H = 32


def _probs(rows: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    k = max(len(r) for r in rows)
    p = torch.tensor([r + [0.0] * (k - len(r)) for r in rows])
    return p, torch.tensor([[i < len(r) for i in range(k)] for r in rows])


def test_offset_alone_reproduces_the_teacher_distribution() -> None:
    p, mask = _probs([[0.2, 0.8], [0.6, 0.4]])
    assert noul_probs(teacher_offset("noul", p, None), 1.0) == pytest.approx(p, abs=1e-6)
    p, mask = _probs([[0.1, 0.6, 0.3], [0.5, 0.5]])
    out = choice_probs(teacher_offset("choice", p, mask), mask, 1.0)
    assert out == pytest.approx(p, abs=1e-6)
    p, mask = _probs([[0.1, 0.2, 0.3, 0.4], [0.7, 0.2, 0.1]])
    cond = score_conditionals(teacher_offset("score", p, mask), mask, 1.0)
    assert score_probs(cond) == pytest.approx(p, abs=1e-6)


def _inputs(b: int, k: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "u": torch.randn(b, H, generator=g),
        "attr_vecs": torch.randn(b, 3, H, generator=g),
        "attr_mask": torch.ones(b, 3, dtype=torch.bool),
        "e": torch.randn(b, k, H, generator=g),
        "option_mask": torch.ones(b, k, dtype=torch.bool),
    }


def test_with_a_silent_prior_network_the_model_starts_as_the_teacher() -> None:
    m = tiny_tt_model(teacher_prior=True)
    for net in (m.heads.noul_prior, m.heads.choice_prior, m.heads.score_prior):
        torch.nn.init.zeros_(_last(net).weight)
        torch.nn.init.zeros_(_last(net).bias)
    p, _ = _probs([[0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
    x = _inputs(2, 3)
    with torch.no_grad():
        out = m.combine("choice", **x, teacher_probs=p, teacher_has=torch.ones(2, dtype=torch.bool))
        probs = m.probs_from("choice", out.logits, out.mask)
    assert probs == pytest.approx(p, abs=1e-5)


def test_a_question_without_a_teacher_answer_gets_the_plain_prior() -> None:
    plain, a1 = tiny_tt_model(seed=3), tiny_tt_model(seed=3, teacher_prior=True)
    x = _inputs(2, 3)
    p, _ = _probs([[0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
    with torch.no_grad():
        a = plain.combine("choice", **x)
        b = a1.combine("choice", **x, teacher_probs=p, teacher_has=torch.zeros(2, dtype=torch.bool))
    assert torch.equal(a.prior, b.prior)


def test_models_without_the_teacher_have_no_new_weights() -> None:
    assert not any("teacher" in k for k in tiny_tt_model().state_dict())
    assert "heads.teacher_w" in tiny_tt_model(teacher_prior=True).state_dict()


def test_the_key_ignores_choice_order_but_not_level_order_or_the_options() -> None:
    # Choice options have no order (SPEC: permuting options permutes probabilities), so a
    # shuffled question must find the same teacher answer; score levels are ordered.
    k = teacher_key("choice", "Pick one", ["a", "b"])
    assert k == teacher_key("choice", "Pick one", ["b", "a"])
    assert k != teacher_key("choice", "Pick one", ["a", "c"])
    assert teacher_key("score", "Rate", ["a", "b"]) != teacher_key("score", "Rate", ["b", "a"])
    assert k != teacher_key("score", "Pick one", ["a", "b"])


def test_lookup_returns_the_teacher_answer_in_the_asked_option_order() -> None:
    table = {
        teacher_key("choice", "Q", ["x", "y", "z"]): to_canonical(
            "choice", ["x", "y", "z"], [0.2, 0.5, 0.3]
        )
    }
    assert teacher_lookup(table, "choice", "Q", ["z", "x", "y"]) == pytest.approx([0.3, 0.2, 0.5])
    assert teacher_lookup(table, "choice", "Q", ["x", "y"]) is None


def test_permuting_choice_options_permutes_a1_probabilities() -> None:
    from schelling.infer.baseline_adapter import make_item

    model, tok = tiny_tt_model(seed=2, residual_std=0.3, teacher_prior=True), tiny_tokenizer()
    opts = ["red", "green", "blue", "yellow"]
    model.teacher_table = {
        teacher_key("choice", "pick a colour", opts): to_canonical(
            "choice", opts, [0.1, 0.2, 0.3, 0.4]
        )
    }
    rng = np.random.default_rng(0)
    base = predict_items(model, tok, [make_item("choice", "pick a colour", opts, None, "", {})])[0]
    for _ in range(5):
        perm = list(rng.permutation(4))
        shuffled = [opts[i] for i in perm]
        out = predict_items(
            model, tok, [make_item("choice", "pick a colour", shuffled, None, "", {})]
        )[0]
        assert out == pytest.approx(base[perm], abs=1e-5)


def _table(frame) -> dict[str, list[float]]:  # type: ignore[no-untyped-def]
    """A made-up teacher answer for every second item of the fixture."""
    out: dict[str, list[float]] = {}
    items = frame.drop_duplicates("item_id").reset_index(drop=True)
    for i, (qtype, text, o, lv) in enumerate(
        items[["qtype", "question", "options", "levels"]].itertuples(index=False)
    ):
        if i % 2:
            continue
        labels = [] if qtype == "noul" else list(o if qtype == "choice" else lv)
        n = 2 if qtype == "noul" else len(labels)
        p = np.random.default_rng(i).dirichlet(np.ones(n))
        out[teacher_key(qtype, text, labels)] = to_canonical(qtype, labels, p.tolist())
    return out


def test_cached_path_equals_live_inference_with_the_teacher(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    rows = frame.head(150).reset_index(drop=True)
    attrs = person_attributes(persons)
    table = _table(rows)
    model, tok = tiny_tt_model(seed=5, residual_std=0.5, teacher_prior=True), tiny_tokenizer()
    with torch.no_grad():
        model.heads.teacher_w.copy_(torch.tensor([0.7, 1.3, 0.9]))
    model.teacher_table = table
    data = build_cache(model, tok, rows, attrs, torch.device("cpu"), teacher=table)
    assert 0 < int(data.t_has.sum()) < data.T.shape[0]
    cached = cached_probs(model, data, np.arange(len(rows)))
    live = predict_items(model, tok, frame_items(rows, attrs), batch_size=16)
    for c, v in zip(cached, live, strict=True):
        assert c == pytest.approx(v, abs=1e-5)
    model.teacher_table = {}  # the teacher must matter, or the comparison above proves nothing
    without = predict_items(model, tok, frame_items(rows, attrs), batch_size=16)
    assert any(not np.allclose(a, b) for a, b in zip(live, without, strict=True))


def test_the_teacher_table_is_saved_and_loaded_with_the_model(tmp_path: Path) -> None:
    model = tiny_tt_model(teacher_prior=True)
    model.teacher_table = {teacher_key("noul", "It rains", []): [0.3, 0.7]}
    Predictor(model, tiny_tokenizer()).save_pretrained(tmp_path)
    assert json.loads((tmp_path / "teacher_prior.json").read_text())["table"]
    loaded = Predictor.from_pretrained(tmp_path).model
    assert loaded.teacher_table == model.teacher_table
    assert load_teacher(tmp_path / "missing.json") == {}
    save_teacher(tmp_path / "t.json", {"k": [1.0]}, {"spec": "x"})
    assert load_teacher(tmp_path / "t.json") == {"k": [1.0]}


def _unseen_items(n_items: int, rows_per: int, seed: int) -> tuple[CachedData, torch.Tensor]:
    """Noul items whose question vectors carry no signal; the teacher knows each item's rate."""
    g = torch.Generator().manual_seed(seed)
    rate = torch.rand(n_items, generator=g) * 0.9 + 0.05
    item = torch.arange(n_items).repeat_interleave(rows_per)
    y = (torch.rand(len(item), generator=g) < rate[item]).long()
    n = len(item)
    return (
        CachedData(
            U=torch.randn(n_items, H, generator=g),
            E=torch.zeros(0, H),
            A=torch.randn(1, H, generator=g),
            q_idx=item,
            person_idx=torch.zeros(n, dtype=torch.long),
            task=torch.zeros(n, dtype=torch.long),
            y=y,
            opt_idx=torch.zeros(n, 1, dtype=torch.long) - 1,
            attr_idx=torch.tensor([[0]]),
            study=np.array(["s"] * n),
            source=np.array(["twin2k"] * n),
            T=torch.stack([1 - rate, rate], dim=1),
            t_idx=item,
            t_has=torch.ones(n_items, dtype=torch.bool),
        ),
        rate,
    )


def test_the_weight_is_learned_and_helps_on_unseen_items() -> None:
    train, _ = _unseen_items(40, 200, seed=0)
    val, _ = _unseen_items(20, 200, seed=1)
    nll = {}
    for use in (False, True):
        m = tiny_tt_model(seed=0, teacher_prior=use)
        fit_combiner(
            m,
            train,
            steps=300,
            batch_size=512,
            lr=3e-3,
            prior_weight=1.0,
            seed=0,
            study_temperature=0.5,
        )
        nll[use], _ = cached_val_nll(m, val, batch_size=1000)
    assert nll[True] < nll[False] - 0.05
    assert m.heads.teacher_w[0].item() > 0.5  # noul weight stays in use
