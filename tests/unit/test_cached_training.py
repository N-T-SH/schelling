"""Cached (frozen-backbone) training: it must agree with live inference and must learn."""

import numpy as np
import pytest
import torch
from tests.helpers import tiny_tokenizer, tiny_tt_model

from schelling.infer.baseline_adapter import frame_items, person_attributes, predict_items
from schelling.train.cached_trainer import (
    CachedData,
    build_cache,
    cached_probs,
    cached_val_nll,
    fit_combiner,
)

pytestmark = pytest.mark.unit
H = 32


def test_cached_path_equals_live_inference_on_real_records(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    rows = frame.head(120).reset_index(drop=True)
    attrs = person_attributes(persons)
    model, tok = tiny_tt_model(seed=5, residual_std=0.5), tiny_tokenizer()
    data = build_cache(model, tok, rows, attrs, torch.device("cpu"))
    cached = cached_probs(model, data, np.arange(len(rows)))
    live = predict_items(model, tok, frame_items(rows, attrs), batch_size=16)
    for c, v in zip(cached, live, strict=True):
        assert c == pytest.approx(v, abs=1e-5)


def test_cache_reuses_vectors_for_repeated_texts(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    attrs = person_attributes(persons)
    data = build_cache(tiny_tt_model(), tiny_tokenizer(), frame, attrs, torch.device("cpu"))
    assert data.U.shape[0] < len(frame) / 3 and data.A.shape[0] < len(attrs) * 14


def _synthetic(n: int = 4000, seed: int = 0) -> CachedData:
    """One item with two options; the answer depends on a person attribute only."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(2, H, generator=g)  # attribute "g: a" and "g: b"
    person_idx = torch.arange(n) % 2
    p_yes = torch.where(person_idx == 0, 0.9, 0.1)
    y = (torch.rand(n, generator=g) < p_yes).long()
    return CachedData(
        U=torch.randn(1, H, generator=g),
        E=torch.randn(2, H, generator=g),
        A=A,
        q_idx=torch.zeros(n, dtype=torch.long),
        person_idx=person_idx,
        task=torch.ones(n, dtype=torch.long),
        y=y,
        opt_idx=torch.tensor([[0, 1]]).repeat(n, 1),
        attr_idx=torch.tensor([[0], [1]]),
        study=np.array(["s"] * n),
        source=np.array(["twin2k"] * n),
    )


def test_person_signal_is_learned_by_the_residual_not_the_prior() -> None:
    data = _synthetic()
    model = tiny_tt_model(seed=0)
    fit_combiner(
        model,
        data,
        steps=300,
        batch_size=256,
        lr=3e-3,
        prior_weight=1.0,
        seed=0,
        study_temperature=0.5,
    )
    full, prior = cached_val_nll(model, data, batch_size=1000)
    assert full < 0.45  # the answer is 90% predictable from the attribute
    assert 0.65 < prior < 0.72  # the prior alone cannot know the person: about ln 2


def test_without_the_person_tower_the_model_cannot_use_the_person() -> None:
    data = _synthetic()
    model = tiny_tt_model(seed=0, use_person=False)
    fit_combiner(
        model,
        data,
        steps=300,
        batch_size=256,
        lr=3e-3,
        prior_weight=1.0,
        seed=0,
        study_temperature=0.5,
    )
    full, _ = cached_val_nll(model, data, batch_size=1000)
    assert full > 0.65


def test_prior_loss_makes_the_prior_learn_the_item_marginal() -> None:
    data = _synthetic()
    with_aux = tiny_tt_model(seed=0)
    fit_combiner(
        with_aux,
        data,
        steps=200,
        batch_size=256,
        lr=3e-3,
        prior_weight=1.0,
        seed=0,
        study_temperature=0.5,
    )
    _, prior = cached_val_nll(with_aux, data, batch_size=1000)
    assert prior == pytest.approx(0.693, abs=0.03)  # marginal is 50/50


def test_fitting_is_deterministic_given_the_seed() -> None:
    data = _synthetic(1000)
    outs = []
    for _ in range(2):
        m = tiny_tt_model(seed=0)
        curve = fit_combiner(
            m,
            data,
            steps=40,
            batch_size=128,
            lr=3e-3,
            prior_weight=1.0,
            seed=3,
            study_temperature=0.5,
        )
        outs.append(curve)
    assert outs[0] == outs[1]
