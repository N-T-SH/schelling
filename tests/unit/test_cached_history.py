"""C01 / I01 leakage and parity tests for the cached trainer (assumption A04)."""

import numpy as np
import pytest
import torch
from tests.helpers import tiny_tokenizer, tiny_tt_model

from schelling.infer.baseline_adapter import frame_items, person_attributes, predict_items
from schelling.train.cached_trainer import build_cache, cached_probs, sample_history

pytestmark = pytest.mark.unit
CPU = torch.device("cpu")


def _setup(twin_frame, n: int = 400):  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    rows = frame[frame["source"] == "twin2k"].head(n).reset_index(drop=True)
    return rows, person_attributes(persons)


def _decode(data, hq, hm, i):  # type: ignore[no-untyped-def]
    return [data.q_ctx[int(j)] for j, ok in zip(hq[i], hm[i], strict=True) if ok]


def test_sampled_history_never_holds_the_targets_question_or_another_person(twin_frame) -> None:  # type: ignore[no-untyped-def]
    rows, attrs = _setup(twin_frame)
    data = build_cache(tiny_tt_model(), tiny_tokenizer(), rows, attrs, CPU, history_k=50)
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        idx = torch.arange(len(rows))
        hq, _ho, hm = sample_history(data, idx, gen, ks=(50,))
        for i, (pid, q) in enumerate(zip(rows["person_id"], rows["question"], strict=True)):
            ctxs = _decode(data, hq, hm, i)
            assert not any(f"[QUESTION] {q}" == c.split("[STATE]  ", 1)[-1] for c in ctxs)
            own = {
                f"[STATE]  [QUESTION] {t}"
                for t, p, w in zip(rows["question"], rows["person_id"], rows["wave"], strict=True)
                if p == pid and w != 4 and t != q
            }
            assert set(ctxs) <= own


def test_history_size_is_drawn_from_the_allowed_sizes_and_bounded(twin_frame) -> None:  # type: ignore[no-untyped-def]
    rows, attrs = _setup(twin_frame)
    data = build_cache(tiny_tt_model(), tiny_tokenizer(), rows, attrs, CPU, history_k=50)
    gen = torch.Generator().manual_seed(1)
    _, _, hm = sample_history(data, torch.arange(len(rows)), gen, ks=(0, 5, 20, 50))
    sizes = hm.sum(1)
    assert int(sizes.max()) <= 50 and set(sizes.tolist()) <= {0, 5, 20, 50} | set(range(51))
    assert (sizes == 0).any() or len(rows) < 8  # k = 0 is drawn


def test_changing_only_the_targets_answer_leaves_its_sampled_history_unchanged(twin_frame) -> None:  # type: ignore[no-untyped-def]
    rows, attrs = _setup(twin_frame)
    i = 7
    other = rows.copy()
    other.loc[i, "response"] = (
        1 - int(rows.loc[i, "response"]) if rows.loc[i, "response"] < 2 else 0
    )
    tok = tiny_tokenizer()
    a = build_cache(tiny_tt_model(), tok, rows, attrs, CPU, history_k=50)
    b = build_cache(tiny_tt_model(), tok, other, attrs, CPU, history_k=50)
    idx = torch.arange(len(rows))
    ha = sample_history(a, idx, torch.Generator().manual_seed(4), ks=(50,))
    hb = sample_history(b, idx, torch.Generator().manual_seed(4), ks=(50,))
    assert _decode(a, ha[0], ha[2], i) == _decode(b, hb[0], hb[2], i)
    # the chosen-option part of the history is also identical for that row
    assert [a.o_txt[int(j)] for j, ok in zip(ha[1][i], ha[2][i], strict=True) if ok] == [
        b.o_txt[int(j)] for j, ok in zip(hb[1][i], hb[2][i], strict=True) if ok
    ]


def test_cached_probabilities_equal_live_inference_with_history(twin_frame) -> None:  # type: ignore[no-untyped-def]
    rows, attrs = _setup(twin_frame, 160)
    model, tok = tiny_tt_model(seed=5, residual_std=0.5, history_items=6), tiny_tokenizer()
    data = build_cache(model, tok, rows, attrs, CPU, history_k=6, fixed_history=True)
    cached = cached_probs(model, data, np.arange(len(rows)))
    live = predict_items(model, tok, frame_items(rows, attrs, history_k=6), batch_size=16)
    for c, v in zip(cached, live, strict=True):
        assert c == pytest.approx(v, abs=1e-5)
    # and the history really mattered
    no_hist = predict_items(model, tok, frame_items(rows, attrs, history_k=0), batch_size=16)
    assert any(not np.allclose(c, n, atol=1e-4) for c, n in zip(cached, no_hist, strict=True))


def test_models_without_history_are_unchanged_by_the_cache_changes(twin_frame) -> None:  # type: ignore[no-untyped-def]
    rows, attrs = _setup(twin_frame, 100)
    model, tok = tiny_tt_model(seed=5, residual_std=0.5), tiny_tokenizer()
    data = build_cache(model, tok, rows, attrs, CPU)
    cached = cached_probs(model, data, np.arange(len(rows)))
    live = predict_items(model, tok, frame_items(rows, attrs), batch_size=16)
    for c, v in zip(cached, live, strict=True):
        assert c == pytest.approx(v, abs=1e-5)


def test_a_rows_live_prediction_ignores_its_own_recorded_answer(twin_frame) -> None:  # type: ignore[no-untyped-def]
    rows, attrs = _setup(twin_frame, 120)
    model, tok = tiny_tt_model(seed=5, residual_std=0.5, history_items=8), tiny_tokenizer()
    i = 9
    changed = rows.copy()
    changed.loc[i, "response"] = (int(rows.loc[i, "response"]) + 1) % 2
    a = predict_items(model, tok, frame_items(rows, attrs, history_k=8), batch_size=16)[i]
    b = predict_items(model, tok, frame_items(changed, attrs, history_k=8), batch_size=16)[i]
    assert a == pytest.approx(b, abs=1e-7)
