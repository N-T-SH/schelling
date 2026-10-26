"""Head behaviour on the tiny random backbone (SPEC "Model heads")."""

import pytest
import torch
from tests.helpers import tiny_model, tiny_tokenizer

from schelling.infer.predictor import Predictor
from schelling.schema import Choice, DecisionRequest, HistoryItem, Noul, PersonContext, Score

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(tiny_model(), tiny_tokenizer())


def _req(
    questions: dict, person: PersonContext | None = None, state: object = "a situation"
) -> DecisionRequest:
    return DecisionRequest(person=person, state=state, questions=questions)  # type: ignore[arg-type]


def test_choice_probabilities_sum_to_one(predictor: Predictor) -> None:
    ans = predictor.decide(_req({"c": Choice(prompt="pick", options=["a", "b", "c", "d"])}))["c"]
    assert sum(ans.probs) == pytest.approx(1.0, abs=1e-5) and len(ans.probs) == 4


def test_padded_options_get_zero_probability_in_mixed_size_batches(predictor: Predictor) -> None:
    reqs = [
        _req({"c": Choice(prompt="two?", options=["a", "b"])}),
        _req({"c": Choice(prompt="five?", options=["a", "b", "c", "d", "e"])}),
    ]
    two, five = (r["c"] for r in predictor.decide_batch(reqs, batch_size=8))
    assert len(two.probs) == 2 and len(five.probs) == 5
    # Same batch, so the two-option row was padded to five internally: check the raw tensor.
    batch = predictor._tensors(
        "choice",
        [(0, "c", reqs[0], reqs[0].questions["c"]), (1, "c", reqs[1], reqs[1].questions["c"])],
    )
    probs = predictor.model("choice", **batch)
    assert probs[0, 2:].abs().max().item() == 0.0
    assert probs.sum(-1).tolist() == pytest.approx([1.0, 1.0], abs=1e-5)


def test_noul_p_yes_strictly_inside_unit_interval_and_probs_consistent(
    predictor: Predictor,
) -> None:
    ans = predictor.decide(_req({"n": Noul(proposition="I accept")}))["n"]
    assert ans.p_yes is not None and 0 < ans.p_yes < 1
    assert ans.probs == pytest.approx([1 - ans.p_yes, ans.p_yes])


def test_score_distribution_expected_level_and_monotone_survival(predictor: Predictor) -> None:
    levels = ["lo", "low-mid", "mid", "high-mid", "hi"]
    ans = predictor.decide(_req({"s": Score(prompt="rate", levels=levels)}))["s"]
    assert sum(ans.probs) == pytest.approx(1.0, abs=1e-5)
    assert ans.expected_level is not None and 0 <= ans.expected_level <= len(levels) - 1
    assert ans.expected_level == pytest.approx(
        sum(i * p for i, p in enumerate(ans.probs)), abs=1e-5
    )
    req = _req({"s": Score(prompt="rate", levels=levels)})
    batch = predictor._tensors("score", [(0, "s", req, req.questions["s"])])
    surv = predictor.model.score_survival(**batch)
    assert (surv[:, :-1] >= surv[:, 1:] - 1e-7).all() and (surv >= 0).all() and (surv <= 1).all()


def test_score_padding_gets_zero_mass(predictor: Predictor) -> None:
    short = _req({"s": Score(prompt="r", levels=["a", "b"])})
    long = _req({"s": Score(prompt="r", levels=["a", "b", "c", "d", "e", "f"])})
    items = [(0, "s", short, short.questions["s"]), (1, "s", long, long.questions["s"])]
    probs = predictor.model("score", **predictor._tensors("score", items))
    assert probs[0, 2:].abs().max().item() == 0.0
    assert probs.sum(-1).tolist() == pytest.approx([1.0, 1.0], abs=1e-5)


@pytest.mark.parametrize(
    "person",
    [
        None,
        PersonContext(),
        PersonContext(attributes={}, history=[]),
        PersonContext(
            history=[HistoryItem(question="q " * 300, options=["a", "b"], answer=1)] * 30
        ),
    ],
)
def test_no_nan_or_inf_for_empty_or_huge_context(
    predictor: Predictor, person: PersonContext | None
) -> None:
    state = "word " * 2000  # far beyond the token budget
    out = predictor.decide(
        _req(
            {
                "n": Noul(proposition="p"),
                "c": Choice(prompt="c", options=["a", "b"]),
                "s": Score(prompt="s", levels=["x", "y", "z"]),
            },
            person=person,
            state=state,
        )
    )
    for ans in out.values():
        assert all(torch.isfinite(torch.tensor(ans.probs))) and ans.entropy == ans.entropy


def test_temperature_changes_sharpness_not_the_ranking() -> None:
    cold = Predictor(tiny_model(seed=3, temperature_choice=0.2), tiny_tokenizer())
    warm = Predictor(tiny_model(seed=3, temperature_choice=5.0), tiny_tokenizer())
    q = _req({"c": Choice(prompt="pick", options=["a", "b", "c"])})
    pc, pw = cold.decide(q)["c"], warm.decide(q)["c"]
    assert max(pc.probs) > max(pw.probs)
    assert sorted(range(3), key=lambda i: pc.probs[i]) == sorted(
        range(3), key=lambda i: pw.probs[i]
    )


def test_untrained_model_is_not_degenerate(predictor: Predictor) -> None:
    """Guards against vacuous invariance tests: distinct inputs must give distinct outputs."""
    out = predictor.decide(
        _req(
            {
                "c": Choice(prompt="pick", options=["alpha", "beta", "gamma"]),
                "s": Score(prompt="rate", levels=["low", "mid", "high"]),
            }
        )
    )
    assert max(out["c"].probs) - min(out["c"].probs) > 1e-4
    assert max(out["s"].probs) - min(out["s"].probs) > 1e-4
    other = predictor.decide(
        _req({"c": Choice(prompt="pick", options=["alpha", "beta", "gamma"])}, state="different")
    )
    assert other["c"].probs != out["c"].probs


def test_forward_pass_is_deterministic(predictor: Predictor) -> None:
    req = _req({"c": Choice(prompt="pick", options=["a", "b", "c"]), "n": Noul(proposition="p")})
    a, b = predictor.decide(req), predictor.decide(req)
    assert a["c"].probs == b["c"].probs and a["n"].probs == b["n"].probs  # bitwise equal
