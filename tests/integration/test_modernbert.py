"""A06 / M3 check with the real backbone. Needs the Hugging Face cache or network; not in the fast suite."""

import pytest
from transformers import AutoTokenizer

from schelling.infer.predictor import Predictor
from schelling.model import SchellingModel
from schelling.schema import Choice, DecisionRequest, Noul, PersonContext, Score

pytestmark = [pytest.mark.network, pytest.mark.slow]
NAME = "answerdotai/ModernBERT-base"


def test_modernbert_base_wraps_saves_reloads_and_batches_consistently(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = SchellingModel.from_backbone(NAME, max_context_tokens=256)
    pred = Predictor(model, AutoTokenizer.from_pretrained(NAME))
    reqs = [
        DecisionRequest(
            person=PersonContext(attributes={"age": 30 + i}),
            state="You are offered a deal." * (i + 1),
            questions={
                "n": Noul(proposition="I accept"),
                "c": Choice(prompt="Which?", options=["red", "green", "blue"]),
                "s": Score(prompt="How much?", levels=["none", "some", "a lot"]),
            },
        )
        for i in range(3)
    ]
    batched = pred.decide_batch(reqs, batch_size=8)
    for req, got in zip(reqs, batched, strict=True):
        single = pred.decide(req)
        for k in single:
            assert got[k].probs == pytest.approx(single[k].probs, abs=1e-4)
            assert sum(got[k].probs) == pytest.approx(1.0, abs=1e-5)
    pred.save_pretrained(tmp_path)
    again = Predictor.from_pretrained(tmp_path).decide(reqs[0])
    for k in again:
        assert again[k].probs == pytest.approx(batched[0][k].probs, abs=1e-5)
