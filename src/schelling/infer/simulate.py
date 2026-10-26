"""Sample a synthetic population: one answer distribution per (person, question), n draws each."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schelling.infer.predictor import Predictor
from schelling.schema import Choice, DecisionRequest, PersonContext, Score


def _labels(q: object, n: int) -> list[str]:
    if isinstance(q, Choice):
        return q.options
    if isinstance(q, Score):
        return q.levels
    return ["no", "yes"][:n]


def sample_population(
    predictor: Predictor,
    persons: list[PersonContext],
    request_template: DecisionRequest,
    n_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Rows: person_id, question, draw, response (index), label. Deterministic given `seed`."""
    requests = [request_template.model_copy(update={"person": p}) for p in persons]
    answers = predictor.decide_batch(requests)
    rng = np.random.default_rng(seed)
    rows = []
    for k, (person, ans) in enumerate(zip(persons, answers, strict=True)):
        for qid, a in ans.items():
            labels = _labels(request_template.questions[qid], len(a.probs))
            draws = rng.choice(len(a.probs), size=n_draws, p=np.asarray(a.probs) / sum(a.probs))
            for d, r in enumerate(draws):
                rows.append(
                    {
                        "person_id": person.person_id or f"person-{k}",
                        "question": qid,
                        "draw": d,
                        "response": int(r),
                        "label": labels[r],
                    }
                )
    return pd.DataFrame(rows)
