"""Fraction of the test-retest ceiling on Twin-2K-500 (wave 1 answer vs wave 4 answer)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def retest_fraction(
    wave4: pd.DataFrame, probs: Sequence[np.ndarray], wave1: pd.DataFrame
) -> dict[str, Any]:
    """Both sides are the chance that one draw matches the person's wave-4 answer.

    The ceiling uses the person's own wave-1 answer as the draw; the model uses its probability
    on the wave-4 answer. Wave-4 rows without a wave-1 answer for the same item are ignored.
    """
    first = {
        (p, i): r
        for p, i, r in zip(wave1["person_id"], wave1["item_id"], wave1["response"], strict=True)
    }
    agree, model = [], []
    for p, i, y, pr in zip(
        wave4["person_id"], wave4["item_id"], wave4["response"], probs, strict=True
    ):
        if (p, i) in first:
            agree.append(float(first[(p, i)] == y))
            model.append(float(pr[y]))
    if not agree:
        return {"n": 0, "ceiling": None, "model_agreement": None, "fraction_of_ceiling": None}
    ceiling, agreement = float(np.mean(agree)), float(np.mean(model))
    return {
        "n": len(agree),
        "ceiling": ceiling,
        "model_agreement": agreement,
        "fraction_of_ceiling": agreement / ceiling if ceiling > 0 else None,
    }
