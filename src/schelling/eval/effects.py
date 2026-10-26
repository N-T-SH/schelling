"""Treatment-effect agreement between human and synthetic populations.

Needs a pre-registered table of (effect, human estimate, model estimate); no such table exists
yet, so this is tested but not run on data in M2.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def effect_size_agreement(
    human: np.ndarray,
    model: np.ndarray,
    null_mask: np.ndarray,
    model_significant: np.ndarray,
) -> dict[str, Any]:
    real = ~null_mask
    return {
        "sign_agreement": float(np.mean(np.sign(human[real]) == np.sign(model[real])))
        if real.any()
        else None,
        "correlation": float(np.corrcoef(human, model)[0, 1]),
        "false_positive_rate": float(np.mean(model_significant[null_mask]))
        if null_mask.any()
        else None,
    }
