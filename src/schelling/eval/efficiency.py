"""Latency and throughput of a predictor on this machine (CPU)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from schelling.baselines.base import Baseline


def latency(
    predictor: Baseline, frame: pd.DataFrame, n_requests: int = 200, seed: int = 0
) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(frame), size=min(n_requests, len(frame)), replace=False)
    sample = frame.iloc[pick].reset_index(drop=True)
    times = []
    for i in range(len(sample)):
        t0 = time.perf_counter()
        predictor.predict_proba(sample.iloc[[i]])
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    predictor.predict_proba(sample)
    batch = time.perf_counter() - t0
    return {
        "device": "cpu",
        "n_requests": float(len(sample)),
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "decisions_per_s": float(len(times) / sum(times)),
        "batch_decisions_per_s": float(len(sample) / batch),
    }
