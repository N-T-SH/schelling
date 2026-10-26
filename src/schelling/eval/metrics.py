"""Scoring rules, calibration and distribution distances (SPEC "Evaluation plan").

Sealed after M2: changes need an `eval-change/<topic>` branch and human approval.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _p_true(probs: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    return np.array([p[i] for p, i in zip(probs, y, strict=True)])


def nll(probs: list[np.ndarray], y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(_p_true(probs, y), EPS, 1.0))))


def brier(probs: list[np.ndarray], y: np.ndarray) -> float:
    total = 0.0
    for p, i in zip(probs, y, strict=True):
        onehot = np.zeros_like(p)
        onehot[i] = 1.0
        total += float(((p - onehot) ** 2).sum())
    return total / len(probs)


def accuracy(probs: list[np.ndarray], y: np.ndarray) -> float:
    return float(np.mean([int(np.argmax(p)) == i for p, i in zip(probs, y, strict=True)]))


def top_label(probs: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.array([p.max() for p in probs])
    correct = np.array([int(np.argmax(p)) == i for p, i in zip(probs, y, strict=True)], dtype=int)
    return conf, correct


def reliability_bins(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> list[dict[str, float]]:
    """Adaptive (equal-mass) bins. Identical confidences always share a bin, otherwise an
    arbitrary tie order would invent calibration error (a uniform predictor has one value)."""
    order = np.argsort(conf, kind="stable")
    s, n = conf[order], len(conf)
    cuts, prev = [], 0
    for k in range(1, n_bins):
        c = max(round(k * n / n_bins), prev)
        while 0 < c < n and s[c - 1] == s[c]:
            c += 1
        cuts.append(c)
        prev = c
    bins = [b for b in np.split(order, cuts) if len(b)]
    return [
        {
            "n": float(len(b)),
            "confidence": float(conf[b].mean()),
            "accuracy": float(correct[b].mean()),
        }
        for b in bins
    ]


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = reliability_bins(conf, correct, n_bins)
    n = sum(b["n"] for b in bins)
    return float(sum(b["n"] / n * abs(b["confidence"] - b["accuracy"]) for b in bins))


def calibration_slope_intercept(conf: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Logistic regression of correctness on logit(confidence), fitted by Newton's method."""
    c = np.clip(conf, 1e-6, 1 - 1e-6)
    x = np.log(c / (1 - c))
    design = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-design @ beta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.linalg.solve(
            design.T @ (design * w[:, None]) + 1e-9 * np.eye(2), design.T @ (correct - mu)
        )
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return float(beta[1]), float(beta[0])


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def wasserstein1(p: np.ndarray, q: np.ndarray) -> float:
    """Earth mover's distance over ordered options with unit spacing."""
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def _sd(p: np.ndarray) -> float:
    idx = np.arange(len(p))
    mean = (p * idx).sum()
    return float(np.sqrt((p * (idx - mean) ** 2).sum()))


def sd_ratio(p_synthetic: np.ndarray, p_human: np.ndarray) -> float:
    h = _sd(p_human)
    return float("nan") if h == 0 else _sd(p_synthetic) / h


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def entropy_ratio(p_synthetic: np.ndarray, p_human: np.ndarray) -> float:
    h = _entropy(p_human)
    return float("nan") if h == 0 else _entropy(p_synthetic) / h
