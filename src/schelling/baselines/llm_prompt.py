"""B5: an LLM asked for a probability per option (fork D models E1 to E3).

Two forms. `population`: one call per question, the same distribution for every person.
`persona`: one call per (attribute profile, question). Neither sees any human answer.

Every response comes through the disk cache. With no client given the baseline is offline:
a request that is not in the cache raises `CacheMissError`, so scoring never spends money.

Choice options are sent in sorted order and mapped back by label, so the prediction for a
label does not depend on the order the harness presents (and permuted rows hit the cache).
Score levels and yes/no keep their natural order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel

from schelling.baselines.base import Baseline, labels_of
from schelling.llm.fireworks import FireworksClient

_SYSTEM = (
    "You estimate how people answer survey and decision questions. "
    'Reply with JSON only, in the form {"probs": [...]}: one number per option, '
    "in the order the options are listed, summing to 1."
)
# Wording variants; the one used is chosen on a dev slice and then frozen (D01 contract).
_POPULATION = (
    "Estimate the share of US adults who would give each answer to this question.",
    "Across the US adult population, what fraction of people would choose each option below?",
)
_PERSONA = (
    "Estimate the probability that the US adult described below gives each answer.",
    "Consider the person described below. How likely is this person to choose each option?",
)


class CacheMissError(LookupError):
    """Raised offline when a request has no cached response."""


class ParseError(ValueError):
    """The model's reply is not a usable probability vector."""


class ChatClient(Protocol):
    def chat(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int, temperature: float
    ) -> dict[str, Any]: ...


class LLMSpec(BaseModel):
    name: str
    model: str
    form: Literal["population", "persona"]
    wording: int = 0
    cache_dir: str = "cache/llm"
    max_tokens: int = 1500
    eps: float = 1e-4  # floor per option before renormalising, so NLL stays finite
    temperature_cal: float = 1.0  # post-hoc; fitted on the dev slice, never on scored items
    # Persona form only: score a seeded subsample of persons with persona prompts and give
    # everyone else the population-form answer. 0 means every person gets a persona prompt.
    person_subsample: int = 0
    subsample_seed: int = 0
    extra: dict[str, Any] = {}


def build_messages(
    question: str, labels: list[str], attrs: dict[str, Any] | None, wording: int
) -> list[dict[str, str]]:
    variants = _POPULATION if attrs is None else _PERSONA
    if not 0 <= wording < len(variants):
        raise ValueError(f"wording must be in 0..{len(variants) - 1}, got {wording}")
    parts = [variants[wording]]
    if attrs is not None:
        parts.append("Person:\n" + "\n".join(f"  {k}: {v}" for k, v in attrs.items()))
    parts.append(f"Question: {question}")
    parts.append("Options:" + "".join(f"\n- {lab}" for lab in labels))
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_probs(text: str, n: int, eps: float) -> np.ndarray:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ParseError("no JSON object")
    try:
        raw = json.loads(text[start : end + 1])["probs"]
    except (json.JSONDecodeError, KeyError, TypeError) as err:
        raise ParseError("no probs list") from err
    if not isinstance(raw, list) or len(raw) != n:
        raise ParseError(f"expected {n} numbers")
    if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in raw):
        raise ParseError("non-numeric entry")
    p = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ParseError("not a distribution")
    p = p / p.sum() + eps
    out: np.ndarray = p / p.sum()
    return out


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    q = np.power(p, 1.0 / t)
    out: np.ndarray = q / q.sum()
    return out


def _offline_transport(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    raise CacheMissError("request not in the LLM cache; run the paid-call script first")


def _order(qtype: str, labels: list[str]) -> list[str]:
    return sorted(labels) if qtype == "choice" else labels


class LLMPromptBaseline:
    def __init__(self, spec: LLMSpec, client: ChatClient | None = None) -> None:
        self.spec = spec
        self.name = spec.name
        self._client: ChatClient = client or FireworksClient(
            cache_dir=Path(spec.cache_dir), transport=_offline_transport, api_key="offline"
        )
        self._attrs: dict[str, dict[str, Any]] = {}
        self.stats = {"requests": 0, "parse_failures": 0, "fallback_rows": 0}

    @classmethod
    def from_spec(cls, path: str | Path, client: ChatClient | None = None) -> LLMPromptBaseline:
        return cls(LLMSpec.model_validate_json(Path(path).read_text()), client=client)

    def fit(self, train: pd.DataFrame, persons: pd.DataFrame) -> Baseline:
        # Nothing is learned from `train`: only the attribute table is kept, for persona prompts.
        cols = [c for c in persons.columns if c != "person_id"]
        self._attrs = {
            str(r["person_id"]): {c: r[c] for c in cols if pd.notna(r[c])}
            for r in persons.to_dict("records")
        }
        return self

    def subsample(self, frame: pd.DataFrame) -> set[str]:
        """The persons who get persona prompts: all of them, or a fixed seeded subsample."""
        persons = np.sort(frame["person_id"].astype(str).unique())
        n = self.spec.person_subsample
        if self.spec.form == "population":
            return set()
        if n == 0 or n >= len(persons):
            return set(persons)
        rng = np.random.default_rng(self.spec.subsample_seed)
        return set(rng.choice(persons, size=n, replace=False))

    def requests(self, frame: pd.DataFrame) -> list[tuple[str, list[str], dict[str, Any] | None]]:
        """One (question, ordered labels, attrs or None) per row, before de-duplication."""
        chosen = self.subsample(frame)
        out = []
        for qtype, question, options, levels, person in zip(
            frame["qtype"],
            frame["question"],
            frame["options"],
            frame["levels"],
            frame["person_id"],
            strict=True,
        ):
            labels = labels_of(
                qtype,
                None if options is None else list(options),
                None if levels is None else list(levels),
            )
            attrs = self._attrs.get(str(person), {}) if str(person) in chosen else None
            out.append((str(question), _order(qtype, labels), attrs))
        return out

    def fetch(
        self, question: str, labels: list[str], attrs: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The raw cached-or-fresh response for one request (used by the paid-call script)."""
        return self._client.chat(
            self.spec.model,
            build_messages(question, labels, attrs, self.spec.wording),
            max_tokens=self.spec.max_tokens,
            temperature=0.0,
            **self.spec.extra,
        )

    def _ask(self, question: str, labels: list[str], attrs: dict[str, Any] | None) -> np.ndarray:
        resp = self.fetch(question, labels, attrs)
        self.stats["requests"] += 1
        try:
            p = parse_probs(resp["text"], len(labels), self.spec.eps)
        except ParseError:
            self.stats["parse_failures"] += 1
            return np.full(len(labels), 1 / len(labels))
        return apply_temperature(p, self.spec.temperature_cal)

    def predict_proba(self, frame: pd.DataFrame) -> list[np.ndarray]:
        self.stats = {"requests": 0, "parse_failures": 0, "fallback_rows": 0}
        memo: dict[str, np.ndarray] = {}
        out = []
        persona = self.spec.form == "persona"
        for (question, ordered, attrs), qtype, options, levels in zip(
            self.requests(frame), frame["qtype"], frame["options"], frame["levels"], strict=True
        ):
            self.stats["fallback_rows"] += persona and attrs is None
            key = json.dumps([question, ordered, attrs], sort_keys=True, default=str)
            if key not in memo:
                memo[key] = self._ask(question, ordered, attrs)
            by_label = dict(zip(ordered, memo[key], strict=True))
            labels = labels_of(
                qtype,
                None if options is None else list(options),
                None if levels is None else list(levels),
            )
            out.append(np.array([by_label[lab] for lab in labels]))
        if self.stats["parse_failures"]:
            print(f"[{self.name}] parse failures: {self.stats['parse_failures']}", file=sys.stderr)
        return out
