"""Offline question paraphrases for M5 augmentation.

Only the question stem is paraphrased; options and scale labels are never touched, so a
label means the same thing in every wording. Generation happens once, offline, and the
result is stored with the generator id; nothing is generated at training time.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def question_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_messages(question: str, n: int) -> list[dict[str, str]]:
    system = (
        "You rewrite survey and decision-task questions. Keep the meaning, the person "
        "being asked, every number, amount, name and time period exactly the same. Do not "
        "answer the question, add hints, or change what a yes/no or scale answer would mean. "
        "Vary vocabulary and sentence structure. Reply with only a JSON array of strings."
    )
    user = f"Write {n} different rewordings of this question:\n\n{question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_NUMBERS = re.compile(r"\d+(?:[.,]\d+)?")
_WORDS = {
    w: str(i)
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
_WORD = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)


def _numbers(text: str) -> list[str]:
    """Numbers in a text, with spelled-out 0 to 20 counted ("eight" equals 8)."""
    spelled = _WORD.sub(lambda m: _WORDS[m.group(1).lower()], text)
    return sorted(_NUMBERS.findall(spelled))


def filter_candidates(source: str, candidates: list[str]) -> list[str]:
    """Drop copies, duplicates, length outliers and rewordings that change a number."""
    want = _numbers(source)
    seen = {source.strip().lower()}
    kept: list[str] = []
    for raw in candidates:
        text = raw.strip()
        key = text.lower()
        if not text or key in seen:
            continue
        if not 0.4 <= len(text) / max(len(source), 1) <= 2.5:
            continue
        if _numbers(text) != want:
            continue
        seen.add(key)
        kept.append(text)
    return kept


def parse_json_list(text: str) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [x for x in data if isinstance(x, str)]


def spend_usd(usage: dict[str, Any], in_per_m: float, out_per_m: float) -> float:
    tokens_in = float(usage.get("prompt_tokens", 0))
    tokens_out = float(usage.get("completion_tokens", 0))
    return (tokens_in * in_per_m + tokens_out * out_per_m) / 1_000_000


class ParaphraseStore:
    """Append-only JSONL keyed by a hash of the source text; rows carry the generator id."""

    def __init__(self, path: Path, generator: str) -> None:
        self.path, self.generator = path, generator
        self._rows: dict[str, list[str]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["generator"] == generator:
                    self._rows[row["key"]] = row["paraphrases"]

    def get(self, text: str) -> list[str]:
        return self._rows.get(question_key(text), [])

    def __contains__(self, text: str) -> bool:
        return question_key(text) in self._rows

    def add(self, text: str, paraphrases: list[str]) -> None:
        key = question_key(text)
        self._rows[key] = paraphrases
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"key": key, "generator": self.generator, "source": text, "paraphrases": paraphrases}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
