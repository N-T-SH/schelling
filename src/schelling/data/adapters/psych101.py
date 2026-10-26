"""Psych-101 adapter.

One upstream row is a whole participant transcript for one experiment, with every choice
wrapped as `<<choice>>`. We emit one record per span. The prediction target is the span;
`state` is the transcript before it (so earlier choices are visible, later ones are not).
Options are the closed set of choice strings observed in the experiment, since the options
themselves are described in free text and differ per experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schelling.data.schema import DecisionRecord, make_record_id

SOURCE = "psych101"
MAX_OPTIONS = 20
MAX_STATE_CHARS = 2000
_SPAN = re.compile(r"<<(.*?)>>")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    choice: str


@dataclass
class AdapterReport:
    n_rows: int = 0
    n_records: int = 0
    dropped_experiments: dict[str, str] = field(default_factory=dict)


def parse_choices(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), m.group(1)) for m in _SPAN.finditer(text)]


def read_rows(path: Path, max_rows: int | None = None, keep_fraction: float = 1.0) -> Iterator[Any]:
    """Stream the jsonl. `keep_fraction` keeps whole participants chosen by hash, so a
    subsample never splits one person's transcript."""
    n = 0
    with path.open("rb") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # a truncated final line of a partial download
            if keep_fraction < 1.0:
                key = f"{row['experiment']}|{row['participant']}".encode()
                if int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64 >= keep_fraction:
                    continue
            yield row
            n += 1
            if max_rows is not None and n >= max_rows:
                return


def records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    report: AdapterReport | None = None,
    max_state_chars: int = MAX_STATE_CHARS,
) -> list[DecisionRecord]:
    rep = report if report is not None else AdapterReport()
    by_exp: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rep.n_rows += 1
        by_exp[str(row["experiment"])].append(row)

    out: list[DecisionRecord] = []
    for exp in sorted(by_exp):
        parsed = [(row, parse_choices(row["text"])) for row in by_exp[exp]]
        choices = sorted({s.choice for _, spans in parsed for s in spans})
        if len(choices) > MAX_OPTIONS:
            rep.dropped_experiments[exp] = f"more than {MAX_OPTIONS} distinct choices"
            continue
        if len(choices) < 2:
            rep.dropped_experiments[exp] = "fewer than 2 distinct choices"
            continue
        index = {c: i for i, c in enumerate(choices)}
        counter: dict[str, int] = defaultdict(int)  # trial index continues across a person's rows
        for row, spans in parsed:
            text, person = row["text"], f"{exp}|{row['participant']}"
            for s in spans:
                line_start = text.rfind("\n", 0, s.start) + 1
                question = text[line_start : s.start].strip()
                idx = counter[person]
                counter[person] += 1
                item_id = f"{exp}:{hashlib.sha1(question.encode()).hexdigest()[:12]}"
                out.append(
                    DecisionRecord(
                        record_id=make_record_id(SOURCE, exp, person, item_id, idx),
                        source=SOURCE,
                        study_id=exp,
                        item_id=item_id,
                        person_id=person,
                        trial_idx=idx,
                        qtype="choice",
                        question=question,
                        options=choices,
                        state=text[max(0, s.start - max_state_chars) : s.start],
                        person_ref=person,
                        response=index[s.choice],
                    )
                )
    rep.n_records += len(out)
    return out
