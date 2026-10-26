"""Twin-2K-500 adapter, built from the question catalog and the response CSVs.

Never read `persona_text`, `persona_json` or `persona_summary`: they contain the person's own
answers, and wave-4 answers replace wave 1-3 ones for repeated items (assumption A04).
`wave` is 1 for the waves 1-3 file and 4 for the wave-4 file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from schelling.data.schema import DecisionRecord, make_record_id
from schelling.schema import HistoryItem, PersonContext

SOURCE = "twin2k"
DEMOGRAPHICS = {
    "QID11": "region",
    "QID12": "sex",
    "QID13": "age_band",
    "QID14": "education",
    "QID15": "race",
    "QID16": "citizen",
    "QID17": "marital_status",
    "QID18": "religion",
    "QID19": "religious_attendance",
    "QID20": "party_id",
    "QID21": "income_band",
    "QID22": "political_ideology",
    "QID23": "household_size",
    "QID24": "employment",
}
# Blocks that are not a named between-subject experiment: `condition` stays None for these.
_GENERIC_BLOCKS = {
    "Cognitive tests",
    "Personality",
    "Product Preferences - Pricing",
    "Economic preferences",
    "Economic preferences - intro",
    "Non-experimental heuristics and biases",
    "Forward Flow",
    "Demographics",
}


@dataclass
class TwinReport:
    unmapped_columns: dict[str, str] = field(default_factory=dict)
    dropped_values: int = 0


@dataclass
class TwinBuild:
    records: list[DecisionRecord]
    persons: pd.DataFrame
    report: TwinReport


@dataclass(frozen=True)
class _Item:
    column: str
    study: str
    condition: str | None
    qtype: str
    question: str
    options: list[str] | None
    levels: list[str] | None
    yes_index: int | None = None


def _items(catalog: list[dict[str, Any]], report: TwinReport) -> dict[str, _Item]:
    items: dict[str, _Item] = {}
    for q in catalog:
        qtype, sel = q["QuestionType"], q.get("Settings", {}).get("Selector")
        block = q.get("BlockName", "").strip()
        cond = None if block in _GENERIC_BLOCKS else block
        cols = q.get("csv_columns", [])
        text = q.get("QuestionText", "").strip()
        if qtype == "DB":
            continue
        if qtype == "MC" and sel in ("SAVR", "SAHR"):
            opts = list(q["Options"])
            low = [o.strip().lower() for o in opts]
            if sorted(low) == ["no", "yes"]:
                it = _Item(cols[0], block, cond, "noul", text, None, None, low.index("yes"))
            else:
                it = _Item(cols[0], block, cond, "choice", text, opts, None)
            items[cols[0]] = it
        elif qtype == "Matrix" and sel == "Likert":
            for col, row in zip(cols, q["Rows"], strict=True):
                items[col] = _Item(
                    col, block, cond, "score", f"{text} {row}".strip(), None, list(q["Columns"])
                )
        elif qtype == "Matrix" and sel == "Bipolar":
            for col, row in zip(cols, q["Rows"], strict=True):
                left, _, right = row.partition(":")
                items[col] = _Item(col, block, cond, "choice", text, [left, right], None)
        else:
            reason = {
                "MC": "multi-select (binary column per option)",
                "Slider": "continuous slider",
                "TE": "free text",
            }.get(qtype, f"unsupported type {qtype}")
            for col in cols:
                report.unmapped_columns[col] = reason
    return items


def _record(
    item: _Item, pid: str, value: float, wave: int, report: TwinReport
) -> DecisionRecord | None:
    n = {"noul": 2, "choice": len(item.options or []), "score": len(item.levels or [])}[item.qtype]
    pos = int(value) - 1  # numeric CSVs hold 1-based positions
    if not 0 <= pos < n:
        report.dropped_values += 1
        return None
    if item.qtype == "noul":
        pos = 1 if pos == item.yes_index else 0  # noul index 1 means "yes"
    return DecisionRecord(
        record_id=make_record_id(SOURCE, item.study, pid, item.column + f"@w{wave}", None),
        source=SOURCE,
        study_id=item.study,
        item_id=item.column,
        person_id=pid,
        qtype=item.qtype,
        question=item.question,
        options=item.options,
        levels=item.levels,
        person_ref=pid,
        response=pos,
        condition=item.condition,
        wave=wave,
    )


def build(raw_dir: Path) -> TwinBuild:
    catalog = json.loads((raw_dir / "question_catalog.json").read_text())
    report = TwinReport()
    items = _items(catalog, report)
    w13 = pd.read_csv(raw_dir / "wave1_3_response.csv")
    w4 = pd.read_csv(raw_dir / "wave4_response.csv")
    labels = pd.read_csv(raw_dir / "wave1_3_response_label.csv")

    persons = labels[["pid", *[c for c in DEMOGRAPHICS if c in labels.columns]]].rename(
        columns={"pid": "person_id", **DEMOGRAPHICS}
    )
    persons["person_id"] = persons["person_id"].astype(str)

    records: list[DecisionRecord] = []
    for wave, frame in ((1, w13), (4, w4)):
        for col in frame.columns[1:]:
            if col in DEMOGRAPHICS or col not in items:
                continue
            for pid, value in zip(frame["pid"].astype(str), frame[col], strict=True):
                if pd.isna(value):
                    continue
                rec = _record(items[col], pid, float(value), wave, report)
                if rec is not None:
                    records.append(rec)
    return TwinBuild(records, persons, report)


def build_context(
    answers: dict[str, tuple[str, list[str], int]],
    target_item: str,
    attributes: dict[str, str | float | int],
    k: int = 50,
) -> PersonContext:
    """Person context for predicting `target_item`.

    `answers` maps item id to (question, options, chosen index) from waves 1-3 only. The target
    item is removed entirely, so its question and answer appear nowhere in the context.
    """
    history = [
        HistoryItem(question=q, options=o, answer=a)
        for item, (q, o, a) in answers.items()
        if item != target_item
    ]
    return PersonContext(attributes=dict(attributes), history=history[-k:])
