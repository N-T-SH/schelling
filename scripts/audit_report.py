"""Check that every metric in a Markdown report traces to a metrics.json value.

A tagged number looks like `0.1234 <!-- src: path/to/metrics.json#individual.overall.nll -->`.
It passes when the number equals the value at that JSON path to the precision it is written
with. Any other decimal or percentage in the prose (outside code, URLs and links) is an
untagged metric-like number and fails the audit. Integers are not checked.

Usage: python scripts/audit_report.py REPORT.md   (exit 0 = pass, 1 = fail)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

TAG = re.compile(r"(-?\d+(?:\.\d+)?)(%?)\s*<!--\s*src:\s*([^#\s]+)#(\S+?)\s*-->")
FENCE = re.compile(r"^```.*?^```", re.S | re.M)
COMMENT = re.compile(r"<!--.*?-->", re.S)
INLINE_CODE = re.compile(r"`[^`\n]*`")
URL = re.compile(r"https?://\S+|\]\([^)]*\)")
METRIC_LIKE = re.compile(r"(?<![\w.])-?\d+\.\d+(?!\.\d)%?|(?<![\w.])\d+%")


def lookup(data: Any, path: str) -> Any:
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def audit(report: Path) -> list[str]:
    text = report.read_text()
    problems: list[str] = []
    for m in TAG.finditer(text):
        shown, pct, file_, path = m.groups()
        line = text.count("\n", 0, m.start()) + 1
        candidates = [Path(file_), report.parent / file_]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            problems.append(f"line {line}: {file_} not found")
            continue
        try:
            actual = float(lookup(json.loads(src.read_text()), path))
        except (KeyError, IndexError, ValueError, TypeError):
            problems.append(f"line {line}: path {path} not found or not numeric in {file_}")
            continue
        value = float(shown) / 100 if pct else float(shown)
        tol = 0.5 * 10 ** -(decimals(shown) + (2 if pct else 0)) + 1e-12
        if abs(value - actual) > tol:
            problems.append(f"line {line}: {shown}{pct} != {actual!r} at {path}")
    bare = TAG.sub("", text)
    for pattern in (FENCE, COMMENT, INLINE_CODE, URL):
        bare = pattern.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), bare)
    for m in METRIC_LIKE.finditer(bare):
        line = bare.count("\n", 0, m.start()) + 1
        problems.append(f"line {line}: untagged metric-like number {m.group()}")
    return problems


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    problems = audit(Path(sys.argv[1]))
    for p in problems:
        print(p)
    print("audit passed" if not problems else f"audit FAILED: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
