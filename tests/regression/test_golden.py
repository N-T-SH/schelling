"""Golden metrics on the frozen fixture benchmark (SPEC "Regression").

Fails if NLL worsens by more than one per cent or ECE by more than 0.01. Updating the golden
file needs `[golden-update]` in the commit message (scripts/make_golden.py).
"""

import json
from pathlib import Path

import pytest

from schelling.train.config import load_config
from schelling.train.golden import golden_config, run_golden

pytestmark = pytest.mark.slow
GOLDEN = Path(__file__).parents[1] / "golden" / "metrics.json"


def test_fixture_benchmark_has_not_regressed(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    got = run_golden(load_config(golden_config(tmp_path, proc, splits), []))
    gold = json.loads(GOLDEN.read_text())
    assert got["val_nll"] <= gold["val_nll"] * 1.01, (got, gold)
    assert got["val_ece"] <= gold["val_ece"] + 0.01, (got, gold)
