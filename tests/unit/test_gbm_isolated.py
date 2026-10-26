"""Runs tests/gbm in a fresh interpreter (see tests/gbm/test_gbm.py for why)."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_gbm_suite_passes_in_a_clean_process() -> None:
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/gbm", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-1000:]
