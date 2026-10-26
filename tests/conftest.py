import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _sealed_test_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests must never depend on the human-only switch (AGENTS.md rule 3).
    monkeypatch.delenv("SCHELLING_ALLOW_TEST", raising=False)
    assert "SCHELLING_ALLOW_TEST" not in os.environ


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES
