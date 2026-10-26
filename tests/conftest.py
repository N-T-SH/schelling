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


@pytest.fixture(scope="session")
def twin_frame(fixtures_dir: Path):  # type: ignore[no-untyped-def]
    """(records DataFrame, persons DataFrame) built from the Twin fixture."""
    import pandas as pd

    from schelling.data.adapters import twin2k

    built = twin2k.build(fixtures_dir / "twin2k")
    return pd.DataFrame([r.model_dump() for r in built.records]), built.persons
