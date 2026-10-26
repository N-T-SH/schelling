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


@pytest.fixture(scope="session")
def fixture_processed(
    tmp_path_factory: pytest.TempPathFactory, fixtures_dir: Path
) -> tuple[Path, Path]:
    """(processed dir, splits dir) built from the fixtures through the real CLI."""
    from typer.testing import CliRunner

    from schelling.cli import app

    runner, tmp = CliRunner(), tmp_path_factory.mktemp("fixture_processed")
    for source, raw in (
        ("twin2k", fixtures_dir / "twin2k"),
        ("psych101", fixtures_dir / "psych101_rows.jsonl"),
    ):
        res = runner.invoke(
            app,
            [
                "data",
                "build",
                "--source",
                source,
                "--raw",
                str(raw),
                "--out",
                str(tmp / "proc"),
                "--manifests",
                str(tmp / "man"),
            ],
        )
        assert res.exit_code == 0, res.output
    res = runner.invoke(
        app, ["data", "splits", "--processed", str(tmp / "proc"), "--out", str(tmp / "splits")]
    )
    assert res.exit_code == 0, res.output
    return tmp / "proc", tmp / "splits"
