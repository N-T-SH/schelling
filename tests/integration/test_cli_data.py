import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from schelling.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()


def _build(fixtures: Path, tmp: Path, tag: str) -> Path:
    out, man = tmp / f"proc_{tag}", tmp / f"man_{tag}"
    for source, raw in (
        ("twin2k", fixtures / "twin2k"),
        ("psych101", fixtures / "psych101_rows.jsonl"),
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
                str(out),
                "--manifests",
                str(man),
            ],
        )
        assert res.exit_code == 0, res.output
    return tmp


def test_build_twice_gives_identical_manifests_and_splits(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    _build(fixtures_dir, tmp_path, "a")
    _build(fixtures_dir, tmp_path, "b")
    for source in ("twin2k", "psych101"):
        a = json.loads((tmp_path / "man_a" / f"{source}.json").read_text())
        b = json.loads((tmp_path / "man_b" / f"{source}.json").read_text())
        assert a == b
        assert a["files"] and a["license"] and a["stats"]["records"] > 0
    for tag in "ab":
        res = runner.invoke(
            app,
            [
                "data",
                "splits",
                "--processed",
                str(tmp_path / f"proc_{tag}"),
                "--out",
                str(tmp_path / f"splits_{tag}"),
            ],
        )
        assert res.exit_code == 0, res.output
    for p in sorted((tmp_path / "splits_a").glob("*.json")):
        assert p.read_text() == (tmp_path / "splits_b" / p.name).read_text()
    assert (
        len(list((tmp_path / "splits_a").glob("*.json"))) == 15
    )  # fourteen splits + contamination
