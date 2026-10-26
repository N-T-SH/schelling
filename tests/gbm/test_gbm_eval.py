import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from schelling.cli import app
from schelling.eval.schema import MetricsFile

pytestmark = pytest.mark.integration
runner = CliRunner()
FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_eval_b3_writes_valid_metrics_and_reports_position_dependence(tmp_path: Path) -> None:
    proc, man, splits = tmp_path / "proc", tmp_path / "man", tmp_path / "splits"
    for source, raw in (
        ("twin2k", FIXTURES / "twin2k"),
        ("psych101", FIXTURES / "psych101_rows.jsonl"),
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
                str(proc),
                "--manifests",
                str(man),
            ],
        )
        assert res.exit_code == 0, res.output
    assert (
        runner.invoke(
            app, ["data", "splits", "--processed", str(proc), "--out", str(splits)]
        ).exit_code
        == 0
    )
    res = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "b3",
            "--split",
            "heldout_person",
            "--partition",
            "val",
            "--processed",
            str(proc),
            "--splits-dir",
            str(splits),
            "--out",
            str(tmp_path / "run"),
            "--min-persons",
            "2",
            "--n-seeds",
            "3",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads((tmp_path / "run" / "metrics.json").read_text())
    MetricsFile.model_validate(data)
    assert set(data["individual"]["by_source"]) == {"psych101", "twin2k"}
