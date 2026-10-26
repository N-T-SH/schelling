import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from schelling.cli import app
from schelling.eval.schema import MetricsFile

pytestmark = pytest.mark.integration
runner = CliRunner()
PROVENANCE = ["git_sha", "dirty", "config_hash", "split_manifest_hash", "seed", "created_at"]


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory, fixtures_dir: Path) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("proc")
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


@pytest.mark.parametrize("model", ["b0", "b1", "b2", "b3"])
def test_eval_writes_valid_metrics_for_each_baseline(
    processed: tuple[Path, Path], tmp_path: Path, model: str
) -> None:
    proc, splits = processed
    out = tmp_path / model
    res = runner.invoke(
        app,
        [
            "eval",
            "--model",
            model,
            "--split",
            "heldout_person",
            "--partition",
            "val",
            "--processed",
            str(proc),
            "--splits-dir",
            str(splits),
            "--out",
            str(out),
            "--min-persons",
            "2",
            "--n-seeds",
            "3",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads((out / "metrics.json").read_text())
    MetricsFile.model_validate(data)
    assert all(k in data for k in PROVENANCE)
    assert (
        data["split_manifest_hash"]
        == json.loads((splits / "heldout_person.json").read_text())["manifest_hash"]
    )
    assert (out / "summary.md").exists() and (out / "predictions.parquet").exists()
    assert set(data["individual"]["by_source"]) == {"psych101", "twin2k"}


def test_eval_refuses_test_partition_without_env(
    processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = processed
    res = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "b0",
            "--split",
            "heldout_person",
            "--partition",
            "test",
            "--processed",
            str(proc),
            "--splits-dir",
            str(splits),
            "--out",
            str(tmp_path),
        ],
    )
    assert res.exit_code != 0 and not (tmp_path / "metrics.json").exists()


def test_same_seed_gives_identical_metrics(processed: tuple[Path, Path], tmp_path: Path) -> None:
    proc, splits = processed
    runs = []
    for tag in "ab":
        out = tmp_path / tag
        res = runner.invoke(
            app,
            [
                "eval",
                "--model",
                "b1",
                "--split",
                "heldout_person",
                "--partition",
                "val",
                "--processed",
                str(proc),
                "--splits-dir",
                str(splits),
                "--out",
                str(out),
                "--min-persons",
                "2",
                "--n-seeds",
                "3",
                "--seed",
                "5",
            ],
        )
        assert res.exit_code == 0, res.output
        d = json.loads((out / "metrics.json").read_text())
        d.pop("created_at"), d.pop("efficiency")  # wall-clock fields differ by construction
        runs.append(d)
    assert runs[0] == runs[1]


def test_retest_split_reports_fraction_of_ceiling(
    processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = processed
    res = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "b1",
            "--split",
            "retest",
            "--partition",
            "val",
            "--processed",
            str(proc),
            "--splits-dir",
            str(splits),
            "--out",
            str(tmp_path),
            "--min-persons",
            "2",
            "--n-seeds",
            "2",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads((tmp_path / "metrics.json").read_text())
    assert data["retest"]["n"] > 0 and 0 <= data["retest"]["ceiling"] <= 1
