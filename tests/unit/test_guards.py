"""Guard metrics (decision 0005): recorded from existing metrics.json files, TVD not binding."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from schelling.cli import app
from schelling.eval.guards import compute_guards

pytestmark = pytest.mark.unit


def _metrics(
    nll: float = 1.0,
    tvd: float = 0.05,
    ece: float = 0.01,
    sd: float = 1.0,
    perm: float = 0.0,
    split: str = "heldout_person",
) -> dict[str, Any]:
    return {
        "git_sha": "abc",
        "split": split,
        "partition": "val",
        "split_manifest_hash": "h",
        "individual": {"by_source": {"twin2k": {"nll": nll, "ece": ece}}},
        "population": {"twin2k": {"summary": {"tvd": {"mean": tvd}, "sd_ratio": {"mean": sd}}}},
        "robustness": {"permutation_max_abs_dp": perm},
    }


def test_a_model_that_beats_the_reference_passes_every_guard() -> None:
    g = compute_guards(_metrics(nll=0.99, tvd=0.05), _metrics())
    assert all(v["pass"] for v in g["guards"].values())
    assert g["binding_pass"] is True


def test_worse_nll_than_the_reference_fails_a_binding_guard() -> None:
    g = compute_guards(_metrics(nll=1.01), _metrics(nll=1.0))
    assert g["guards"]["person_nll"]["pass"] is False
    assert g["guards"]["person_nll"]["binding"] is True
    assert g["binding_pass"] is False


def test_worse_tvd_is_recorded_but_does_not_fail_the_binding_result() -> None:
    g = compute_guards(_metrics(tvd=0.06), _metrics(tvd=0.05))
    tvd = g["guards"]["population_tvd"]
    assert tvd["pass"] is False and tvd["binding"] is False
    assert g["binding_pass"] is True


@pytest.mark.parametrize(
    ("kw", "name"),
    [
        ({"ece": 0.06}, "ece"),
        ({"sd": 0.8}, "sd_ratio"),
        ({"sd": 1.2}, "sd_ratio"),
        ({"perm": 0.02}, "permutation"),
    ],
)
def test_existing_limits_are_binding(kw: dict[str, float], name: str) -> None:
    g = compute_guards(_metrics(**kw), _metrics())
    assert g["guards"][name]["pass"] is False and g["binding_pass"] is False


def test_limits_are_inclusive() -> None:
    g = compute_guards(_metrics(ece=0.05, sd=1.15, perm=0.01), _metrics())
    assert g["guards"]["ece"]["pass"] and g["guards"]["sd_ratio"]["pass"]


def test_a_reference_on_a_different_split_is_refused() -> None:
    with pytest.raises(ValueError, match="same split"):
        compute_guards(_metrics(), _metrics(split="heldout_item"))


def test_cli_writes_guards_json_with_provenance(tmp_path: Path) -> None:
    for name, m in (("run", _metrics(nll=0.99)), ("ref", _metrics())):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps(m))
    out = tmp_path / "guards.json"
    res = CliRunner().invoke(
        app,
        [
            "guards",
            "--run",
            str(tmp_path / "run"),
            "--reference",
            str(tmp_path / "ref"),
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(out.read_text())
    assert data["run"]["git_sha"] == "abc" and "created_at" in data and "git_sha" in data
    assert data["binding_pass"] is True
