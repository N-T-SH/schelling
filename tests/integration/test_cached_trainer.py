import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from schelling.cli import app
from schelling.eval.schema import MetricsFile
from schelling.infer.predictor import Predictor
from schelling.schema import Choice, DecisionRequest, PersonContext
from schelling.train.config import load_config
from schelling.train.trainer import train

pytestmark = pytest.mark.integration
runner = CliRunner()
VOCAB = Path(__file__).parents[1] / "fixtures" / "tiny_vocab.txt"


def _cfg(tmp: Path, proc: Path, splits: Path, name: str = "tt", **model: object) -> Path:
    cfg = {
        "name": name,
        "seed": 0,
        "split": "heldout_person",
        "processed_dir": str(proc),
        "splits_dir": str(splits),
        "out_dir": str(tmp / name),
        "device": "cpu",
        "deterministic": True,
        "model": {
            "architecture": "two_tower",
            "tiny_vocab": str(VOCAB),
            "dim": 16,
            "person_heads": 2,
            "max_question_tokens": 96,
            "max_option_tokens": 8,
            "max_attr_tokens": 8,
            **model,
        },
        "data": {"sources": ["twin2k"]},
        "optim": {
            "steps": 80,
            "batch_size": 64,
            "lr_head": 3e-3,
            "lr_backbone": 0.0,
            "warmup_frac": 0.1,
        },
        "eval": {"every": 40, "max_rows": 400, "patience": 5, "calibration_rows": 400},
    }
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_run_decreases_loss_writes_artifacts_and_loads(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    res = train(load_config(_cfg(tmp_path, proc, splits), []))
    metrics = json.loads((res.run_dir / "metrics.json").read_text())
    curve = metrics["train_loss_curve"]
    assert len(curve) == 80 and sum(curve[-10:]) / 10 < sum(curve[:10]) / 10
    meta = json.loads((res.run_dir / "run.json").read_text())
    for key in (
        "config",
        "git_sha",
        "dirty",
        "split_manifest_hash",
        "package_versions",
        "seed",
        "created_at",
    ):
        assert key in meta
    assert meta["partitions_read"] == ["train", "val"] and meta["config"]["data"]["sources"] == [
        "twin2k"
    ]
    pred = Predictor.from_pretrained(res.run_dir / "final")
    assert type(pred.model).__name__ == "SchellingTwoTower"
    ans = pred.decide(
        DecisionRequest(
            person=PersonContext(attributes={"sex": "Male"}),
            state="",
            questions={"q": Choice(prompt="pick", options=["a", "b"])},
        )
    )["q"]
    assert sum(ans.probs) == pytest.approx(1.0, abs=1e-5)
    for k in ("temperature_noul", "temperature_choice", "temperature_score"):
        assert json.loads((res.run_dir / "final" / "config.json").read_text())[k] > 0


def test_same_seed_gives_identical_metrics(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    outs = []
    for tag in "ab":
        res = train(load_config(_cfg(tmp_path, proc, splits, name=f"d{tag}"), []))
        m = json.loads((res.run_dir / "metrics.json").read_text())
        m.pop("created_at", None), m.pop("wall_clock_s", None)
        outs.append(m)
    assert outs[0] == outs[1]


def test_ablation_flags_reach_the_saved_model(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    res = train(load_config(_cfg(tmp_path, proc, splits, name="abl", use_person=False), []))
    cfg = json.loads((res.run_dir / "final" / "config.json").read_text())
    assert cfg["use_person"] is False


def test_cli_train_then_eval_through_the_unchanged_harness(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    cfg = _cfg(tmp_path, proc, splits, name="cli")
    r = runner.invoke(
        app, ["train", "--config", str(cfg), "--set", "optim.steps=20", "--set", "eval.every=20"]
    )
    assert r.exit_code == 0, r.output
    out = tmp_path / "eval"
    r = runner.invoke(
        app,
        [
            "eval",
            "--model",
            str(tmp_path / "cli" / "final"),
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
            "2",
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads((out / "metrics.json").read_text())
    MetricsFile.model_validate(data)
    assert data["individual"]["by_source"]["twin2k"]["n"] > 0
