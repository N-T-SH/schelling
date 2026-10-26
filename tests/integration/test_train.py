"""End-to-end training on fixtures with the tiny random backbone (CPU, seconds)."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from schelling.cli import app
from schelling.eval.schema import MetricsFile
from schelling.infer.predictor import Predictor
from schelling.train.config import load_config
from schelling.train.trainer import train

pytestmark = pytest.mark.integration
runner = CliRunner()
VOCAB = Path(__file__).parents[1] / "fixtures" / "tiny_vocab.txt"


def _config(tmp: Path, proc: Path, splits: Path, name: str = "run", **over: object) -> Path:
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
            "tiny_vocab": str(VOCAB),
            "history_items": 0,
            "max_context_tokens": 96,
            "max_option_tokens": 8,
        },
        "optim": {
            "steps": 50,
            "batch_size": 8,
            "lr_backbone": 3e-3,
            "lr_head": 3e-3,
            "warmup_frac": 0.1,
        },
        "eval": {"every": 25, "max_rows": 400, "patience": 5},
    }
    cfg.update(over)  # type: ignore[arg-type]
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_fifty_step_run_decreases_loss_and_writes_complete_artifacts(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    result = train(load_config(_config(tmp_path, proc, splits), []))
    run = result.run_dir
    losses = json.loads((run / "metrics.json").read_text())["train_loss_curve"]
    assert len(losses) == 50 and sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
    meta = json.loads((run / "run.json").read_text())
    for key in (
        "config",
        "git_sha",
        "dirty",
        "split_manifest_hash",
        "package_versions",
        "seed",
        "created_at",
        "device",
    ):
        assert key in meta, key
    assert (
        meta["split_manifest_hash"]
        == json.loads((splits / "heldout_person.json").read_text())["manifest_hash"]
    )
    assert meta["package_versions"]["torch"] and meta["package_versions"]["transformers"]
    for sub in ("best", "final"):
        names = {p.name for p in (run / sub).iterdir()}
        assert "model.safetensors" in names and not any(
            n.endswith((".bin", ".pt", ".pkl")) for n in names
        )


def test_training_never_reads_the_test_partition(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    cfg = load_config(_config(tmp_path, proc, splits, name="guard"), [])
    result = train(cfg)
    meta = json.loads((result.run_dir / "run.json").read_text())
    assert meta["partitions_read"] == ["train", "val"]


def test_calibration_temperatures_are_stored_in_the_final_model_config(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    result = train(load_config(_config(tmp_path, proc, splits, name="cal"), []))
    cfg = json.loads((result.run_dir / "final" / "config.json").read_text())
    for k in ("temperature_noul", "temperature_choice", "temperature_score"):
        assert cfg[k] > 0
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert (
        metrics["calibration"]["val_nll_after"] <= metrics["calibration"]["val_nll_before"] + 1e-9
    )


def test_same_seed_on_cpu_gives_bitwise_identical_metrics(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    """A13: fixed seeds plus deterministic algorithms reproduce the run exactly."""
    proc, splits = fixture_processed
    outs = []
    for tag in "ab":
        result = train(load_config(_config(tmp_path, proc, splits, name=f"det{tag}"), []))
        m = json.loads((result.run_dir / "metrics.json").read_text())
        m.pop("created_at", None), m.pop("wall_clock_s", None)
        outs.append(m)
    assert outs[0] == outs[1]


def test_different_seeds_differ(fixture_processed: tuple[Path, Path], tmp_path: Path) -> None:
    proc, splits = fixture_processed
    a = train(load_config(_config(tmp_path, proc, splits, name="s0"), ["seed=0"]))
    b = train(load_config(_config(tmp_path, proc, splits, name="s1"), ["seed=1"]))
    ma = json.loads((a.run_dir / "metrics.json").read_text())["train_loss_curve"]
    mb = json.loads((b.run_dir / "metrics.json").read_text())["train_loss_curve"]
    assert ma != mb


def test_cli_train_then_eval_the_saved_model(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    cfg = _config(tmp_path, proc, splits, name="cli")
    res = runner.invoke(
        app, ["train", "--config", str(cfg), "--set", "optim.steps=10", "--set", "eval.every=10"]
    )
    assert res.exit_code == 0, res.output
    final = tmp_path / "cli" / "final"
    out = tmp_path / "eval"
    res = runner.invoke(
        app,
        [
            "eval",
            "--model",
            str(final),
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
    assert res.exit_code == 0, res.output
    data = json.loads((out / "metrics.json").read_text())
    MetricsFile.model_validate(data)
    assert data["individual"]["by_source"]["twin2k"]["n"] > 0
    assert Predictor.from_pretrained(final) is not None
