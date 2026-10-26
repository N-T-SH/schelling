import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tests.helpers import tiny_model, tiny_tokenizer
from typer.testing import CliRunner

from schelling.cli import app
from schelling.infer.predictor import Predictor
from schelling.infer.simulate import sample_population
from schelling.schema import Choice, DecisionRequest, Noul, PersonContext, Score

pytestmark = pytest.mark.integration
runner = CliRunner()

REQUEST = DecisionRequest(
    state="You are offered a deal.",
    questions={
        "n": Noul(proposition="I accept the offer"),
        "c": Choice(prompt="Which?", options=["red", "green", "blue"]),
        "s": Score(prompt="How much?", levels=["none", "some", "a lot"]),
    },
)


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model")
    Predictor(tiny_model(seed=7), tiny_tokenizer()).save_pretrained(path)
    return path


def test_save_load_round_trip_is_identical(saved: Path) -> None:
    ref = Predictor(tiny_model(seed=7), tiny_tokenizer()).decide(REQUEST)
    got = Predictor.from_pretrained(saved).decide(REQUEST)
    for k in ref:
        assert got[k].probs == pytest.approx(ref[k].probs, abs=1e-6)


def test_weights_are_safetensors_only(saved: Path) -> None:
    names = {p.name for p in saved.iterdir()}
    assert "model.safetensors" in names
    assert not any(n.endswith((".bin", ".pt", ".pth", ".pkl")) for n in names)
    assert {"configuration_schelling.py", "modeling_schelling.py", "heads.py"} <= names
    assert json.loads((saved / "config.json").read_text())["model_type"] == "schelling"


def test_automodel_with_remote_code_works_in_a_fresh_process(saved: Path, tmp_path: Path) -> None:
    ref = Predictor(tiny_model(seed=7), tiny_tokenizer())
    batch = ref._tensors("noul", [(0, "n", REQUEST, REQUEST.questions["n"])])
    expected = ref.model("noul", **batch).tolist()
    torch.save(batch, tmp_path / "batch.pt")
    script = f"""
import json, sys, torch
from transformers import AutoModel
m = AutoModel.from_pretrained({str(saved)!r}, trust_remote_code=True).eval()
assert type(m).__module__.startswith("transformers_modules"), type(m).__module__
batch = torch.load({str(tmp_path / "batch.pt")!r})
print(json.dumps(m("noul", **batch).tolist()))
"""
    out = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, cwd=tmp_path
    )
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got[0] == pytest.approx(expected[0], abs=1e-6)


def test_sample_population_is_deterministic_and_has_one_row_per_draw() -> None:
    pred = Predictor(tiny_model(seed=2), tiny_tokenizer())
    people = [PersonContext(person_id=f"p{i}", attributes={"age": 20 + i}) for i in range(4)]
    a = sample_population(pred, people, REQUEST, n_draws=5, seed=3)
    b = sample_population(pred, people, REQUEST, n_draws=5, seed=3)
    c = sample_population(pred, people, REQUEST, n_draws=5, seed=4)
    assert a.equals(b) and not a.equals(c)
    assert len(a) == 4 * 3 * 5 and set(a.columns) >= {
        "person_id",
        "question",
        "draw",
        "response",
        "label",
    }
    assert a.groupby("question").response.max().le({"n": 1, "c": 2, "s": 2}["n"] + 1).all()


def test_simulate_cli_smoke(saved: Path, tmp_path: Path) -> None:
    (tmp_path / "req.json").write_text(REQUEST.model_dump_json())
    (tmp_path / "people.json").write_text(
        json.dumps([{"person_id": "a", "attributes": {"age": 30}}, {"person_id": "b"}])
    )
    out = tmp_path / "pop.parquet"
    res = runner.invoke(
        app,
        [
            "simulate",
            "--model",
            str(saved),
            "--persons",
            str(tmp_path / "people.json"),
            "--request",
            str(tmp_path / "req.json"),
            "--n-draws",
            "3",
            "--seed",
            "1",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    import pandas as pd

    assert len(pd.read_parquet(out)) == 2 * 3 * 3
