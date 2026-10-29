import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tests.helpers import tiny_tokenizer, tiny_tt_model

from schelling.infer.predictor import Predictor
from schelling.schema import Choice, DecisionRequest, Noul, PersonContext, Score

pytestmark = pytest.mark.integration
REQ = DecisionRequest(
    person=PersonContext(attributes={"age": "30", "sex": "f"}),
    state="You are offered a deal.",
    questions={
        "n": Noul(proposition="I accept"),
        "c": Choice(prompt="Which?", options=["red", "green", "blue"]),
        "s": Score(prompt="How much?", levels=["none", "some", "a lot"]),
    },
)


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tt")
    Predictor(tiny_tt_model(seed=7, residual_std=0.5), tiny_tokenizer()).save_pretrained(path)
    return path


def test_round_trip_is_identical_and_dispatches_to_the_two_tower_class(saved: Path) -> None:
    ref = Predictor(tiny_tt_model(seed=7, residual_std=0.5), tiny_tokenizer()).decide(REQ)
    loaded = Predictor.from_pretrained(saved)
    assert type(loaded.model).__name__ == "SchellingTwoTower"
    for k, v in loaded.decide(REQ).items():
        assert v.probs == pytest.approx(ref[k].probs, abs=1e-6)


def test_weights_are_safetensors_and_custom_code_files_are_copied(saved: Path) -> None:
    names = {p.name for p in saved.iterdir()}
    assert "model.safetensors" in names and not any(
        n.endswith((".bin", ".pt", ".pkl")) for n in names
    )
    assert {
        "configuration_schelling_tt.py",
        "modeling_schelling_tt.py",
        "heads.py",
        "heads_tt.py",
    } <= names
    assert json.loads((saved / "config.json").read_text())["model_type"] == "schelling_tt"


def test_automodel_with_remote_code_works_in_a_fresh_process(saved: Path, tmp_path: Path) -> None:
    ref = Predictor(tiny_tt_model(seed=7, residual_std=0.5), tiny_tokenizer())
    items = [(0, "c", REQ, REQ.questions["c"])]
    batch = ref._tensors("choice", items)
    expected = ref.model("choice", **batch).tolist()
    torch.save(batch, tmp_path / "batch.pt")
    script = f"""
import json, torch
from transformers import AutoModel
m = AutoModel.from_pretrained({str(saved)!r}, trust_remote_code=True).eval()
assert type(m).__module__.startswith("transformers_modules"), type(m).__module__
batch = torch.load({str(tmp_path / "batch.pt")!r})
print(json.dumps(m("choice", **batch).tolist()))
"""
    out = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, cwd=tmp_path
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1])[0] == pytest.approx(
        expected[0], abs=1e-6
    )
