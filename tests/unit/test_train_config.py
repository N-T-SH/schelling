from pathlib import Path

import pytest
from pydantic import ValidationError

from schelling.train.config import TrainConfig, load_config

pytestmark = pytest.mark.unit


def test_yaml_loads_and_set_overrides_apply(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("name: x\nseed: 3\noptim:\n  steps: 10\n  batch_size: 4\n")
    cfg = load_config(f, ["optim.steps=25", "seed=9", "data.augment_option_permutation=false"])
    assert cfg.name == "x" and cfg.optim.steps == 25 and cfg.seed == 9 and cfg.optim.batch_size == 4
    assert cfg.data.augment_option_permutation is False


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("name: x\nbogus: 1\n")
    with pytest.raises(ValidationError):
        load_config(f, [])
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"name": "x", "optim": {"stepz": 3}})


def test_bad_override_syntax_is_an_error(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("name: x\n")
    with pytest.raises(ValueError):
        load_config(f, ["no-equals-sign"])


def test_two_tower_options_have_defaults_and_validate(tmp_path: Path) -> None:
    cfg = TrainConfig.model_validate(
        {"name": "x", "model": {"architecture": "two_tower", "dim": 64, "use_person": False}}
    )
    assert (
        cfg.model.architecture == "two_tower"
        and cfg.model.use_person is False
        and cfg.model.prior_weight == 1.0
    )
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"name": "x", "model": {"architecture": "mystery"}})
