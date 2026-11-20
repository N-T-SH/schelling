import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_cached_trainer import _cfg

from schelling.augment.paraphrases import ParaphraseStore
from schelling.data.loader import load_records, partition_frame, select_partition
from schelling.infer.baseline_adapter import person_attributes
from schelling.train.cached_trainer import _build_model, build_cache
from schelling.train.config import load_config
from schelling.train.trainer import train

pytestmark = pytest.mark.integration


_MODEL: dict[str, tuple[object, object]] = {}


def _store(proc: Path, path: Path, only: set[str] | None = None) -> Path:
    frame = pd.read_parquet(proc / "records_twin2k.parquet", columns=["question"])
    store = ParaphraseStore(path, generator="test@v1")
    for text in sorted(frame["question"].unique()):
        if only is None or text in only:
            store.add(text, [f"Please consider this. {text}", f"Question: {text} Think carefully."])
    return path


def _cfg_aug(tmp: Path, proc: Path, splits: Path, store: Path, name: str, n: int) -> Path:
    path = _cfg(tmp, proc, splits, name=name)
    import yaml

    raw = yaml.safe_load(path.read_text())
    raw["data"].update(
        {"paraphrase_path": str(store), "paraphrases_per_question": n, "paraphrase_prob": 0.7}
    )
    raw["data"]["paraphrase_generator"] = "test@v1"
    path.write_text(yaml.safe_dump(raw))
    return path


def _cache(proc: Path, splits: Path, tmp: Path, store: ParaphraseStore | None, n: int):  # type: ignore[no-untyped-def]
    # one fixed model so vectors of unchanged texts are comparable between calls
    cfg = load_config(_cfg(tmp, proc, splits, name="c"), [])
    frame = load_records(proc)
    frame = frame[frame["source"] == "twin2k"].reset_index(drop=True)
    parted = partition_frame(frame, "heldout_person", splits)
    train_df = select_partition(parted, "train").reset_index(drop=True)
    model, tok = _MODEL.setdefault("m", _build_model(cfg))
    attrs = person_attributes(pd.read_parquet(proc / "persons_twin2k.parquet"))
    return build_cache(model, tok, train_df, attrs, torch.device("cpu"), store, n), train_df


def test_cache_adds_alternative_wordings_and_leaves_labels_alone(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    base, _ = _cache(proc, splits, tmp_path, None, 0)
    aug, _ = _cache(
        proc, splits, tmp_path, ParaphraseStore(_store(proc, tmp_path / "p.jsonl"), "test@v1"), 2
    )
    assert base.q_alt is None and aug.q_alt is not None
    assert aug.U.shape[0] > base.U.shape[0]
    assert torch.equal(aug.q_idx, base.q_idx) and torch.equal(aug.y, base.y)
    assert torch.equal(aug.opt_idx, base.opt_idx) and torch.equal(aug.E, base.E)
    # every base question has alternatives; an alternative is never the original vector
    alt = aug.q_alt[torch.unique(aug.q_idx)]
    assert bool((alt >= 0).all())
    assert not bool(torch.isin(alt, torch.unique(aug.q_idx)).any())


def test_questions_without_stored_paraphrases_keep_their_wording(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    texts = sorted(pd.read_parquet(proc / "records_twin2k.parquet")["question"].unique())
    store = ParaphraseStore(_store(proc, tmp_path / "p.jsonl", {texts[0]}), "test@v1")
    aug, _ = _cache(proc, splits, tmp_path, store, 2)
    assert aug.q_alt is not None
    has_alt = (aug.q_alt >= 0).any(1)
    assert 0 < int(has_alt.sum()) < int(torch.unique(aug.q_idx).numel())


def test_default_config_is_the_old_behaviour(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    store = _store(proc, tmp_path / "p.jsonl")
    off = train(load_config(_cfg_aug(tmp_path, proc, splits, store, "off", 0), []))
    ref = train(load_config(_cfg(tmp_path, proc, splits, name="ref"), []))
    a, b = (json.loads((r.run_dir / "metrics.json").read_text()) for r in (off, ref))
    assert a["train_loss_curve"] == b["train_loss_curve"]


def test_augmented_run_trains_and_changes_the_training_signal(
    fixture_processed: tuple[Path, Path], tmp_path: Path
) -> None:
    proc, splits = fixture_processed
    store = _store(proc, tmp_path / "p.jsonl")
    on = train(load_config(_cfg_aug(tmp_path, proc, splits, store, "on", 2), []))
    ref = train(load_config(_cfg(tmp_path, proc, splits, name="ref2"), []))
    a = json.loads((on.run_dir / "metrics.json").read_text())["train_loss_curve"]
    b = json.loads((ref.run_dir / "metrics.json").read_text())["train_loss_curve"]
    assert a != b and sum(a[-10:]) < sum(a[:10])
    meta = json.loads((on.run_dir / "run.json").read_text())
    assert meta["config"]["data"]["paraphrases_per_question"] == 2
