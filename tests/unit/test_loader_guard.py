"""AGENTS.md rule 3: the sealed test partition is refused unless the human sets the env var."""

import pandas as pd
import pytest

from schelling.data.loader import SealedPartitionError, select_partition

pytestmark = pytest.mark.unit


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame({"record_id": list("abc"), "partition": ["train", "val", "test"]})


def test_train_and_val_are_available(frame: pd.DataFrame) -> None:
    assert list(select_partition(frame, "train").record_id) == ["a"]
    assert list(select_partition(frame, "val").record_id) == ["b"]


def test_test_rows_refused_without_env(frame: pd.DataFrame) -> None:
    with pytest.raises(SealedPartitionError):
        select_partition(frame, "test")


def test_test_rows_refused_for_any_other_env_value(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCHELLING_ALLOW_TEST", "true")
    with pytest.raises(SealedPartitionError):
        select_partition(frame, "test")


def test_test_rows_allowed_only_with_exact_flag(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCHELLING_ALLOW_TEST", "1")
    assert list(select_partition(frame, "test").record_id) == ["c"]


def test_unknown_partition_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        select_partition(frame, "dev")
