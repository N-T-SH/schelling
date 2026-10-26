"""Split tests (SPEC "Data adapters and splits")."""

import pandas as pd
import pytest

from schelling.data.splits import SPLIT_NAMES, assign_partitions, split_manifest

pytestmark = pytest.mark.unit


def _frame() -> pd.DataFrame:
    rows = []
    for source, n_study, n_person, n_item in [("psych101", 40, 60, 5), ("twin2k", 10, 400, 25)]:
        for s in range(n_study):
            for p in range(n_person):
                for i in range(n_item):
                    rows.append(
                        {
                            "record_id": f"{source}-{s}-{p}-{i}",
                            "source": source,
                            "study_id": f"{source}-study{s}",
                            "person_id": f"{source}-p{p}" if source == "twin2k" else f"{s}-{p}",
                            "item_id": f"{source}-{s}-item{i}",
                            "trial_idx": i,
                            "wave": 4 if (source == "twin2k" and i % 5 == 0) else 1,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


def test_all_five_splits_plus_retest_exist() -> None:
    assert set(SPLIT_NAMES) == {
        "heldout_person",
        "heldout_item",
        "heldout_study",
        "heldout_domain",
        "temporal",
        "retest",
        "heldout_experiment",
    }


def test_heldout_person_is_person_disjoint(frame: pd.DataFrame) -> None:
    part = assign_partitions(frame, "heldout_person", seed=0)
    f = frame.assign(partition=part)
    train = set(f[f.partition == "train"].person_id)
    held = set(f[f.partition != "train"].person_id)
    assert train.isdisjoint(held) and held


def test_heldout_item_is_item_disjoint(frame: pd.DataFrame) -> None:
    f = frame.assign(partition=assign_partitions(frame, "heldout_item", seed=0))
    assert set(f[f.partition == "train"].item_id).isdisjoint(set(f[f.partition != "train"].item_id))


def test_heldout_study_is_study_disjoint(frame: pd.DataFrame) -> None:
    f = frame.assign(partition=assign_partitions(frame, "heldout_study", seed=0))
    assert set(f[f.partition == "train"].study_id).isdisjoint(
        set(f[f.partition != "train"].study_id)
    )


def test_heldout_domain_trains_on_one_source_only(frame: pd.DataFrame) -> None:
    f = frame.assign(partition=assign_partitions(frame, "heldout_domain", seed=0))
    assert set(f[f.partition == "train"].source) == {"psych101"}
    assert set(f[f.partition != "train"].source) == {"twin2k"}


def test_temporal_predicts_later_trials_from_earlier(frame: pd.DataFrame) -> None:
    f = frame.assign(partition=assign_partitions(frame, "temporal", seed=0))
    g = f.groupby(["source", "study_id", "person_id"])
    for _, grp in list(g)[:200]:
        tr = grp[grp.partition == "train"].trial_idx
        te = grp[grp.partition != "train"].trial_idx
        if len(tr) and len(te):
            assert tr.max() < te.min()


def test_retest_uses_only_wave4_and_no_train(frame: pd.DataFrame) -> None:
    part = assign_partitions(frame, "retest", seed=0)
    f = frame.assign(partition=part)
    kept = f[f.partition.notna() & (f.partition != "")]
    assert set(kept.wave) == {4}
    assert "train" not in set(kept.partition)


@pytest.mark.parametrize("name", ["heldout_person", "heldout_item", "heldout_study"])
def test_deterministic_for_same_seed_and_differs_across_seeds(
    frame: pd.DataFrame, name: str
) -> None:
    a = assign_partitions(frame, name, seed=0)
    b = assign_partitions(frame, name, seed=0)
    c = assign_partitions(frame, name, seed=1)
    assert a.equals(b)
    assert not a.equals(c)


def test_person_split_sizes_within_one_point_per_source(frame: pd.DataFrame) -> None:
    f = frame.assign(partition=assign_partitions(frame, "heldout_person", seed=0))
    persons = f[f.source == "twin2k"].drop_duplicates("person_id")
    share = persons.partition.value_counts(normalize=True)
    # 400 persons: binomial noise is larger than 1pp, so sizes come from hash ranks, not coin flips
    assert abs(share["val"] - 0.10) <= 0.01
    assert abs(share["test"] - 0.10) <= 0.01


def test_manifest_hash_is_stable_and_changes_with_assignment(frame: pd.DataFrame) -> None:
    a = split_manifest(frame, "heldout_person", seed=0)
    b = split_manifest(frame, "heldout_person", seed=0)
    c = split_manifest(frame, "heldout_person", seed=1)
    assert a == b and a["manifest_hash"] != c["manifest_hash"]
    assert set(a["counts"]) >= {"train", "val", "test"}


def test_excluded_items_leave_val_and_test_but_not_train(frame: pd.DataFrame) -> None:
    base = assign_partitions(frame, "heldout_item", seed=0)
    held_items = sorted(set(frame[base.isin(["val", "test"])].item_id))[:5]
    train_items = sorted(set(frame[base == "train"].item_id))[:5]
    part = assign_partitions(
        frame, "heldout_item", seed=0, exclude_item_ids=[*held_items, *train_items]
    )
    f = frame.assign(partition=part)
    assert f[f.item_id.isin(held_items)].partition.isna().all()
    assert (f[f.item_id.isin(train_items)].partition == "train").all()
    assert set(split_manifest(frame, "heldout_item", 0, held_items)["removed_item_ids"]) == set(
        held_items
    )


def _experiment_frame() -> pd.DataFrame:
    from schelling.eval.effects import load_pairs

    items = sorted({a for p in load_pairs() for a in (p["arm_a"], p["arm_b"])})
    rows = [
        {
            "record_id": f"{i}-{k}",
            "source": "twin2k",
            "study_id": "s",
            "item_id": i,
            "person_id": f"p{k}",
            "trial_idx": None,
            "wave": 1,
        }
        for i in items
        for k in range(5)
    ]
    rows += [
        {
            "record_id": f"o-{k}",
            "source": "twin2k",
            "study_id": "s",
            "item_id": "QID999",
            "person_id": f"p{k}",
            "trial_idx": None,
            "wave": 1,
        }
        for k in range(5)
    ]
    return pd.DataFrame(rows)


def test_heldout_experiment_keeps_both_arms_together_and_other_items_in_train() -> None:
    from schelling.eval.effects import experiment_of_item, load_pairs

    f = _experiment_frame()
    f = f.assign(partition=assign_partitions(f, "heldout_experiment", seed=0))
    for p in load_pairs():
        a = set(f[f.item_id == p["arm_a"]].partition)
        b = set(f[f.item_id == p["arm_b"]].partition)
        assert a == b and len(a) == 1
    by_exp = f.assign(exp=f.item_id.map(experiment_of_item())).dropna(subset=["exp"])
    assert by_exp.groupby("exp").partition.nunique().max() == 1
    counts = by_exp.drop_duplicates("exp").partition.value_counts()
    assert counts["val"] == 2 and counts["test"] == 2 and counts["train"] == 4
    assert (f[f.item_id == "QID999"].partition == "train").all()


def test_heldout_experiment_depends_on_seed_and_is_deterministic() -> None:
    f = _experiment_frame()
    chosen = {
        s: tuple(sorted(set(f[assign_partitions(f, "heldout_experiment", s) == "val"].item_id)))
        for s in range(6)
    }
    assert chosen[0] == tuple(
        sorted(set(f[assign_partitions(f, "heldout_experiment", 0) == "val"].item_id))
    )
    assert len(set(chosen.values())) > 1
