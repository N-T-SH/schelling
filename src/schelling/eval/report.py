"""`schelling eval`: fit a baseline, score it on a split partition, write the report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from schelling.baselines import make_baseline
from schelling.data.loader import (
    ALLOW_TEST_ENV,
    SealedPartitionError,
    load_records,
    partition_frame,
    select_partition,
)
from schelling.data.splits import split_manifest
from schelling.eval import metrics as m
from schelling.eval import population as pop
from schelling.eval import provenance
from schelling.eval.efficiency import latency
from schelling.eval.retest import retest_fraction
from schelling.eval.robustness import permutation_invariance
from schelling.eval.schema import MetricsFile

NOT_RUN = [
    "paraphrase_invariance (needs paraphrased items, M5)",
    "memorisation_probe (needs paraphrased items, M5)",
    "treatment_effects (needs a pre-registered pairing table)",
    "regression_sign_agreement (needs pre-registered models)",
]
SUBGROUP_COLUMNS = ("sex", "age_band", "party_id")
CORRELATION_ITEMS = 60
CORRELATION_COVERAGE = 0.9


def _individual(probs: list[np.ndarray], y: np.ndarray) -> dict[str, Any]:
    if len(y) == 0:
        return {"n": 0}
    conf, correct = m.top_label(probs, y)
    slope, intercept = m.calibration_slope_intercept(conf, correct)
    return {
        "n": len(y),
        "nll": m.nll(probs, y),
        "brier": m.brier(probs, y),
        "accuracy": m.accuracy(probs, y),
        "ece": m.ece(conf, correct),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _correlation_items(frame: pd.DataFrame) -> list[str]:
    """Pre-registered: score items answered in wave 1 by >= 90% of persons, first 60 by id."""
    s = frame[(frame["qtype"] == "score") & (frame["wave"] == 1)]
    if s.empty:
        return []
    coverage = s.groupby("item_id")["person_id"].nunique() / frame["person_id"].nunique()
    return sorted(coverage[coverage >= CORRELATION_COVERAGE].index)[:CORRELATION_ITEMS]


def run_eval(
    model: str,
    processed: Path,
    splits_dir: Path,
    split: str,
    partition: str,
    out: Path,
    seed: int = 0,
    n_seeds: int = 20,
    min_persons: int = 30,
) -> dict[str, Any]:
    if partition == "test" and os.environ.get(ALLOW_TEST_ENV) != "1":
        raise SealedPartitionError(
            f"the test partition is sealed; only the human sets {ALLOW_TEST_ENV}=1"
        )
    frame = load_records(processed)
    persons_path = processed / "persons_twin2k.parquet"
    persons = (
        pd.read_parquet(persons_path) if persons_path.exists() else pd.DataFrame({"person_id": []})
    )
    parted = partition_frame(frame, split, splits_dir)
    eval_df = select_partition(parted, partition).reset_index(drop=True)
    if eval_df.empty:
        raise ValueError(f"no rows in {split}/{partition}")
    if split == "retest":
        # The split has no train partition; fit on wave-1 answers of persons we do not score.
        fit_df = parted[
            (parted["wave"] == 1) & ~parted["person_id"].isin(set(eval_df["person_id"]))
        ]
    else:
        fit_df = select_partition(parted, "train")
    manifest_path = splits_dir / f"{split}.json"
    manifest_hash = (
        json.loads(manifest_path.read_text())["manifest_hash"]
        if manifest_path.exists()
        else split_manifest(frame, split, 0)["manifest_hash"]
    )

    baseline = make_baseline(model, seed).fit(fit_df, persons)
    probs = baseline.predict_proba(eval_df)
    y = eval_df["response"].to_numpy()
    sources = sorted(eval_df["source"].unique())

    individual: dict[str, Any] = {
        "overall": _individual(probs, y),
        "by_source": {},
        "by_source_qtype": {},
    }
    reliability: dict[str, Any] = {}
    for src in sources:
        mask = (eval_df["source"] == src).to_numpy()
        sub = [p for p, k in zip(probs, mask, strict=True) if k]
        individual["by_source"][src] = _individual(sub, y[mask])
        conf, correct = m.top_label(sub, y[mask])
        reliability[src] = m.reliability_bins(conf, correct)
        for qt in sorted(eval_df.loc[mask, "qtype"].unique()):
            qmask = mask & (eval_df["qtype"] == qt).to_numpy()
            individual["by_source_qtype"][f"{src}__{qt}"] = _individual(
                [p for p, k in zip(probs, qmask, strict=True) if k], y[qmask]
            )
    if hasattr(baseline, "n_unseen"):
        individual["unseen_item_rows"] = int(baseline.n_unseen)

    # Population metrics use one observation per person and item: wave-4 repeats are excluded.
    pop_mask = (eval_df["wave"] != 4).to_numpy()
    draws = pop.sample_draws(probs, n_seeds, seed)
    population: dict[str, Any] = {}
    for src in sources:
        mask = pop_mask & (eval_df["source"] == src).to_numpy()
        sub_frame = eval_df[mask]
        sub_probs = [p for p, k in zip(probs, mask, strict=True) if k]
        res = pop.population_from_draws(sub_frame, sub_probs, draws[mask], min_persons)
        if src == "twin2k" and len(persons.columns) > 1 and res["n_items"] > 0:
            cols = [c for c in SUBGROUP_COLUMNS if c in persons.columns]
            res["subgroups"] = pop.subgroup_fidelity(
                sub_frame, sub_probs, persons, cols, n_seeds, seed, min_persons, draws=draws[mask]
            )
            items = _correlation_items(sub_frame)
            res["correlation_items"] = len(items)
            res["correlation_frobenius"] = (
                pop.correlation_distance(sub_frame, draws[mask], items) if len(items) >= 5 else None
            )
        population[src] = res

    retest = None
    if split == "retest":
        w4 = (eval_df["wave"] == 4).to_numpy()
        retest = retest_fraction(
            eval_df[w4],
            [p for p, k in zip(probs, w4, strict=True) if k],
            parted[parted["wave"] == 1],
        )

    config = {
        "model": model,
        "split": split,
        "partition": partition,
        "n_seeds": n_seeds,
        "min_persons": min_persons,
        "fit_rows": len(fit_df),
        "population_excludes": "wave 4",
        "subgroup_columns": list(SUBGROUP_COLUMNS),
    }
    metrics = {
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "config_hash": provenance.config_hash({**config, "seed": seed}),
        "split_manifest_hash": manifest_hash,
        "seed": seed,
        "created_at": provenance.now_iso(),
        "model": model,
        "split": split,
        "partition": partition,
        "config": config,
        "n_rows": len(eval_df),
        "individual": individual,
        "population": population,
        "retest": retest,
        "robustness": {
            "permutation_max_abs_dp": permutation_invariance(baseline, eval_df, seed=seed)
        },
        "efficiency": latency(baseline, eval_df, seed=seed),
        "not_run": NOT_RUN,
    }
    MetricsFile.model_validate(metrics)

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (out / "reliability.json").write_text(json.dumps(reliability, indent=2) + "\n")
    pd.DataFrame(
        {"record_id": eval_df["record_id"], "probs": [p.tolist() for p in probs]}
    ).to_parquet(out / "predictions.parquet", index=False)
    (out / "summary.md").write_text(_summary(metrics))
    return metrics


def _tag(value: float | None, path: str, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f} <!-- src: metrics.json#{path} -->"


def _summary(mt: dict[str, Any]) -> str:
    lines = [
        f"# {mt['model']} on {mt['split']} ({mt['partition']})",
        "",
        f"Code revision {mt['git_sha'][:10]}, dirty tree: {mt['dirty']}, seed {mt['seed']}.",
        "",
        "## Individual level",
        "",
        "| source | rows | NLL | Brier | accuracy | ECE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for src, v in mt["individual"]["by_source"].items():
        p = f"individual.by_source.{src}"
        lines.append(
            f"| {src} | {v['n']} | {_tag(v.get('nll'), p + '.nll')} | {_tag(v.get('brier'), p + '.brier')} | "
            f"{_tag(v.get('accuracy'), p + '.accuracy')} | {_tag(v.get('ece'), p + '.ece')} |"
        )
    lines += [
        "",
        "## Population level (mean over seeds)",
        "",
        "| source | items | TVD | SD ratio | entropy ratio | mode-collapse rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for src, v in mt["population"].items():
        s = v["summary"]
        if s is None:
            lines.append(f"| {src} | {v['n_items']} | n/a | n/a | n/a | n/a |")
            continue
        p = f"population.{src}.summary"
        lines.append(
            f"| {src} | {v['n_items']} | {_tag(s['tvd']['mean'], p + '.tvd.mean')} | "
            f"{_tag(s['sd_ratio']['mean'], p + '.sd_ratio.mean')} | "
            f"{_tag(s['entropy_ratio']['mean'], p + '.entropy_ratio.mean')} | "
            f"{_tag(s['mode_collapse_rate']['mean'], p + '.mode_collapse_rate.mean')} |"
        )
    if mt["retest"]:
        r = mt["retest"]
        lines += [
            "",
            "## Test-retest",
            "",
            f"Fraction of ceiling: {_tag(r['fraction_of_ceiling'], 'retest.fraction_of_ceiling')} "
            f"(ceiling {_tag(r['ceiling'], 'retest.ceiling')}, model {_tag(r['model_agreement'], 'retest.model_agreement')}).",
        ]
    lines += [
        "",
        "## Robustness and efficiency",
        "",
        f"Option-permutation max change in probability: {_tag(mt['robustness']['permutation_max_abs_dp'], 'robustness.permutation_max_abs_dp')}.",
        f"p50 latency in ms on CPU: {_tag(mt['efficiency']['p50_ms'], 'efficiency.p50_ms', 3)}.",
        "",
        "## Not run",
        "",
    ] + [f"- {x}" for x in mt["not_run"]]
    return "\n".join(lines) + "\n"
