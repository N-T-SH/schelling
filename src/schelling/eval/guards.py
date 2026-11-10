"""Guard metrics (decision 0005): pass/fail checks read from existing metrics.json files.

A guard never changes a metric. The population TVD guard is recorded but not binding until
the human decides otherwise.
"""

from __future__ import annotations

from typing import Any

ECE_LIMIT = 0.05
SD_RATIO_BAND = (0.85, 1.15)
PERMUTATION_LIMIT = 0.01
SOURCE = "twin2k"


def _guard(
    value: float, passed: bool, *, binding: bool, **extra: float | list[float]
) -> dict[str, Any]:
    return {"value": value, "pass": bool(passed), "binding": binding, **extra}


def compute_guards(run: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    if (run["split"], run["partition"], run["split_manifest_hash"]) != (
        reference["split"],
        reference["partition"],
        reference["split_manifest_hash"],
    ):
        raise ValueError("the reference must be evaluated on the same split, partition and rows")
    ind, ref_ind = (
        run["individual"]["by_source"][SOURCE],
        reference["individual"]["by_source"][SOURCE],
    )
    pop, ref_pop = run["population"][SOURCE]["summary"], reference["population"][SOURCE]["summary"]
    sd = pop["sd_ratio"]["mean"]
    guards = {
        "person_nll": _guard(
            ind["nll"], ind["nll"] <= ref_ind["nll"], binding=True, reference=ref_ind["nll"]
        ),
        "population_tvd": _guard(
            pop["tvd"]["mean"],
            pop["tvd"]["mean"] <= ref_pop["tvd"]["mean"],
            binding=False,
            reference=ref_pop["tvd"]["mean"],
        ),
        "ece": _guard(ind["ece"], ind["ece"] <= ECE_LIMIT, binding=True, limit=ECE_LIMIT),
        "sd_ratio": _guard(
            sd, SD_RATIO_BAND[0] <= sd <= SD_RATIO_BAND[1], binding=True, band=list(SD_RATIO_BAND)
        ),
        "permutation": _guard(
            run["robustness"]["permutation_max_abs_dp"],
            run["robustness"]["permutation_max_abs_dp"] <= PERMUTATION_LIMIT,
            binding=True,
            limit=PERMUTATION_LIMIT,
        ),
    }
    return {
        "guards": guards,
        "binding_pass": all(g["pass"] for g in guards.values() if g["binding"]),
        "all_pass": all(g["pass"] for g in guards.values()),
    }
