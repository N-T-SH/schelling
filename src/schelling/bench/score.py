"""D01-lite metrics. Three questions, each answered with `schelling.eval.metrics`:

1. Crowd: distribution accuracy (1 - TVD) per item, against a human split-half noise ceiling.
   For each random split of the human respondents into halves A and B, the model is compared
   with B and half A is compared with B, so both carry the same sampling noise.
2. Spread: SD ratio of the predicted distribution to the human one, per item (median reported).
3. Person: NLL of each sampled person's real answer, persona prompt against population prompt.

Plus the bias test: standardised effect of each pre-registered contrast (decision 0002 pairs),
human against model, as correlation and sign agreement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from schelling.eval import metrics


def hist(responses: np.ndarray, k: int) -> np.ndarray:
    h = np.bincount(responses, minlength=k).astype(float)
    out: np.ndarray = h / h.sum()
    return out


def split_half(
    model_p: np.ndarray, responses: np.ndarray, n_splits: int, rng: np.random.Generator
) -> tuple[float, float]:
    """(mean 1 - TVD of model vs half B, mean 1 - TVD of half A vs half B)."""
    k = len(model_p)
    acc_m, acc_h = [], []
    for _ in range(n_splits):
        perm = rng.permutation(responses)
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        hb = hist(b, k)
        acc_m.append(1 - metrics.tvd(model_p, hb))
        acc_h.append(1 - metrics.tvd(hist(a, k), hb))
    return float(np.mean(acc_m)), float(np.mean(acc_h))


def bootstrap_mean(
    values: np.ndarray, n_boot: int, rng: np.random.Generator, level: float = 0.95
) -> dict[str, float]:
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def expected_index(p: np.ndarray) -> float:
    return float((p * np.arange(len(p))).sum())


def human_effect(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict[str, float]:
    """Standardised mean difference (B minus A) with a bootstrap interval over persons."""
    sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    if sd == 0:
        return {"d": 0.0, "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    boots = [
        (rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()) / sd for _ in range(n_boot)
    ]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "d": float((b.mean() - a.mean()) / sd),
        "sd": sd,
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def effect_agreement(human_d: np.ndarray, model_d: np.ndarray, real: np.ndarray) -> dict[str, Any]:
    corr = (
        float(np.corrcoef(human_d, model_d)[0, 1])
        if len(human_d) > 2 and human_d.std() > 0 and model_d.std() > 0
        else None
    )
    sign = float(np.mean(np.sign(human_d[real]) == np.sign(model_d[real]))) if real.any() else None
    return {
        "correlation": corr,
        "sign_agreement": sign,
        "n_pairs": len(human_d),
        "n_real": int(real.sum()),
    }


def score_model(
    prompts: pd.DataFrame,
    probs: dict[str, np.ndarray],
    human: pd.DataFrame,
    reference: pd.DataFrame,
    pairs: list[dict[str, str]],
    n_splits: int = 200,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """`prompts`: one row per prompt record. `probs`: prompt_id -> probabilities (missing or
    unparsed prompts fall back to uniform and are counted). `human`: evaluation rows (item_id,
    person_id, response). `reference`: rows for the human effect sizes."""
    rng = np.random.default_rng(seed)
    p = prompts.copy()
    k = p["labels"].map(len)
    have = p["prompt_id"].map(lambda i: i in probs and probs[i] is not None)
    vectors: list[np.ndarray] = [
        np.asarray(probs[i], dtype=float) if ok else np.full(n, 1 / n)
        for i, ok, n in zip(p["prompt_id"], have, k, strict=True)
    ]
    p["p"] = pd.Series(vectors, index=p.index, dtype=object)
    out: dict[str, Any] = {
        "coverage": {arm: float(have[p["arm"] == arm].mean()) for arm in ("population", "persona")}
    }

    resp = {i: g["response"].to_numpy(dtype=int) for i, g in human.groupby("item_id")}
    per_item = []
    for item, g in p.groupby("item_id", sort=True):
        if item not in resp or len(resp[item]) < 4:
            continue
        pop = g.loc[g["arm"] == "population", "p"]
        per = g.loc[g["arm"] == "persona", "p"]
        row: dict[str, Any] = {"item_id": item, "n_human": len(resp[item])}
        h_all = hist(resp[item], int(k[g.index[0]]))
        for arm, vals in (("population", pop), ("persona", per)):
            if len(vals) == 0:
                continue
            m = np.mean(np.stack(list(vals)), axis=0)
            acc, ceil = split_half(m, resp[item], n_splits, rng)
            row[f"{arm}_acc"], row["ceiling"] = acc, ceil
            row[f"{arm}_sd_ratio"] = metrics.sd_ratio(m, h_all)
        per_item.append(row)
    items = pd.DataFrame(per_item)
    out["n_items"] = len(items)
    out["ceiling"] = float(items["ceiling"].mean())
    for arm in ("population", "persona"):
        if f"{arm}_acc" not in items:
            continue
        # Items without an answer for this arm (unparsed replies dropped upstream) are skipped,
        # and the ceiling is taken over the same items.
        have_arm = items[f"{arm}_acc"].notna()
        arm_acc = items.loc[have_arm, f"{arm}_acc"].to_numpy(dtype=float)
        arm_ceiling = float(items.loc[have_arm, "ceiling"].mean())
        out[arm] = {
            "n_items": int(have_arm.sum()),
            "accuracy": bootstrap_mean(arm_acc, n_boot, rng),
            "accuracy_over_ceiling": float(arm_acc.mean() / arm_ceiling),
            "sd_ratio_median": float(np.nanmedian(items[f"{arm}_sd_ratio"])),
        }

    # Person level: real answers of the sampled persons.
    persona = p[p["arm"] == "persona"].set_index(["person_id", "item_id"])["p"]
    population = p[p["arm"] == "population"].set_index("item_id")["p"]
    rows = human[human.set_index(["person_id", "item_id"]).index.isin(persona.index)]
    # Persona against population needs both prompts for the item.
    rows = rows[rows["item_id"].isin(population.index)]
    if len(rows):
        keys = list(zip(rows["person_id"], rows["item_id"], strict=True))
        y = rows["response"].to_numpy(dtype=int)
        lp_per = np.log(
            np.clip([persona[kk][yy] for kk, yy in zip(keys, y, strict=True)], 1e-12, 1)
        )
        lp_pop = np.log(
            np.clip([population[i][yy] for i, yy in zip(rows["item_id"], y, strict=True)], 1e-12, 1)
        )
        by_person = pd.DataFrame({"person": rows["person_id"].to_numpy(), "d": lp_pop - lp_per})
        diff = by_person.groupby("person")["d"].mean().to_numpy()  # NLL persona minus population
        out["individual"] = {
            "n_rows": len(rows),
            "n_persons": len(diff),
            "nll_population": float(-lp_pop.mean()),
            "nll_persona": float(-lp_per.mean()),
            "persona_minus_population": bootstrap_mean(diff, n_boot, rng),
        }

    # Bias test on the pre-registered contrasts whose two arms are both present.
    ref = {i: g["response"].to_numpy(dtype=int) for i, g in reference.groupby("item_id")}
    present = set(p["item_id"])
    eff = []
    for pair in pairs:
        a, b = pair["arm_a"], pair["arm_b"]
        if a not in present or b not in present or a not in ref or b not in ref:
            continue
        h = human_effect(ref[a], ref[b], n_boot, rng)
        rec: dict[str, Any] = {"id": pair["id"], "experiment": pair["experiment"], "human": h}
        for arm in ("population", "persona"):
            ea = [expected_index(x) for x in p.loc[(p["item_id"] == a) & (p["arm"] == arm), "p"]]
            eb = [expected_index(x) for x in p.loc[(p["item_id"] == b) & (p["arm"] == arm), "p"]]
            if ea and eb and h["sd"] > 0:
                rec[arm] = float((np.mean(eb) - np.mean(ea)) / h["sd"])
        eff.append(rec)
    if eff:
        hd = np.array([e["human"]["d"] for e in eff])
        real = np.array([not (e["human"]["ci_low"] <= 0 <= e["human"]["ci_high"]) for e in eff])
        out["effects"] = {"pairs": eff}
        for arm in ("population", "persona"):
            if all(arm in e for e in eff):
                out["effects"][arm] = effect_agreement(hd, np.array([e[arm] for e in eff]), real)
    out["per_item"] = items.to_dict("records")
    return out


def compare_items(
    a: list[dict[str, Any]], b: list[dict[str, Any]], key: str, n_boot: int, seed: int
) -> dict[str, Any]:
    """Paired item bootstrap of model A minus model B on a per-item metric."""
    da = {r["item_id"]: r[key] for r in a if key in r and not np.isnan(r[key])}
    db = {r["item_id"]: r[key] for r in b if key in r and not np.isnan(r[key])}
    shared = sorted(set(da) & set(db))
    diff = np.array([da[i] - db[i] for i in shared])
    return {
        "metric": key,
        "n_items": len(shared),
        **bootstrap_mean(diff, n_boot, np.random.default_rng(seed)),
    }
