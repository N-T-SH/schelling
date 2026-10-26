from pathlib import Path

from schelling.baselines.base import Baseline
from schelling.baselines.demo_cell import DemoCellMean
from schelling.baselines.gbm import GradientBoosted
from schelling.baselines.item_mean import ItemMean
from schelling.baselines.uniform import Uniform


def make_baseline(name: str, seed: int = 0) -> Baseline:
    if name == "b0":
        return Uniform()
    if name == "b1":
        return ItemMean()
    if name == "b2":
        return DemoCellMean()
    if name == "b3":
        return GradientBoosted(seed=seed)
    if Path(name).is_dir():  # a saved SchellingModel
        from schelling.infer.baseline_adapter import ModelBaseline

        return ModelBaseline.from_path(name)
    raise ValueError(f"unknown baseline {name!r}; B4 to B6 arrive in M6")


__all__ = ["Baseline", "DemoCellMean", "GradientBoosted", "ItemMean", "Uniform", "make_baseline"]
