"""Command-line entry point. Stubs exit with code 2 until their milestone lands."""

import hashlib
import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from schelling.data import contamination, manifest, splits
from schelling.data.adapters import psych101, twin2k

app = typer.Typer(help="Schelling: data | train | eval | simulate | push", no_args_is_help=True)
data_app = typer.Typer(help="Build datasets, manifests and splits.", no_args_is_help=True)
app.add_typer(data_app, name="data")

_UPSTREAM = {
    "psych101": ("https://huggingface.co/datasets/marcelbinz/Psych-101", "apache-2.0"),
    "twin2k": ("https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500", "cc-by-4.0"),
}


def _not_implemented(name: str, milestone: str) -> None:
    # Exit non-zero so scripts cannot mistake a stub for success.
    typer.echo(f"`schelling {name}` is not implemented yet (planned for {milestone}).", err=True)
    raise typer.Exit(code=2)


@data_app.command("build")
def data_build(
    source: Annotated[str, typer.Option(help="psych101 or twin2k")],
    raw: Annotated[
        Path, typer.Option(help="raw dir (twin2k) or prompts_training.jsonl (psych101)")
    ],
    out: Annotated[Path, typer.Option(help="processed output dir")] = Path("data/processed"),
    manifests: Annotated[Path, typer.Option(help="manifest dir")] = Path("data/manifests"),
    revision: Annotated[
        str, typer.Option(help="upstream revision, recorded in the manifest")
    ] = "unknown",
    max_rows: Annotated[int | None, typer.Option(help="psych101: stop after N rows")] = None,
    keep_fraction: Annotated[
        float, typer.Option(help="psych101: keep this share of participants")
    ] = 1.0,
) -> None:
    """Convert one raw source to unified records and write its manifest."""
    out.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    if source == "psych101":
        rep = psych101.AdapterReport()
        recs = psych101.records_from_rows(psych101.read_rows(raw, max_rows, keep_fraction), rep)
        files = [raw]
        stats = {
            "rows": rep.n_rows,
            "records": len(recs),
            "dropped_experiments": rep.dropped_experiments,
            "max_rows": max_rows,
            "keep_fraction": keep_fraction,
        }
    elif source == "twin2k":
        built = twin2k.build(raw)
        recs = built.records
        built.persons.to_parquet(out / "persons_twin2k.parquet", index=False)
        names = [
            "question_catalog.json",
            "wave1_3_response.csv",
            "wave1_3_response_label.csv",
            "wave4_response.csv",
        ]
        files = [raw / n for n in names]
        stats = {
            "records": len(recs),
            "persons": len(built.persons),
            "unmapped_columns": built.report.unmapped_columns,
            "dropped_values": built.report.dropped_values,
        }
    else:
        raise typer.BadParameter("source must be psych101 or twin2k")
    frame = pd.DataFrame([r.model_dump() for r in recs])
    frame.to_parquet(out / f"records_{source}.parquet", index=False)
    url, lic = _UPSTREAM[source]
    stats["record_ids_sha256"] = hashlib.sha256(
        "\n".join(sorted(frame["record_id"])).encode()
    ).hexdigest()
    m = manifest.build_manifest(source, url, revision, lic, files, stats)
    (manifests / f"{source}.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    typer.echo(f"{source}: {len(recs)} records -> {out}")


@data_app.command("splits")
def data_splits(
    processed: Annotated[Path, typer.Option(help="dir with records_*.parquet")] = Path(
        "data/processed"
    ),
    out: Annotated[Path, typer.Option(help="where to write splits/*.json")] = Path("splits"),
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Write split manifests for every split and a contamination report for heldout_item."""
    from schelling.data.loader import load_records

    frame = load_records(processed)
    # Contamination is measured on heldout_item's raw partition, then those items are removed
    # from its val and test. Compare question plus options: Likert rows share a long
    # instruction, so question text alone would flag every row as a sibling's near duplicate.
    part = splits.assign_partitions(frame, "heldout_item", seed)
    keyed = frame.assign(
        question=frame["question"]
        + " | "
        + frame["options"].map(lambda o: " ".join(o) if o is not None else "")
        + frame["levels"].map(lambda o: " ".join(o) if o is not None else "")
    )
    rep = contamination.contamination_report(
        keyed[part == "train"], keyed[part.isin(["val", "test"])]
    )
    removed = rep["remove_item_ids"]
    for name in splits.SPLIT_NAMES:
        excl = removed if name == "heldout_item" else ()
        path = splits.write_split(splits.split_manifest(frame, name, seed, excl), out)
        typer.echo(f"wrote {path}")
    summary = {
        "threshold": rep["threshold"],
        "n_held_items": rep["n_held_items"],
        "n_exact": len(rep["exact"]),
        "n_near": len(rep["near"]),
        "n_removed": len(removed),
    }
    (out / "contamination_heldout_item.json").write_text(json.dumps(summary, indent=2) + "\n")


@app.command("effects-pool")
def effects_pool(
    run: Annotated[
        list[Path], typer.Option(help="run directory with metrics.json; repeat per fold")
    ],
    out: Annotated[Path, typer.Option(help="where to write the pooled effects.json")],
) -> None:
    """Pool treatment-effect results across leave-one-experiment-out folds."""
    from schelling.eval import provenance
    from schelling.eval.effects import pool_runs

    metrics = [json.loads((r / "metrics.json").read_text()) for r in run]
    pooled = pool_runs([m["effects"] for m in metrics])
    if pooled is None:
        typer.echo("no estimable contrasts in these runs", err=True)
        raise typer.Exit(code=1)
    payload = {
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "created_at": provenance.now_iso(),
        "models": sorted({m["model"] for m in metrics}),
        "source_runs": [
            {
                "split": m["split"],
                "partition": m["partition"],
                "seed": m["seed"],
                "split_manifest_hash": m["split_manifest_hash"],
                "config_hash": m["config_hash"],
                "git_sha": m["git_sha"],
            }
            for m in metrics
        ],
        "effects": pooled,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    typer.echo(f"wrote {out}")


@app.command()
def train(
    config: Annotated[Path, typer.Option(help="YAML training config")],
    set_: Annotated[
        list[str] | None, typer.Option("--set", help="override, e.g. optim.steps=100")
    ] = None,
) -> None:
    """Train and calibrate a model. Reads the train and val partitions only."""
    from schelling.train.config import load_config
    from schelling.train.trainer import train as run_train

    result = run_train(load_config(config, set_ or []))
    typer.echo(
        f"best validation NLL {result.best_val_nll:.4f}; model in {result.run_dir / 'final'}"
    )


@app.command()
def eval(
    model: Annotated[str, typer.Option(help="baseline: b0, b1, b2 or b3")],
    split: Annotated[str, typer.Option()] = "heldout_person",
    partition: Annotated[str, typer.Option(help="train, val or test (test is sealed)")] = "val",
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
    out: Annotated[Path, typer.Option(help="run directory for metrics.json and the report")] = Path(
        "reports/run"
    ),
    seed: Annotated[int, typer.Option()] = 0,
    n_seeds: Annotated[int, typer.Option(help="population sampling seeds")] = 20,
    min_persons: Annotated[
        int, typer.Option(help="min persons per item for population metrics")
    ] = 30,
) -> None:
    """Score a baseline on one split partition and write metrics.json plus a Markdown summary."""
    from schelling.data.loader import SealedPartitionError
    from schelling.eval.report import run_eval

    try:
        run_eval(model, processed, splits_dir, split, partition, out, seed, n_seeds, min_persons)
    except SealedPartitionError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=3) from e
    typer.echo(f"wrote {out / 'metrics.json'}")


@app.command()
def simulate(
    model: Annotated[Path, typer.Option(help="saved model directory")],
    persons: Annotated[Path, typer.Option(help="JSON list of PersonContext")],
    request: Annotated[Path, typer.Option(help="JSON DecisionRequest used as the template")],
    out: Annotated[Path, typer.Option(help="parquet file to write")],
    n_draws: Annotated[int, typer.Option()] = 1,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Sample a synthetic population: n draws per person and question."""
    from schelling.infer.predictor import Predictor
    from schelling.infer.simulate import sample_population
    from schelling.schema import DecisionRequest, PersonContext

    people = [PersonContext.model_validate(p) for p in json.loads(persons.read_text())]
    template = DecisionRequest.model_validate_json(request.read_text())
    frame = sample_population(Predictor.from_pretrained(model), people, template, n_draws, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    typer.echo(f"wrote {len(frame)} draws to {out}")


@app.command()
def push() -> None:
    """Push a model to the Hugging Face Hub."""
    _not_implemented("push", "M7")
