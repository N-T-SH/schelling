"""Command-line entry point. Stubs exit with code 2 until their milestone lands."""

import typer

app = typer.Typer(help="Schelling: data | train | eval | simulate | push", no_args_is_help=True)


def _not_implemented(name: str, milestone: str) -> None:
    # Exit non-zero so scripts cannot mistake a stub for success.
    typer.echo(f"`schelling {name}` is not implemented yet (planned for {milestone}).", err=True)
    raise typer.Exit(code=2)


@app.command()
def data() -> None:
    """Build datasets, manifests and splits."""
    _not_implemented("data", "M1")


@app.command()
def train() -> None:
    """Train a model from a YAML config."""
    _not_implemented("train", "M4")


@app.command()
def eval() -> None:
    """Evaluate a model or baseline on a split."""
    _not_implemented("eval", "M2")


@app.command()
def simulate() -> None:
    """Sample a synthetic population."""
    _not_implemented("simulate", "M3")


@app.command()
def push() -> None:
    """Push a model to the Hugging Face Hub."""
    _not_implemented("push", "M7")
