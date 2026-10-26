"""Create tests/golden/metrics.json from the fixture benchmark.

Updating the golden file needs `[golden-update]` in the commit message (SPEC "Regression").
"""

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from schelling.cli import app
from schelling.train.config import load_config
from schelling.train.golden import golden_config, run_golden

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    runner = CliRunner()
    fx = ROOT / "tests" / "fixtures"
    for source, raw in (("twin2k", fx / "twin2k"), ("psych101", fx / "psych101_rows.jsonl")):
        assert (
            runner.invoke(
                app,
                [
                    "data",
                    "build",
                    "--source",
                    source,
                    "--raw",
                    str(raw),
                    "--out",
                    str(tmp / "proc"),
                    "--manifests",
                    str(tmp / "man"),
                ],
            ).exit_code
            == 0
        )
    assert (
        runner.invoke(
            app, ["data", "splits", "--processed", str(tmp / "proc"), "--out", str(tmp / "splits")]
        ).exit_code
        == 0
    )
    out = run_golden(load_config(golden_config(tmp, tmp / "proc", tmp / "splits"), []))
    (ROOT / "tests" / "golden" / "metrics.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    print(out)
