import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_report.py"


def _run(report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(report)], capture_output=True, text=True
    )


@pytest.fixture()
def metrics(tmp_path: Path) -> Path:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"individual": {"nll": 1.2345}, "per_seed": {"tvd": [0.1, 0.25]}}))
    return path


def test_matching_tagged_numbers_pass(tmp_path: Path, metrics: Path) -> None:
    md = tmp_path / "r.md"
    md.write_text(
        f"NLL is 1.2345 <!-- src: {metrics}#individual.nll --> and 0.25 <!-- src: {metrics}#per_seed.tvd.1 -->.\n"
    )
    assert _run(md).returncode == 0


def test_rounded_number_matching_displayed_precision_passes(tmp_path: Path, metrics: Path) -> None:
    md = tmp_path / "r.md"
    md.write_text(f"NLL 1.23 <!-- src: {metrics}#individual.nll -->\n")
    assert _run(md).returncode == 0


def test_wrong_number_fails(tmp_path: Path, metrics: Path) -> None:
    md = tmp_path / "r.md"
    md.write_text(f"NLL is 1.3 <!-- src: {metrics}#individual.nll -->\n")
    res = _run(md)
    assert res.returncode == 1 and "individual.nll" in res.stdout


def test_untagged_metric_like_number_fails(tmp_path: Path, metrics: Path) -> None:
    md = tmp_path / "r.md"
    md.write_text("The model reaches 0.42 NLL.\n")
    assert _run(md).returncode == 1


def test_numbers_in_code_urls_and_shas_are_not_flagged(tmp_path: Path) -> None:
    md = tmp_path / "r.md"
    md.write_text("See https://x.org/v1.2.3 at `0.5` commit abc1234.\n```\nloss 0.123\n```\n")
    assert _run(md).returncode == 0


def test_missing_json_path_fails(tmp_path: Path, metrics: Path) -> None:
    md = tmp_path / "r.md"
    md.write_text(f"x 1.0 <!-- src: {metrics}#nope.nothing -->\n")
    assert _run(md).returncode == 1
