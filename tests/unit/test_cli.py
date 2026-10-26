import pytest
from typer.testing import CliRunner

from schelling.cli import app

pytestmark = pytest.mark.unit
runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("data", "train", "eval", "simulate", "push"):
        assert cmd in result.output


@pytest.mark.parametrize("cmd", ["train", "eval", "simulate", "push"])
def test_stub_commands_fail_loudly_not_silently(cmd: str) -> None:
    result = runner.invoke(app, [cmd])
    assert result.exit_code == 2  # "not implemented" must never look like success
    assert "not implemented" in result.output.lower()


def test_data_group_lists_build_and_splits() -> None:
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "build" in result.output and "splits" in result.output
