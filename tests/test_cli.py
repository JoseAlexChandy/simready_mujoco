import json

from typer.testing import CliRunner

from simready_mujoco import __version__
from simready_mujoco.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_inspect_json(cabinet_usd):
    result = runner.invoke(app, ["inspect", str(cabinet_usd), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data["bodies"]) == 3


def test_inspect_table(cabinet_usd):
    result = runner.invoke(app, ["inspect", str(cabinet_usd)])
    assert result.exit_code == 0, result.output
    assert "Lost by the baseline" in result.output


def test_missing_file_is_a_clear_error():
    result = runner.invoke(app, ["inspect", "nope.usd"])
    assert result.exit_code != 0
